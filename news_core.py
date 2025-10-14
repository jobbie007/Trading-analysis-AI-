from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

try:
    from bs4 import BeautifulSoup 
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None  

# Optional DDG Search client
try:
    from duckduckgo_search import DDGS  
except Exception:  # pragma: no cover
    DDGS = None 

# Optional Gemini (google-generativeai)
try:
    import google.generativeai as genai  
    GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover
    genai = None 
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)


# ------------------------- Settings (env-based) -------------------------
class Settings:
    """Minimal environment-based configuration.
    Keeps this module dependency-free (no pydantic requirement).
    """

    def __init__(self) -> None:
        # Gemini
        self.gemini_api_keys: List[str] = [
            k.strip()
            for k in [
                os.getenv("GEMINI_API_KEY", ""),
                os.getenv("GEMINI_API_KEY2", ""),
                os.getenv("GEMINI_API_KEY3", ""),
                os.getenv("GEMINI_API_KEY4", ""),
                os.getenv("GEMINI_API_KEY5", ""),
                os.getenv("GEMINI_API_KEY6", ""),
                os.getenv("GEMINI_API_KEY7", ""),
            ]
            if k and k.strip()
        ]
        self.gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

        # News providers
        self.gnews_api_keys: List[str] = [
            k.strip()
            for k in [
                os.getenv("GNEWS_API_KEY1", ""),
                os.getenv("GNEWS_API_KEY2", ""),
                os.getenv("GNEWS_API_KEY3", ""),
                os.getenv("GNEWS_API_KEY4", ""),
            ]
            if k and k.strip()
        ]
        self.newsapi_key: Optional[str] = os.getenv("NEWSAPI_KEY")
        self.cryptopanic_api_key: Optional[str] = os.getenv("CRYPTOPANIC_API_KEY")

        # Local LLM (OpenAI-compatible)
        self.local_llm_base_url: str = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1").strip()
        self.local_llm_api_key: str = os.getenv("LOCAL_LLM_API_KEY", "not-needed").strip()
        self.local_model_name: str = os.getenv("MODEL_NAME", os.getenv("LOCAL_LLM_MODEL", "local-model")).strip()

        # Misc
        try:
            self.temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
        except Exception:
            self.temperature = 0.1

        # HTTP timeout for OpenAI-compatible calls (seconds)
        for key in ("LLM_HTTP_TIMEOUT", "LOCAL_LLM_TIMEOUT", "LLM_TIMEOUT"):
            raw = os.getenv(key)
            if raw:
                try:
                    self.local_llm_timeout = max(float(raw), 1.0)
                    break
                except Exception:
                    continue
        else:
            self.local_llm_timeout = 180.0
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.disable_api_keys: bool = (os.getenv("DISABLE_API_KEYS", "false").strip().lower() in ("1", "true", "yes", "on"))


settings = Settings()


# ------------------------------ Data model ------------------------------
@dataclass
class NewsArticle:
    title: str
    url: str
    description: str
    published_at: str
    source: str
    content: str = ""
    sentiment_score: float = 0.0
    sentiment_label: str = "NEUTRAL"
    relevance_score: float = 0.0
    ai_summary: str = ""
    analysis: Dict[str, Any] = field(default_factory=dict)


# ------------------------------ Fetcher ---------------------------------
class NewsFetcher:
    _gnews_key_index = 0
    _serper_key_index = 0

    def __init__(self, gnews_keys: List[str], newsapi_key: Optional[str], cryptopanic_key: Optional[str], disable_api_keys: bool = False):
        self.gnews_keys = gnews_keys if not disable_api_keys else []
        self.newsapi_key = newsapi_key if not disable_api_keys else None
        self.cryptopanic_key = cryptopanic_key if not disable_api_keys else None
        self._last_attempts: List[tuple[str, int]] = []
        self._last_attempts_summary = ""
        # Diagnostics
        self._gnews_last_status: Optional[int] = None
        self._gnews_last_msg = ""
        self._gnews_last_q = ""
        self._gnews_last_from = ""
        self._gnews_last_in = ""
        self._gnews_last_sortby = ""
        self._serper_last_status: Optional[int] = None
        self._serper_last_msg = ""

        # Serper keys (rotation)
        self._serper_keys: List[str] = []
        if not disable_api_keys:
            for k in ("SERPER_API_KEY", "SERPER_API_KEY2", "SERPER_API_KEY3"):
                v = (os.getenv(k) or "").strip()
                if v:
                    self._serper_keys.append(v)

        # DDGS availability
        self._ddgs_cls = DDGS
        if self._ddgs_cls is None:
            logging.getLogger(__name__).warning(
                "DuckDuckGo search client not available. Install 'duckduckgo_search' to enable DDGS fallback."
            )

    # --- Key rotation helpers ---
    def get_current_gnews_key(self) -> Optional[str]:
        if not self.gnews_keys:
            return None
        return self.gnews_keys[self._gnews_key_index % len(self.gnews_keys)]

    def rotate_gnews_key(self):
        if self.gnews_keys:
            self._gnews_key_index = (self._gnews_key_index + 1) % len(self.gnews_keys)
            logger.info("Rotated to GNews key index %s", self._gnews_key_index)

    def get_current_serper_key(self) -> Optional[str]:
        if not self._serper_keys:
            return None
        return self._serper_keys[self._serper_key_index % len(self._serper_keys)]

    def rotate_serper_key(self):
        if self._serper_keys:
            self._serper_key_index = (self._serper_key_index + 1) % len(self._serper_keys)
            logger.info("Rotated to Serper key index %s", self._serper_key_index)

    # --- Public API ---
    def fetch_news(self, query: str, days_back: int = 7, max_articles: int = 10) -> List[NewsArticle]:
        all_articles: List[NewsArticle] = []
        attempts: List[tuple[str, int]] = []

        if len(all_articles) < max_articles:
            needed = max_articles - len(all_articles)
            gnews_items = self.fetch_gnews_articles(query, needed, days_back)
            attempts.append(("GNews", len(gnews_items)))
            all_articles.extend(gnews_items)

        if len(all_articles) < max_articles:
            if self.newsapi_key:
                needed = max_articles - len(all_articles)
                newsapi_items = self.fetch_newsapi_articles(query, needed, days_back)
                attempts.append(("NewsAPI", len(newsapi_items)))
                all_articles.extend(newsapi_items)
            else:
                attempts.append(("NewsAPI", 0))

        if self.cryptopanic_key and query.upper() in {"BTC", "ETH", "SOL"} and len(all_articles) < max_articles:
            needed = max_articles - len(all_articles)
            cp_items = self.fetch_cryptopanic_articles(query, needed)
            attempts.append(("CryptoPanic", len(cp_items)))
            all_articles.extend(cp_items)

        if len(all_articles) < max_articles:
            if self._serper_keys:
                needed = max_articles - len(all_articles)
                serper_items = self.fetch_serper_news(query, needed, days_back)
                attempts.append(("Serper", len(serper_items)))
                all_articles.extend(serper_items)
            else:
                attempts.append(("Serper", 0))

        if self._ddgs_cls and len(all_articles) < max_articles:
            needed = max_articles - len(all_articles)
            ddgs_items = self.fetch_ddgs_articles(query, days_back, needed)
            attempts.append(("DDGS", len(ddgs_items)))
            all_articles.extend(ddgs_items)

        # Deduplicate by title
        unique_articles: List[NewsArticle] = []
        seen = set()
        for a in all_articles:
            t = (a.title or "").strip().lower()
            if t and t not in seen:
                unique_articles.append(a)
                seen.add(t)

        self._last_attempts = attempts
        try:
            parts = []
            for n, c in attempts:
                if n == "GNews" and c == 0 and self._gnews_last_status is not None:
                    detail = f"GNews:0 (HTTP {self._gnews_last_status})"
                    if self._gnews_last_q:
                        detail += f" [q='{self._gnews_last_q}'"
                        if self._gnews_last_from:
                            detail += f", from={self._gnews_last_from}"
                        if self._gnews_last_in:
                            detail += f", in={self._gnews_last_in}"
                        if self._gnews_last_sortby:
                            detail += f", sortby={self._gnews_last_sortby}"
                        detail += "]"
                    parts.append(detail)
                else:
                    parts.append(f"{n}:{c}")
            self._last_attempts_summary = ", ".join(parts) or "no-attempts"
        except Exception:
            pass

        return unique_articles[:max_articles]

    def get_last_attempts_summary(self) -> str:
        try:
            return self._last_attempts_summary or ", ".join(f"{n}:{c}" for n, c in (self._last_attempts or []))
        except Exception:
            return ""

    # --- Providers ---
    def _expand_query_for_news(self, query: str) -> str:
        q = (query or "").strip()
        if not q:
            return q
        q_upper = q.upper()
        for suf in ("/USDT", "/USD", "-USDT", "-USD"):
            if q_upper.endswith(suf):
                q = q[: -len(suf)]
                q_upper = q_upper[: -len(suf)]
                break
        for suf in ("USDT", "USD", "USDC"):
            if q_upper.endswith(suf) and len(q_upper) > len(suf) + 1:
                q = q[: -len(suf)]
                q_upper = q_upper[: -len(suf)]
                break
        mapping = {
            "BTC": "Bitcoin OR BTC",
            "ETH": "Ethereum OR ETH",
            "SOL": "Solana OR SOL",
            "XRP": "XRP OR Ripple",
            "ADA": "Cardano OR ADA",
            "DOGE": "Dogecoin OR DOGE",
            "BNB": "Binance Coin OR BNB",
            "AVAX": "Avalanche OR AVAX",
            "AAPL": "Apple OR AAPL",
            "MSFT": "Microsoft OR MSFT",
            "TSLA": "Tesla OR TSLA",
            "NVDA": "Nvidia OR NVDA",
            "AMZN": "Amazon OR AMZN",
            "GOOG": "Google OR Alphabet OR GOOG OR GOOGL",
            "GOOGL": "Google OR Alphabet OR GOOG OR GOOGL",
            "META": "Meta OR Facebook OR META",
        }
        if q_upper in mapping:
            return mapping[q_upper]
        if q.isupper() and 2 <= len(q) <= 5:
            return f"{q} OR {q} stock OR {q} crypto"
        return q

    def fetch_gnews_articles(self, query: str, max_results: int, days_back: int = 7) -> List[NewsArticle]:
        key = self.get_current_gnews_key()
        if not key:
            return []
        url = "https://gnews.io/api/v4/search"
        q = self._expand_query_for_news(query)
        from_date = (datetime.utcnow() - timedelta(days=max(1, int(days_back or 1)))).strftime("%Y-%m-%d")
        params = {
            "q": q,
            "token": key,
            "lang": "en",
            "max": max(1, min(int(max_results or 10), 10)),
            "from": from_date,
            "sortby": "publishedAt",
            "in": "title,description,content",
        }
        try:
            self._gnews_last_status = None
            self._gnews_last_msg = ""
            self._gnews_last_q = q
            self._gnews_last_from = from_date
            self._gnews_last_in = params.get("in", "")
            self._gnews_last_sortby = params.get("sortby", "")
            resp = requests.get(url, params=params, timeout=10)
            self._gnews_last_status = resp.status_code
            if resp.status_code == 429:
                logger.warning("GNews rate limited (429). Rotating API key.")
                self.rotate_gnews_key()
                return []
            if resp.status_code == 200:
                data = resp.json() or {}
                articles = [
                    NewsArticle(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        description=item.get("description", ""),
                        published_at=item.get("publishedAt", ""),
                        source=(item.get("source") or {}).get("name", "GNews"),
                    )
                    for item in data.get("articles", [])
                ]
                if not articles and " OR " in q:
                    simple_q = q.split(" OR ", 1)[0].strip()
                    if simple_q and simple_q != q:
                        params2 = dict(params)
                        params2["q"] = simple_q
                        self._gnews_last_q = simple_q
                        try:
                            resp2 = requests.get(url, params=params2, timeout=10)
                            self._gnews_last_status = resp2.status_code
                            if resp2.status_code == 200:
                                data2 = resp2.json() or {}
                                articles = [
                                    NewsArticle(
                                        title=i.get("title", ""),
                                        url=i.get("url", ""),
                                        description=i.get("description", ""),
                                        published_at=i.get("publishedAt", ""),
                                        source=(i.get("source") or {}).get("name", "GNews"),
                                    )
                                    for i in data2.get("articles", [])
                                ]
                        except requests.RequestException:
                            pass
                if not articles:
                    params3 = dict(params)
                    params3.pop("from", None)
                    params3.pop("in", None)
                    params3.pop("sortby", None)
                    broaden_q = q
                    if q.isupper() and 2 <= len(q) <= 5 and " " not in q:
                        broaden_q = f"{q} stock"
                    params3["q"] = broaden_q
                    try:
                        resp3 = requests.get(url, params=params3, timeout=10)
                        self._gnews_last_status = resp3.status_code
                        if resp3.status_code == 200:
                            data3 = resp3.json() or {}
                            articles = [
                                NewsArticle(
                                    title=i.get("title", ""),
                                    url=i.get("url", ""),
                                    description=i.get("description", ""),
                                    published_at=i.get("publishedAt", ""),
                                    source=(i.get("source") or {}).get("name", "GNews"),
                                )
                                for i in data3.get("articles", [])
                            ]
                    except requests.RequestException:
                        pass
                return articles
            else:
                text = (resp.text or "")[:200].replace("\n", " ")
                self._gnews_last_msg = text
                logger.warning("GNews HTTP %s: %s", resp.status_code, text)
        except requests.RequestException as e:
            self._gnews_last_status = -1
            self._gnews_last_msg = str(e)
            logger.error("GNews request failed: %s", e)
        return []

    def fetch_serper_news(self, query: str, max_results: int, days_back: int = 7) -> List[NewsArticle]:
        key = self.get_current_serper_key()
        if not key:
            return []
        url = "https://google.serper.dev/news"
        q = self._expand_query_for_news(query)
        tbs = "qdr:d" if days_back <= 3 else ("qdr:w" if days_back <= 10 else "qdr:m")
        body = {
            "q": q,
            "num": max(1, min(int(max_results or 10), 10)),
            "gl": "us",
            "hl": "en",
            "tbs": tbs,
        }
        headers = {"X-API-KEY": key, "Content-Type": "application/json"}
        results: List[NewsArticle] = []
        try:
            r = requests.post(url, headers=headers, json=body, timeout=15)
            self._serper_last_status = r.status_code
            if r.status_code in (401, 403, 429, 500, 502, 503):
                self._serper_last_msg = (r.text or "")[:200].replace("\n", " ")
                self.rotate_serper_key()
                return []
            if r.status_code != 200:
                self._serper_last_msg = (r.text or "")[:200].replace("\n", " ")
                return []
            data = r.json() or {}
            items = data.get("news") or []
            for it in items:
                try:
                    src = it.get("source") or "Serper"
                    results.append(
                        NewsArticle(
                            title=it.get("title", ""),
                            url=it.get("link", "") or it.get("url", ""),
                            description=it.get("snippet", "") or it.get("description", ""),
                            published_at=it.get("date", ""),
                            source=src,
                        )
                    )
                except Exception:
                    continue
        except requests.RequestException as e:
            self._serper_last_status = -1
            self._serper_last_msg = str(e)
        return results

    def fetch_newsapi_articles(self, query: str, max_results: int, days_back: int) -> List[NewsArticle]:
        if not self.newsapi_key:
            return []
        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        params = {
            "q": query,
            "apiKey": self.newsapi_key,
            "language": "en",
            "pageSize": max_results,
            "from": from_date,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json() or {}
                if isinstance(data, dict) and data.get("status") == "error":
                    logger.warning("NewsAPI error: %s: %s", data.get("code"), data.get("message"))
                    return []
                return [
                    NewsArticle(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        description=item.get("description", ""),
                        published_at=item.get("publishedAt", ""),
                        source=(item.get("source") or {}).get("name", "NewsAPI"),
                    )
                    for item in data.get("articles", [])
                ]
            else:
                text = (resp.text or "")[:200].replace("\n", " ")
                logger.warning("NewsAPI HTTP %s: %s", resp.status_code, text)
        except requests.RequestException as e:
            logger.error("NewsAPI request failed: %s", e)
        return []

    def fetch_cryptopanic_articles(self, currency: str, max_results: int) -> List[NewsArticle]:
        if not self.cryptopanic_key:
            return []
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {"auth_token": self.cryptopanic_key, "currencies": currency.upper(), "kind": "news"}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json() or {}
                return [
                    NewsArticle(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        description=item.get("title", ""),
                        published_at=item.get("created_at", ""),
                        source="CryptoPanic",
                    )
                    for item in data.get("results", [])
                ][:max_results]
            else:
                text = (resp.text or "")[:200].replace("\n", " ")
                logger.warning("CryptoPanic HTTP %s: %s", resp.status_code, text)
        except requests.RequestException as e:
            logger.error("CryptoPanic request failed: %s", e)
        return []

    def fetch_ddgs_articles(self, query: str, days_back: int, max_results: int) -> List[NewsArticle]:
        if not self._ddgs_cls:
            return []
        timelimit = "d" if days_back <= 3 else "w" if days_back <= 10 else "m"

        def _ddgs_iter(ddgs):
            try:
                return ddgs.news(query, region="wt-wt", safesearch="moderate", time=timelimit, max_results=max_results)
            except TypeError:
                return ddgs.news(query, region="wt-wt", safesearch="moderate", timelimit=timelimit, max_results=max_results)

        results: List[NewsArticle] = []
        try:
            with self._ddgs_cls() as ddgs:  # type: ignore[operator]
                for r in _ddgs_iter(ddgs):
                    try:
                        src = r.get("source") or urlparse(r.get("url", "")).netloc
                        published = r.get("date") or r.get("published") or r.get("published_at") or ""
                        desc = r.get("body") or r.get("excerpt") or r.get("description") or ""
                        results.append(
                            NewsArticle(
                                title=r.get("title", ""),
                                url=r.get("url", ""),
                                description=desc,
                                published_at=published,
                                source=src,
                            )
                        )
                    except Exception:
                        continue
        except Exception as e:
            logger.error("DDGS news search failed: %s", e)
        return results

    def scrape_article_content(self, url: str) -> Optional[str]:
        if not BeautifulSoup:
            return None
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in ["script", "style", "nav", "footer", "header", "aside"]:
                for e in soup.find_all(tag):
                    e.decompose()
            content_selectors = ["article", ".article-content", ".post-content", "main"]
            content = ""
            for sel in content_selectors:
                element = soup.select_one(sel)
                if element:
                    content = element.get_text(" ", strip=True)
                    break
            if not content:
                content = " ".join([p.get_text(" ", strip=True) for p in soup.find_all("p")])
            return content[:8000] if content else None
        except requests.RequestException:
            return None


# ------------------------------ Analyst ---------------------------------
class LLMClientManager:
    def __init__(self, gemini_keys: List[str], gemini_model: str):
        self.gemini_model_name = gemini_model
        self.gemini_clients: List[Dict[str, Any]] = []
        self.current_gemini_index: int = 0
        self._initialize_gemini(gemini_keys)

    def _initialize_gemini(self, api_keys: List[str]):
        if not GEMINI_AVAILABLE or not api_keys:
            logger.info("Gemini client not initialized (library missing or no API keys). Running without Gemini is supported.")
            return
        for i, api_key in enumerate(api_keys):
            try:
                genai.configure(api_key=api_key)  # type: ignore[attr-defined]
                client = genai.GenerativeModel(self.gemini_model_name)  # type: ignore[attr-defined]
                self.gemini_clients.append({"client": client, "api_key": api_key, "quota_exceeded": False})
                logger.info(f"Gemini client #{i+1} initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client #{i+1}: {e}")

    def get_gemini_client(self) -> Optional[Dict[str, Any]]:
        if not self.gemini_clients:
            return None
        current = self.gemini_clients[self.current_gemini_index]
        if not current["quota_exceeded"]:
            return current
        for i, client in enumerate(self.gemini_clients):
            if not client["quota_exceeded"]:
                self.current_gemini_index = i
                logger.info(f"Rotated to Gemini client #{i+1}.")
                return client
        logger.warning("All Gemini clients have exceeded their usage quotas.")
        return None

    def rotate_gemini_client(self):
        if not self.gemini_clients:
            return
        client_num = self.current_gemini_index + 1
        logger.warning(f"Marking Gemini client #{client_num} as quota-exceeded.")
        self.gemini_clients[self.current_gemini_index]["quota_exceeded"] = True
        self.get_gemini_client()


class Analyst:
    def __init__(self, llm_manager: LLMClientManager, local_llm_url: str, local_llm_key: str, local_model_name: str):
        self.llm_manager = llm_manager
        self.local_llm_url = local_llm_url
        self.local_llm_key = local_llm_key
        self.local_model_name = local_model_name

    def analyze_article(self, article: NewsArticle, asset: str, model_preference: str) -> Dict[str, Any]:
        logger.info(
            f"Starting analysis for article: '{(article.title or '')[:60]}...' | Asset: {asset} | Model: {model_preference}"
        )
        system_prompt = self._get_system_prompt(asset)
        user_prompt = self._get_user_prompt(article, asset)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        analysis_result: Optional[Dict[str, Any]] = None

        if model_preference in ("gemini", "auto"):
            analysis_result = self._query_gemini(messages)

        if not analysis_result and model_preference in ("local", "auto"):
            analysis_result = self._query_local_llm(messages)

        if not analysis_result:
            analysis_result = self._deterministic_fallback(article)

        article.analysis = analysis_result
        article.sentiment_score = analysis_result.get("sentiment_score", 0.0)
        article.sentiment_label = analysis_result.get("sentiment_label", "NEUTRAL")
        article.relevance_score = analysis_result.get("relevance_score", 0.0)
        article.ai_summary = analysis_result.get("executive_summary", "")
        return analysis_result

    def _query_gemini(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        gemini_client_info = self.llm_manager.get_gemini_client()
        if not gemini_client_info:
            return None
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        try:
            response = gemini_client_info["client"].generate_content(prompt)  # type: ignore[call-arg]
            raw_text = response.text  # type: ignore[attr-defined]
            return self._parse_llm_response(raw_text)
        except Exception as e:
            if "quota" in str(e).lower():
                self.llm_manager.rotate_gemini_client()
            return None

    def _query_local_llm(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.local_llm_key and self.local_llm_key != "not-needed":
            headers["Authorization"] = f"Bearer {self.local_llm_key}"

        json_schema = {
            "name": "analysis",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "sentiment_score": {"type": "number"},
                    "sentiment_label": {"type": "string"},
                    "relevance_score": {"type": "number"},
                    "executive_summary": {"type": "string"},
                    "market_verdict": {"type": "string"},
                    "market_impact": {"type": "string"},
                    "confidence_level": {"type": "number"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "sentiment_score",
                    "sentiment_label",
                    "relevance_score",
                    "executive_summary",
                    "market_verdict",
                    "market_impact",
                    "confidence_level",
                    "key_points",
                ],
            },
            "strict": True,
        }

        chat_payload_schema = {
            "model": self.local_model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 512,
            "response_format": {"type": "json_schema", "json_schema": json_schema},
            "stream": False,
        }
        chat_payload_simple = {
            "model": self.local_model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
        }

        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        completions_payload = {
            "model": self.local_model_name,
            "prompt": prompt + "\nReturn ONLY a valid JSON object.",
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
        }

        try:
            base = self.local_llm_url.rstrip("/")
            response = requests.post(
                f"{base}/chat/completions",
                headers=headers,
                json=chat_payload_schema,
                timeout=settings.local_llm_timeout,
            )

            if response.status_code == 400 and "response_format" in (response.text or "").lower():
                chat_payload_text = dict(chat_payload_simple)
                chat_payload_text["response_format"] = {"type": "text"}
                response = requests.post(
                    f"{base}/chat/completions",
                    headers=headers,
                    json=chat_payload_text,
                    timeout=settings.local_llm_timeout,
                )

            if response.status_code == 200:
                data = response.json()
                raw_text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content")
                ) or data.get("choices", [{}])[0].get("text", "")
                if not raw_text:
                    return None
                return self._parse_llm_response(raw_text)

            # Fallback: try /completions endpoint
            resp2 = requests.post(
                f"{base}/completions",
                headers=headers,
                json=completions_payload,
                timeout=settings.local_llm_timeout,
            )
            if resp2.status_code == 200:
                data2 = resp2.json()
                raw_text2 = data2.get("choices", [{}])[0].get("text", "")
                if not raw_text2:
                    return None
                return self._parse_llm_response(raw_text2)
            return None
        except requests.RequestException:
            return None

    def _parse_llm_response(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _get_system_prompt(self, asset: str) -> str:
        return (
            f"You are a top-tier financial analyst. Analyze the news impact on {asset.upper()}. "
            "Return ONLY a valid JSON object with keys: sentiment_score (-1..1), sentiment_label (BULLISH/BEARISH/NEUTRAL), "
            "relevance_score (0..1), executive_summary (2–3 factual sentences), market_verdict (STRONG BUY/BUY/HOLD/SELL/STRONG SELL), "
            "market_impact (HIGH/MEDIUM/LOW), confidence_level (0..1), and key_points (array of strings)."
        )

    def _get_user_prompt(self, article: NewsArticle, asset: str) -> str:
        content_preview = (article.content or "")[:500] if (article.content or "").strip() else "Limited content available."
        return (
            f"Analyze this article for {asset.upper()} and return only JSON.\n\n"
            f"Title: {article.title}\n"
            f"Description: {article.description or 'N/A'}\n"
            f"Content Preview: {content_preview}\n"
            f"Source: {article.source}\n"
            f"Published: {article.published_at}"
        )

    def _deterministic_fallback(self, article: NewsArticle) -> Dict[str, Any]:
        text = f"{article.title} {article.description}".lower()
        score = 0.0
        if any(k in text for k in ["bullish", "gain", "surge", "positive", "growth"]):
            score += 0.5
        if any(k in text for k in ["bearish", "loss", "plunge", "negative", "decline"]):
            score -= 0.5
        label = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
        return {
            "sentiment_score": score,
            "sentiment_label": label,
            "relevance_score": 0.5,
            "executive_summary": article.description or article.title,
            "market_verdict": "HOLD",
            "market_impact": "LOW",
            "confidence_level": 0.3,
            "key_points": ["Fallback analysis used."],
        }
