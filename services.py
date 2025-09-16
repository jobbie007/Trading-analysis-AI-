import os
import sys
import pathlib
import requests
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Callable
import textwrap
import io
import pandas as pd
import plotly.graph_objects as go
import requests as _requests
from urllib.parse import urlparse
try:
    from duckduckgo_search import DDGS as _DDGS
except Exception:
    _DDGS = None
yf = None  # lazy-imported via importlib when needed
PdfReader = None  # lazy-imported via importlib when needed
try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:
    _load_dotenv = None

# Task dataclass for workflow management
@dataclass
class Task:
    agent_name: str  # The name of the agent method to call
    output_key: str  # Where to store the result in the final output dictionary
    params: Dict[str, Any] = field(default_factory=dict)

# ---- Serper.dev Google Search (key-rotation) ----
class _SerperClient:
    def __init__(self):
        self._keys = [
            (os.getenv("SERPER_API_KEY") or "").strip(),
            (os.getenv("SERPER_API_KEY2") or "").strip(),
            (os.getenv("SERPER_API_KEY3") or "").strip(),
        ]
        self._keys = [k for k in self._keys if k]
        self._i = 0

    def has_keys(self) -> bool:
        return bool(self._keys)

    def _next_key(self) -> str:
        if not self._keys:
            return ""
        k = self._keys[self._i % len(self._keys)]
        self._i = (self._i + 1) % max(1, len(self._keys))
        return k

    def search(self, query: str, num: int = 10) -> list[dict]:
        """Return list of {title, link, snippet}. Rotates keys on 403/429/empty."""
        if not self._keys:
            return []
        url = "https://google.serper.dev/search"
        tried = 0
        out: list[dict] = []
        while tried < max(1, len(self._keys)) and not out:
            key = self._next_key()
            headers = {"X-API-KEY": key, "Content-Type": "application/json"}
            payload = {"q": query, "num": max(1, min(int(num or 10), 20)), "gl": "us", "hl": "en"}
            try:
                r = _requests.post(url, headers=headers, json=payload, timeout=15)
                if r.status_code in (401, 402, 403, 429, 500, 503):
                    tried += 1
                    continue
                if r.status_code == 200:
                    data = r.json() or {}
                    organic = data.get("organic") or []
                    for it in organic[: payload["num"]]:
                        out.append({
                            "title": (it.get("title") or "")[:200],
                            "link": it.get("link") or it.get("url") or "",
                            "snippet": (it.get("snippet") or it.get("description") or "")[:300],
                        })
                break
            except Exception:
                tried += 1
                continue
        return out

# Lazy, single import of yfinance
def _get_yf():
    """Return the cached yfinance module, importing it lazily once."""
    global yf
    if yf is None:
        try:
            import importlib as _il
            yf = _il.import_module("yfinance")
        except Exception as e:
            raise RuntimeError("yfinance not installed") from e
    return yf

# Ensure the project root is on sys.path so we can import the sibling 'marketNews' package
_THIS_FILE = pathlib.Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
# Add sibling 'marketNews' directory (which contains the 'marketnews' package) to import path
_SIBLING_MARKETNEWS_DIR = _PROJECT_ROOT / "marketNews"
for p in (str(_SIBLING_MARKETNEWS_DIR), str(_PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.append(p)

# Proactively load env variables from the sibling marketNews/.env if present
if _load_dotenv:
    _env_path = _SIBLING_MARKETNEWS_DIR / ".env"
    if _env_path.exists():
        _load_dotenv(dotenv_path=str(_env_path), override=False)

# Quiet noisy libraries (e.g., duckduckgo_search/httpx) without affecting app logs
for _name in ("duckduckgo_search", "ddg", "ddgs", "httpx", "urllib3"):
    try:
        _log = logging.getLogger(_name)
        _log.setLevel(logging.WARNING)
        _log.propagate = False
    except Exception:
        pass

# Central app logger (writes to dash/logs/ai_providers.log)
def _get_app_logger() -> logging.Logger:
    logger = logging.getLogger("dash.ai")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        try:
            logs_dir = _PROJECT_ROOT / "dash" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(logs_dir / "ai_providers.log"), encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            # If file logging fails, fallback to stderr only
            sh = logging.StreamHandler()
            logger.addHandler(sh)
    return logger

# Lightweight bridge that reuses your marketNews services when available
# Robust multi-path import that works whether the package is named 'marketnews' or 'marketNews'
_MNArticle = None
_MNFetcher = None
_MNLLM = None
_MNSettings = None
_MNAnalyst = None
_HAS_MN = False
try:
    import importlib
    for _pkg in ("marketnews", "marketNews"):
        try:
            _a = importlib.import_module(f"{_pkg}.core.article")
            _s = importlib.import_module(f"{_pkg}.services.news_fetcher")
            _l = importlib.import_module(f"{_pkg}.services.llm_client")
            _an = importlib.import_module(f"{_pkg}.analysis.analyst")
            _cfg = importlib.import_module(f"{_pkg}.config.settings")
            _MNArticle = getattr(_a, "NewsArticle", None)
            _MNFetcher = getattr(_s, "NewsFetcher", None)
            _MNLLM = getattr(_l, "LLMClientManager", None)
            _MNAnalyst = getattr(_an, "Analyst", None)
            _MNSettings = getattr(_cfg, "settings", None)
            if all([_MNArticle, _MNFetcher, _MNLLM, _MNAnalyst, _MNSettings]):
                _HAS_MN = True
                break
        except Exception:
            continue
except Exception:
    pass

@dataclass
class NewsItem:
    title: str
    url: str
    description: str
    published_at: str
    source: str
    content: str = ""  
    ai_summary: str = ""
    sentiment_label: str = ""
    sentiment_score: float = 0.0
    analysis_provider: str = ""

# Replace your entire existing NewsBridge class with this one.

class NewsBridge:
    def __init__(self):
        self.logger = _get_app_logger()
        self.has_mn = _HAS_MN
        if self.has_mn:
            # Allow env to override DISABLE_API_KEYS at runtime
            try:
                _disable_env = os.getenv("DISABLE_API_KEYS")
                if _disable_env is not None:
                    _disable = str(_disable_env).strip().lower() in ("1", "true", "yes", "on")
                else:
                    _disable = bool(getattr(_MNSettings, "disable_api_keys", False))
            except Exception:
                _disable = bool(getattr(_MNSettings, "disable_api_keys", False))
            self.fetcher = _MNFetcher(
                gnews_keys=_MNSettings.gnews_api_keys,
                newsapi_key=_MNSettings.newsapi_key,
                cryptopanic_key=_MNSettings.cryptopanic_api_key,
                disable_api_keys=_disable,
            )
            # Initialize LLM manager and analyst for AI summaries & sentiment
            self.llm_manager = _MNLLM(_MNSettings.gemini_api_keys, _MNSettings.gemini_model_name)
            self.analyst = _MNAnalyst(
                llm_manager=self.llm_manager,
                local_llm_url=_MNSettings.local_llm_base_url,
                local_llm_key=_MNSettings.local_llm_api_key,
                local_model_name=_MNSettings.local_model_name,
            )
            # Track which engine produced results
            self._last_provider = None
            try:
                if hasattr(self.analyst, "_query_gemini"):
                    _orig_gem = self.analyst._query_gemini
                    def _wrap_gemini(messages):
                        res = _orig_gem(messages)
                        if res is not None:
                            self._last_provider = "gemini"
                        return res
                    self.analyst._query_gemini = _wrap_gemini  # type: ignore
                if hasattr(self.analyst, "_query_local_llm"):
                    _orig_loc = self.analyst._query_local_llm
                    def _wrap_local(messages):
                        res = _orig_loc(messages)
                        if res is not None and not self._last_provider:
                            self._last_provider = "local"
                        return res
                    self.analyst._query_local_llm = _wrap_local  # type: ignore
            except Exception:
                pass
        else:
            self.fetcher = None
            self.llm_manager = None
            self.analyst = None
            self._last_provider = None
        self.last_logs = []
        self._last_fetch_summary = ""
        # Local LLM tunables are now accessed directly via os.getenv in helper methods

    # --- helpers ---
    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Finds and parses the first complete JSON object in a string."""
        if not text:
            return None
        try:
            j = json.loads(text)
            if isinstance(j, dict): return j
        except Exception: pass
        start = text.find("{")
        while start != -1 and start < len(text):
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : i + 1]
                        try:
                            j = json.loads(chunk)
                            if isinstance(j, dict): return j
                        except Exception: break
            start = text.find("{", start + 1)
        return None
    
    def _get_provider_config(self) -> Dict[str, Dict[str, Any]]:
        """Consolidates provider configurations from environment variables."""
        # Use getattr for safe access to self attributes that might not be set
        local_llm_base_url = getattr(self, "local_llm_base_url", "")
        local_llm_api_key = getattr(self, "local_llm_api_key", "not-needed")
        local_llm_model = getattr(self, "local_llm_model", "gpt-4o-mini-compat")

        return {
            "do": {
                "name": "DigitalOcean",
                "base_url": (os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1") or "").strip().rstrip("/"),
                "api_key": (os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY") or "").strip(),
                "model": (os.getenv("DO_AI_MODEL") or "llama3-8b-instruct").strip(),
            },
            "hf": {
                "name": "Hugging Face",
                "base_url": (os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1") or "").strip().rstrip("/"),
                "api_key": (os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN2") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip(),
                "model": (os.getenv("HF_MODEL") or "OpenAI/gpt-oss-20B").strip(),
            },
            "local": {
                "name": "Local OpenAI-Compat",
                "base_url": (os.getenv("LOCAL_LLM_BASE_URL", local_llm_base_url)).strip().rstrip("/"),
                "api_key": (os.getenv("LOCAL_LLM_API_KEY", local_llm_api_key)).strip(),
                "model": (os.getenv("LOCAL_LLM_MODEL", local_llm_model)).strip(),
            },
        }

    def _query_llm(self, provider_config: Dict[str, Any], messages: List[Dict[str, str]], temperature: float) -> Optional[str]:
        """A generalized function to query any OpenAI-compatible LLM endpoint."""
        base_url, api_key, model = provider_config.get("base_url"), provider_config.get("api_key"), provider_config.get("model")
        if not (base_url and model): return None

        try:
            headers = {"Content-Type": "application/json"}
            if api_key and api_key != "not-needed":
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096")),
                "stream": False,
            }
            
            r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
            self.logger.info("news._query_llm: provider=%s model=%s status=%s", provider_config.get("name", "Unknown"), model, r.status_code)

            if r.status_code != 200:
                self.logger.warning("news._query_llm error (%s): %s", provider_config.get("name"), (r.text or "")[:200])
                return None
            
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content") or data.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            self.logger.exception("news._query_llm exception (%s): %s", provider_config.get("name"), e)
            return None

    def fetch(self, asset: str, days_back: int = 7, max_articles: int = 10, *, model_preference: str = "auto", analyze: bool = True) -> List[NewsItem]:
        self.logger.info("news.fetch: asset=%s days=%s max=%s model_pref=%s analyze=%s", asset, days_back, max_articles, model_preference, analyze)
        if not self.fetcher:
            return [NewsItem(title=f"{asset} placeholder item", url="#", description="Wire marketNews to enable live articles.", published_at="", source="local")]
        
        arts = self.fetcher.fetch_news(asset, days_back, max_articles)
        self._last_fetch_summary = getattr(self.fetcher, 'get_last_attempts_summary', lambda: '')() or ""
        
        for a in arts[:5]:
            if not getattr(a, 'content', None):
                a.content = self.fetcher.scrape_article_content(a.url)
        
        if not (analyze and self.analyst):
            return [NewsItem(**{k: getattr(a, k, '') for k in NewsItem.__annotations__}) for a in arts]

        self.last_logs = []
        providers = self._get_provider_config()
        use_do = (str(model_preference or "").lower() == "do") or (os.getenv("AI_PROVIDER", "").strip().lower() == "do")
        use_hf = (str(model_preference or "").lower() == "hf") or (os.getenv("AI_PROVIDER", "").strip().lower() == "hf")

        system_msg = ("You are a financial news analyst. Given an article, return a compact JSON with keys: "
                      "'summary' (2-4 sentences), 'sentiment_label' (bullish|bearish|neutral), and 'sentiment_score' (a float between -1 and 1).")
        
        for a in arts:
            try:
                user_msg = textwrap.dedent(f"""
                    Title: {getattr(a, 'title', '')}
                    Description: {getattr(a, 'description', '')}
                    Content: {(getattr(a, 'content', '') or '')[:8000]}
                    Return only a JSON object, no extra text. Do not include any chain-of-thought or analysis outside JSON.
                """)
                messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
                
                analysis_done = False
                for provider_key, use_flag in [("do", use_do), ("hf", use_hf)]:
                    if use_flag:
                        raw_text = self._query_llm(providers[provider_key], messages, 0.2)
                        parsed = self._extract_json_object(raw_text) if raw_text else None
                        if parsed:
                            setattr(a, 'ai_summary', parsed.get('summary', '') or '')
                            setattr(a, 'sentiment_label', (parsed.get('sentiment_label', '') or '').lower())
                            try: setattr(a, 'sentiment_score', float(parsed.get('sentiment_score', 0.0) or 0.0))
                            except Exception: setattr(a, 'sentiment_score', 0.0)
                            setattr(a, 'analysis_provider', provider_key)
                            self.last_logs.append(f"{asset} | {provider_key.upper()} | {(a.title or '')[:80]}")
                            analysis_done = True
                            break
                
                if analysis_done:
                    continue

                self._last_provider = None
                self.analyst.analyze_article(a, asset, model_preference=model_preference)
                provider = (self._last_provider or "fallback").lower()
                setattr(a, 'analysis_provider', provider)
                self.last_logs.append(f"{asset} | {provider.upper()} | {(a.title or '')[:80]}")
            except Exception:
                continue

        return [NewsItem(**{k: getattr(a, k, v) for k, v in NewsItem.__dataclass_fields__.items()}) for a in arts]
    
    def _get_technical_ai_summary(self, symbol: str, tech_summary: str, model_preference: str = "auto") -> Dict[str, Any]:
        system_msg = textwrap.dedent("""
            You are an expert technical analyst. Based on the provided technical indicators for a stock, you will:
            1. Provide a recommendation: "Buy", "Sell", or "Hold".
            2. Provide a confidence score for your recommendation on a scale of 1 to 10.
            3. Provide a detailed justification for your recommendation, referencing the specific indicator values provided.
            Respond in a JSON format with three keys: "recommendation", "confidence", "justification".
        """)
        user_msg = f"Symbol: {symbol}\n\nTechnical Summary:\n{tech_summary}"
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]

        ai_provider = (os.getenv("AI_PROVIDER", model_preference) or "auto").strip().lower()
        providers_config = self._get_provider_config()

        order: List[str] = []
        if ai_provider in {"local", "do", "gemini"}:
            order = [ai_provider]
        else:
            if providers_config["local"].get("base_url"): order.append("local")
            if providers_config["do"].get("api_key"): order.append("do")
            order.extend(["gemini", "bridgelocal"])

        bridge_callers: Dict[str, Callable] = {
            "gemini": lambda: hasattr(self.analyst, "_query_gemini") and self.analyst._query_gemini(messages),
            "bridgelocal": lambda: hasattr(self.analyst, "_query_local_llm") and self.analyst._query_local_llm(messages)
        }
        
        for prov in order:
            result = None
            if prov in providers_config:
                raw_text = self._query_llm(providers_config[prov], messages, 0.1)
                if raw_text: result = self._extract_json_object(raw_text)
            elif prov in bridge_callers:
                result = bridge_callers[prov]()

            if isinstance(result, dict):
                return result
        
        for prov in {"bridgelocal", "gemini"}:
            if prov not in order:
                result = bridge_callers.get(prov, lambda: None)()
                if isinstance(result, dict): return result
        
        return {}
    
    def summarize_overall(self, asset: str, items: List[NewsItem], *, model_preference: str = "auto", max_chars: int = 20000) -> Dict[str, str]:
        if self.fetcher:
            try:
                for a in items:
                    url = getattr(a, 'url', '')
                    content = getattr(a, 'content', '')
                    if url and not content:
                        scraped_content = self.fetcher.scrape_article_content(url)
                        if scraped_content:
                            setattr(a, 'content', scraped_content)
            except Exception:
                pass
            
        context_lines = []
        for i, it in enumerate(items[:4], 1):
            score = f" {float(getattr(it, 'sentiment_score', 0)):+.2f}" if getattr(it, 'sentiment_score', None) is not None else ""
            clipped_content = (getattr(it, 'content', '') or '')[:3000]
            context_lines.append(textwrap.dedent(f"""
            ### Item {i}
            Title: {it.title}
            Source/Date: {it.source} • {it.published_at}
            Sentiment: {it.sentiment_label}{score}
            Summary: {it.ai_summary}
            Description: {it.description}
            Content: {clipped_content}
            URL: {it.url}
            """))
        context = "\n".join(context_lines)[:min(int(max_chars or 20000), 12000)]
        
        print("=" * 60)
        print(f"DEBUG (summarize_overall): Starting for asset '{asset}'")
        print(f"DEBUG: Total context length being sent to LLM: {len(context)} characters")
        if len(context) < 100:
            print("DEBUG: WARNING - Context is very short or empty. No news items to summarize?")
            return {"summary": "Not enough news content to generate a summary.", "provider": "system", "overall_sentiment": "neutral", "confidence": 0.0, "payload": {}}

        system_msg = (
            "You are a seasoned financial markets analyst. Given multiple news items about a symbol, "
            "produce a concise executive summary. Reply as strict JSON only, no prose, using this schema: "
            "{\n"
            "  \"executive_summary\": string,\n"
            "  \"overall_sentiment\": string,  # one of: bullish | bearish | neutral,\n"
            "  \"confidence\": number,        # 0.0-1.0,\n"
            "  \"key_points\": [string]      # optional\n"
            "}\n"
            "Keep the summary tight and non-repetitive."
        )
        user_msg = f"Symbol: {asset}\n\nCorpus:\n{context}"
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        
        providers_config = self._get_provider_config()
        ai_provider = (os.getenv("AI_PROVIDER", model_preference) or "auto").strip().lower()

        order: List[str] = []
        if ai_provider in providers_config: order.append(ai_provider)
        for p in ["local", "do", "hf", "gemini", "bridgelocal"]:
            if p not in order: order.append(p)
        
        print(f"DEBUG: Provider attempt order: {order}")

        bridge_callers = {
            "gemini": lambda: self.analyst._query_gemini(messages) if hasattr(self.analyst, "_query_gemini") else None,
            "bridgelocal": lambda: self.analyst._query_local_llm(messages) if hasattr(self.analyst, "_query_local_llm") else None,
        }

        raw_result, provider = None, "fallback"
        for p_name in order:
            print(f"DEBUG: --> Attempting provider: '{p_name}'")
            if p_name in providers_config:
                raw_result = self._query_llm(providers_config[p_name], messages, 0.1)
            elif p_name in bridge_callers:
                raw_result = bridge_callers[p_name]()
            
            if raw_result:
                print(f"DEBUG: SUCCESS with provider '{p_name}'!")
                provider = p_name
                break
            else:
                print(f"DEBUG: FAILED with provider '{p_name}'. Trying next.")
        
        if not raw_result:
            print("DEBUG: All providers failed to return a result.")
        print("=" * 60)

        payload = self._extract_json_object(raw_result) if isinstance(raw_result, str) else (raw_result if isinstance(raw_result, dict) else {})
        summary_text = (payload or {}).get('executive_summary') or (payload or {}).get('summary') or ""
        
        if not summary_text:
            bullets = [f"- {it.title}: {(it.ai_summary or it.description)[:180]}" for it in items[:8]]
            summary_text = "Overall summary (fallback):\n" + "\n".join(bullets)
        
        sent = (str((payload or {}).get("overall_sentiment", "")).lower() or "neutral")
        if not sent in ["bullish", "bearish", "neutral"]: sent = "neutral"

        conf = 0.5
        try: conf = float((payload or {}).get('confidence', 0.5))
        except: pass

        return {"summary": summary_text, "provider": provider, "overall_sentiment": sent, "confidence": conf, "payload": payload or {}}

    def get_logs(self) -> List[str]:
        return list(self.last_logs)

    def get_fetch_summary(self) -> str:
        return getattr(self, '_last_fetch_summary', '') or ''

# ---------------- Multi‑Agent Research (free-data) ----------------
# NOTE: The rest of the file (ResearchOrchestrator, etc.) remains unchanged.
# I am including it here for completeness so you can copy the whole file.
class ResearchOrchestrator:
    """Lightweight multi-agent system tailored for this app.
    Agents:
      - DataAgent: fetch OHLCV using free sources (yfinance; stooq fallback).
      - NewsAgent: leverage NewsBridge to get headlines/AI summaries.
      - RequirementsAgent: read local requirements.pdf and project description to shape tasks.
      - SynthesisAgent: merge signals and propose actions.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = pathlib.Path(base_dir or _PROJECT_ROOT)
        self.bridge = NewsBridge()

    WHITELIST_DOMAINS = {
        "sec.gov", "seekingalpha.com", "bloomberg.com", "reuters.com",
        "wsj.com", "ft.com", "cnbc.com", "marketwatch.com", "morningstar.com",
        "fool.com", "investopedia.com", "finance.yahoo.com", "nasdaq.com",
        "markets.businessinsider.com", "thefly.com", "tipranks.com"
    }

    BLOCK_DOMAINS = {
        "youtube.com", "music.youtube.com", "facebook.com", "instagram.com",
        "twitter.com", "tiktok.com", "pinterest.com", "reddit.com"
    }

    FINANCE_KEYWORDS = {
        "stock", "earnings", "revenue", "valuation", "sec", "investor",
        "analyst", "nasdaq", "nyse", "p/e", "guidance", "quarterly", "annual"
    }

    # Agents
    def data_agent(self, symbol: str, period: str, interval: str = "1d") -> Tuple[str, Dict[str, Any]]:
        """Fetch price data with yfinance; fallback to stooq via pandas-datareader path if needed.
        Returns: (message, payload)
        """
        msg = ""
        payload: Dict[str, Any] = {}
        try:
            t_yf = _get_yf()
            ticker = t_yf.Ticker(symbol)
            # Map friendly hourly to yfinance supported values
            yf_interval = interval if interval in {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"} else "1d"
            hist = ticker.history(period=period, interval=yf_interval, auto_adjust=False)
            if hist is None or hist.empty:
                raise RuntimeError("No data from yfinance")
            payload["source"] = "yfinance"
            payload["hist"] = hist.tail(180)
            msg = f"DataAgent: fetched {len(hist)} rows from yfinance for {symbol}."
            return msg, payload
        except Exception as e:
            # Try stooq via pandas-datareader-like URL (no external dep):
            try:
                import pandas as pd
                import datetime as _dt
                base = "https://stooq.com/q/d/l/"
                # Stooq expects e.g., msft.us
                sym = symbol.lower().replace(":", ".")
                if not sym.endswith(".us") and len(sym) <= 5:
                    sym = sym + ".us"
                url = f"{base}?s={sym}&i=d"
                df = pd.read_csv(url)
                if not df.empty:
                    df.rename(columns=str.title, inplace=True)
                    df["Date"] = pd.to_datetime(df["Date"])  # type: ignore
                    df.set_index("Date", inplace=True)
                    payload["source"] = "stooq"
                    payload["hist"] = df.tail(180)
                    msg = f"DataAgent: fetched {len(df)} rows from Stooq for {symbol}."
                    return msg, payload
            except Exception:
                pass
            msg = f"DataAgent error: {e}"
            return msg, payload

    def technical_strategist_agent(self, symbol: str, price_hist: Optional[pd.DataFrame], tech_summary: str, ai_provider: str = "auto") -> Tuple[str, str]:
        """
        A more advanced agent that calculates key levels and asks the LLM to generate
        a strategic analysis instead of a simple justification.
        """
        if price_hist is None or price_hist.empty or len(price_hist) < 2:
            return "StrategistAgent: Not enough data.", ""

        try:
            # --- 1. Calculate Enriched Data ---
            current_price = price_hist["Close"].iloc[-1]
            recent_high = price_hist["High"].max()
            recent_low = price_hist["Low"].min()

            # Calculate Pivot Points from the previous period's data
            prev_h = price_hist["High"].iloc[-2]
            prev_l = price_hist["Low"].iloc[-2]
            prev_c = price_hist["Close"].iloc[-2]
            
            p = (prev_h + prev_l + prev_c) / 3
            r1 = (2 * p) - prev_l
            s1 = (2 * p) - prev_h
            r2 = p + (prev_h - prev_l)
            s2 = p - (prev_h - prev_l)
            
            pivots = {
                "r2": f"{r2:.2f}", "r1": f"{r1:.2f}", "p": f"{p:.2f}",
                "s1": f"{s1:.2f}", "s2": f"{s2:.2f}"
            }

            # Calculate Fibonacci Retracement Levels for the period
            diff = recent_high - recent_low
            fib_levels = {
                "0.236": f"{recent_high - 0.236 * diff:.2f}",
                "0.382": f"{recent_high - 0.382 * diff:.2f}",
                "0.500": f"{recent_high - 0.500 * diff:.2f}",
                "0.618": f"{recent_high - 0.618 * diff:.2f}",
            }
            
        except Exception as e:
            return f"StrategistAgent: Failed to calculate levels - {e}", ""

        # --- 2. Construct the Advanced Prompt ---
        system_msg = (
            "You are an expert technical trading strategist. Your analysis is concise, data-driven, and actionable. "
            "You will be given a snapshot of technical indicators along with calculated support and resistance levels. "
            "Your task is to synthesize this information into a clear, strategic trading plan."
        )

        user_msg = textwrap.dedent(f"""
        Analyze the technical data for the ticker **{symbol}** and provide a strategic plan.

        **Current Indicator Snapshot:**
        {tech_summary}

        **Key Price Levels:**
        - Current Price: {current_price:.2f}
        - Period High: {recent_high:.2f}
        - Period Low: {recent_low:.2f}

        **Calculated Pivot Points (for current period):**
        - Resistance 2 (R2): {pivots['r2']}
        - Resistance 1 (R1): {pivots['r1']}
        - Pivot Point (P): {pivots['p']}
        - Support 1 (S1): {s1:.2f}
        - Support 2 (S2): {s2:.2f}

        **Fibonacci Retracement Levels (based on period high/low):**
        - 0.236 Level: {fib_levels['0.236']}
        - 0.382 Level: {fib_levels['0.382']}
        - 0.500 Level: {fib_levels['0.500']}
        - 0.618 Level: {fib_levels['0.618']}

        ---
        **Instructions:**

        Based on all the data above, provide a concise, actionable trading plan. Structure your response in markdown with the following sections:

        1.  **Overall Assessment:** A brief 2-3 sentence summary of the current technical picture (e.g., "The stock is in a strong uptrend but appears overbought, approaching a key resistance level...").
        2.  **Key Levels to Watch:** Clearly list the most important support and resistance levels to monitor, referencing the pivot points, Fibonacci levels, and recent highs/lows.
        3.  **Potential Trade Scenarios:**
            -   **Bullish Scenario (Potential Buy):** Describe what price action or indicator signal would confirm a bullish outlook (e.g., "A breakout and hold above R1 at {pivots['r1']} could signal a continuation...").
            -   **Bearish Scenario (Potential Sell/Caution):** Describe what would signal a reversal or breakdown (e.g., "A break below the key support at the 0.382 Fibonacci level of {fib_levels['0.382']} could indicate a deeper pullback...").
        """)
        
        # --- 3. Call the LLM and Return ---
        strategy_text = self._llm_generate(system_msg, user_msg, ai_provider=ai_provider)
        
        if not strategy_text:
            return "StrategistAgent: LLM call failed.", "Failed to generate strategic analysis."
            
        return "StrategistAgent: Generated strategic analysis.", strategy_text

    def data_agent_with_dates(self, symbol: str, start: str, end: str, interval: str = "1d") -> Tuple[str, Dict[str, Any]]:
        """Fetch price data for an explicit date range using yfinance.
        Returns: (message, {source, hist})
        """
        msg = ""
        payload: Dict[str, Any] = {}
        try:
            t_yf = _get_yf()
            ticker = t_yf.Ticker(symbol)
            yf_interval = interval if interval in {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"} else "1d"
            hist = ticker.history(start=start, end=end, interval=yf_interval, auto_adjust=False)
            if hist is None or hist.empty:
                # Fallback to approximate window
                return self.data_agent(symbol, period="1y", interval=interval)
            payload["source"] = "yfinance"
            payload["hist"] = hist
            msg = f"DataAgent(dates): {symbol} {start}->{end} rows={len(hist)}"
            return msg, payload
        except Exception as e:
            return f"DataAgent(dates) error: {e}", payload

    def synthesis_agent_for_technicals(
        self,
        hist: Optional[pd.DataFrame],
        indicators: List[str],
        length: int,
        rsi_length: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal_len: int,
    ) -> Tuple[go.Figure, str]:
        """Create a multi-panel technical chart and a brief textual summary.
        Returns (figure, summary_text). Raises if hist is missing/empty.
        """
        if hist is None or len(hist) == 0:
            raise RuntimeError("No historical data provided")

        df = hist.copy()
        # Normalize column names and accessors
        cols = {c.lower(): c for c in df.columns}
        def _col(name: str) -> str:
            return cols.get(name.lower(), name)

        close = df[_col("Close")].astype(float)
        high = df[_col("High")].astype(float)
        low = df[_col("Low")].astype(float)

        # Compute indicators (with graceful fallback if ta is missing)
        w = max(2, int(length or 20))
        sma = close.rolling(window=w).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        try:
            import importlib as _il
            _ta_trend = _il.import_module("ta.trend")
            _ta_mom = _il.import_module("ta.momentum")
            _ta_vol = _il.import_module("ta.volatility")
            _MACD = getattr(_ta_trend, "MACD", None)
            _RSI = getattr(_ta_mom, "RSIIndicator", None)
            _BB = getattr(_ta_vol, "BollingerBands", None)
            if not (_MACD and _RSI and _BB):
                raise ImportError("ta library not available")
            macd_obj = _MACD(close, window_fast=max(2, macd_fast or 12), window_slow=max(3, macd_slow or 26), window_sign=max(2, macd_signal_len or 9))
            macd_line = macd_obj.macd()
            macd_signal = macd_obj.macd_signal()
            rsi = _RSI(close, window=max(2, rsi_length or 14)).rsi()
            bb = _BB(close, window=max(5, length or 20), window_dev=2)
            bb_u = bb.bollinger_hband(); bb_l = bb.bollinger_lband()
        except Exception:
            ema_fast = close.ewm(span=max(2, macd_fast or 12), adjust=False).mean()
            ema_slow = close.ewm(span=max(3, macd_slow or 26), adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=max(2, macd_signal_len or 9), adjust=False).mean()
            delta = close.diff()
            gain = (delta.where(delta > 0, 0.0)).rolling(window=max(2, rsi_length or 14)).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=max(2, rsi_length or 14)).mean()
            rs = gain / (loss.replace(0, 1e-9))
            rsi = 100 - (100 / (1 + rs))
            rolling = close.rolling(window=max(5, length or 20))
            bb_mid = rolling.mean(); bb_std = rolling.std(ddof=0)
            bb_u = bb_mid + 2 * bb_std; bb_l = bb_mid - 2 * bb_std

        from plotly.subplots import make_subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        # Price panel
        fig.add_trace(
            go.Candlestick(x=df.index, open=df[_col("Open")], high=high, low=low, close=df[_col("Close")], name="Price"),
            row=1, col=1
        )
        fig.add_trace(go.Scatter(x=df.index, y=sma, name=f"SMA({length})", line=dict(color="#eab308")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA(20)", line=dict(color="#22c55e")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA(50)", line=dict(color="#3b82f6")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_u, name="BBand U", line=dict(color="#94a3b8", width=1), opacity=0.8), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb_l, name="BBand L", line=dict(color="#94a3b8", width=1), opacity=0.8), row=1, col=1)

        # RSI panel
        fig.add_trace(go.Scatter(x=df.index, y=rsi, name=f"RSI({rsi_length})", line=dict(color="#f97316")), row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#64748b", row=2, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#64748b", row=2, col=1)

        # MACD panel
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="#14b8a6")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=macd_signal, name="Signal", line=dict(color="#a78bfa")), row=3, col=1)

        fig.update_layout(template="plotly_dark", height=800, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridcolor="#1f2937")

        # Summary of last values
        try:
            parts = []
            parts.append(f"Close: {float(close.iloc[-1]):.2f}")
            if pd.notna(sma.iloc[-1]):
                parts.append(f"SMA({length}): {float(sma.iloc[-1]):.2f}")
            if pd.notna(ema20.iloc[-1]):
                parts.append(f"EMA20: {float(ema20.iloc[-1]):.2f}")
            if pd.notna(ema50.iloc[-1]):
                parts.append(f"EMA50: {float(ema50.iloc[-1]):.2f}")
            if pd.notna(rsi.iloc[-1]):
                parts.append(f"RSI: {float(rsi.iloc[-1]):.1f}")
            if pd.notna(macd_line.iloc[-1]) and pd.notna(macd_signal.iloc[-1]):
                parts.append(f"MACD: {float(macd_line.iloc[-1]):.2f}/{float(macd_signal.iloc[-1]):.2f}")
            summary = ", ".join(parts)
        except Exception:
            summary = ""

        return fig, summary

    def news_agent(self, symbol: str, count: int = 5) -> Tuple[str, List[NewsItem]]:
        items = self.bridge.fetch(symbol, days_back=7, max_articles=count, model_preference="auto", analyze=True)
        msg = f"NewsAgent: {len(items)} articles analyzed for {symbol}."
        return msg, items

    def web_search_agent(self, query: str, max_results: int = 5) -> Tuple[str, List[Dict[str, str]]]:
        """
        Robust web search agent that prioritizes Serper.dev and uses DuckDuckGo as a fallback.
        Returns (message, results)
        """
        enhanced_query = f'{query} stock analysis investor relations news'

        try:
            from telemetry import log_event
            log_event("web_search.request", {"query": enhanced_query, "max_results": int(max_results)})
        except Exception:
            pass

        # --- 1. Prioritize Serper.dev ---
        serper_client = _SerperClient()
        if serper_client.has_keys():
            try:
                # Serper is fast and reliable, so we can fetch and return directly.
                raw_results = serper_client.search(enhanced_query, num=max_results)

                # Normalize the output to match the expected format ('url' key)
                final_results = []
                for r in raw_results:
                    final_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("link", ""), # Serper uses 'link', we normalize to 'url'
                        "snippet": r.get("snippet", "")
                    })

                msg = f"WebSearchAgent: Found {len(final_results)} results via Serper."

                try: # Telemetry logging
                    from telemetry import log_event
                    log_event("web_search.result", {"provider": "serper", "count": len(final_results), "sample": final_results[:3]})
                except Exception:
                    pass

                return msg, final_results

            except Exception as e:
                msg = f"WebSearchAgent: Serper query failed with error: {e}. No results found."
                try: # Telemetry logging
                    from telemetry import log_event
                    log_event("web_search.error", {"provider": "serper", "error": str(e)})
                except Exception:
                    pass

                return msg, []

        # --- 2. Fallback to DuckDuckGo ---
        # If we reach this point, it means no SERPER_API_KEY was found in the .env file.

        if _DDGS is None:
            return "WebSearchAgent: Serper not configured and duckduckgo_search not installed.", []

        scored_results = []
        try:
            with _DDGS() as ddgs:
                # Fetch a larger pool to allow for strict filtering
                raw_results = ddgs.text(enhanced_query, max_results=max_results * 4)

                for r in raw_results:
                    title = (r.get("title") or "")[:200]
                    url = r.get("href") or r.get("url") or ""
                    snippet = (r.get("body") or r.get("snippet") or "")[:300]

                    if not url or not title:
                        continue

                    try:
                        domain = urlparse(url).netloc.replace("www.", "")
                    except Exception:
                        continue

                    if domain in self.BLOCK_DOMAINS:
                        continue

                    score = 0
                    is_whitelisted = any(whitelisted_domain in domain for whitelisted_domain in self.WHITELIST_DOMAINS)
                    if is_whitelisted:
                        score += 5

                    text_content = (title + " " + snippet).lower()
                    for keyword in self.FINANCE_KEYWORDS:
                        if keyword in text_content:
                            score += 1

                    if not is_whitelisted and score == 0:
                        continue

                    scored_results.append((score, {"title": title, "url": url, "snippet": snippet}))

            scored_results.sort(key=lambda x: x[0], reverse=True)
            final_results = [res for score, res in scored_results[:max_results]]

            msg = f"WebSearchAgent: Serper not configured. Found {len(final_results)} relevant results via DuckDuckGo (fallback)."

            try: # Telemetry logging
                from telemetry import log_event
                log_event("web_search.result", {"provider": "duckduckgo", "count": len(final_results), "sample": final_results[:3]})
            except Exception:
                pass

            return msg, final_results
        except Exception as e:
            try:
                from telemetry import log_event
                log_event("web_search.error", {"provider": "duckduckgo", "error": str(e)})
            except Exception:
                pass
            return f"WebSearchAgent error: {e}", []
   
    def requirements_agent(self) -> Tuple[str, Dict[str, str]]:
        docs = self._read_project_documents()
        keys = ", ".join(docs.keys()) or "none"
        return f"RequirementsAgent: loaded {keys}.", docs

    def synthesis_agent(self, symbol: str, data_payload: Dict[str, Any], news_items: List[NewsItem], docs: Dict[str, str]) -> Tuple[str, Dict[str, Any]]:
        # This research mode focuses on technical analysis; still include optional overall news summary for context
        overall = {"summary": "", "provider": ""}
        try:
            if news_items:
                overall = self.bridge.summarize_overall(symbol, news_items, model_preference="auto", max_chars=2000)
        except Exception:
            pass

        # Compute technical indicators for charting
        tech: Dict[str, Any] = {}
        try:
            import pandas as pd
            try:
                import importlib as _il
                _ta_trend = _il.import_module("ta.trend")
                _ta_vol = _il.import_module("ta.volatility")
                _ta_mom = _il.import_module("ta.momentum")
                _ta_volu = _il.import_module("ta.volume")
                SMAIndicator = getattr(_ta_trend, "SMAIndicator", None)
                EMAIndicator = getattr(_ta_trend, "EMAIndicator", None)
                MACD = getattr(_ta_trend, "MACD", None)
                CCIIndicator = getattr(_ta_trend, "CCIIndicator", None)
                BollingerBands = getattr(_ta_vol, "BollingerBands", None)
                AverageTrueRange = getattr(_ta_vol, "AverageTrueRange", None)
                RSIIndicator = getattr(_ta_mom, "RSIIndicator", None)
                StochasticOscillator = getattr(_ta_mom, "StochasticOscillator", None)
                ROCIndicator = getattr(_ta_mom, "ROCIndicator", None)
                MFIIndicator = getattr(_ta_volu, "MFIIndicator", None)
                OnBalanceVolumeIndicator = getattr(_ta_volu, "OnBalanceVolumeIndicator", None)
            except Exception as _e_ta:
                # TA library not installed; skip detailed indicators
                SMAIndicator = EMAIndicator = MACD = CCIIndicator = None  # type: ignore
                BollingerBands = AverageTrueRange = None  # type: ignore
                RSIIndicator = StochasticOscillator = ROCIndicator = None  # type: ignore
                MFIIndicator = OnBalanceVolumeIndicator = None  # type: ignore
            hist = data_payload.get("hist")
            if hist is not None and not hist.empty:
                df = hist.copy()
                df = df.rename(columns=str.title)
                close = df["Close"].astype(float)
                high = df["High"].astype(float)
                low = df["Low"].astype(float)
                vol = df["Volume"].astype(float)

                # Compute indicators only if TA lib available
                sma20 = EMA20 = EMA50 = bb_u = bb_l = rsi = macd_line = macd_signal = atr = cci = roc = mfi = obv = vwap = None
                if SMAIndicator and EMAIndicator and MACD:
                    sma20 = SMAIndicator(close, window=20).sma_indicator()
                    EMA20 = EMAIndicator(close, window=20).ema_indicator()
                    EMA50 = EMAIndicator(close, window=50).ema_indicator()
                    macd = MACD(close)
                    macd_line = macd.macd()
                    macd_signal = macd.macd_signal()
                if BollingerBands:
                    bb = BollingerBands(close, window=20, window_dev=2)
                    bb_u = bb.bollinger_hband()
                    bb_l = bb.bollinger_lband()
                if RSIIndicator:
                    rsi = RSIIndicator(close, window=14).rsi()
                if AverageTrueRange:
                    atr = AverageTrueRange(high, low, close, window=14).average_true_range()
                if CCIIndicator:
                    cci = CCIIndicator(high, low, close, window=20).cci()
                if ROCIndicator:
                    roc = ROCIndicator(close, window=12).roc()
                if MFIIndicator and OnBalanceVolumeIndicator:
                    mfi = MFIIndicator(high, low, close, vol, window=14).money_flow_index()
                    obv = OnBalanceVolumeIndicator(close, vol).on_balance_volume()

                # Simple VWAP approximation (daily): cumulative price*vol / cumulative vol
                pv = (close * vol).cumsum()
                v = vol.cumsum().replace(0, pd.NA)
                vwap = pv / v

                tech = {"last": float(close.iloc[-1])}
                if sma20 is not None: tech["sma20"] = sma20
                if EMA20 is not None: tech["ema20"] = EMA20
                if EMA50 is not None: tech["ema50"] = EMA50
                if bb_u is not None: tech["bb_u"] = bb_u
                if bb_l is not None: tech["bb_l"] = bb_l
                if rsi is not None: tech["rsi"] = rsi
                if macd_line is not None: tech["macd"] = macd_line
                if macd_signal is not None: tech["macd_signal"] = macd_signal
                if atr is not None: tech["atr"] = atr
                if cci is not None: tech["cci"] = cci
                if roc is not None: tech["roc"] = roc
                if mfi is not None: tech["mfi"] = mfi
                if obv is not None: tech["obv"] = obv
                # Simple VWAP approximation (daily): cumulative price*vol / cumulative vol
                pv = (close * vol).cumsum()
                v = vol.cumsum().replace(0, pd.NA)
                vwap = pv / v
                tech["vwap"] = vwap
        except Exception:
            pass

        # Use requirements text to drive delegation goals (heuristic)
        goals = []
        try:
            req = (docs.get("requirements_pdf", "") + "\n" + docs.get("project_description", "")).lower()
            if "agent" in req:
                goals.append("Build multi-agent workflows for research and trading insights.")
            if "free" in req or "no api key" in req:
                goals.append("Use only free data sources when possible.")
            if "dashboard" in req:
                goals.append("Render clean summaries and visuals in a Dash tab.")
        except Exception:
            pass

        msg = "SynthesisAgent: produced findings and goals."
        payload = {"overall": overall, "tech": tech, "goals": goals}
        return msg, payload

    def run(self, symbol: str, period: str = "6mo", interval: str = "1d", news_count: int = 5, include_news: bool = True) -> Dict[str, Any]:
        chat: List[Dict[str, str]] = []

        m, data_payload = self.data_agent(symbol, period, interval)
        chat.append({"agent": "DataAgent", "message": m})

        m, docs = self.requirements_agent()
        chat.append({"agent": "RequirementsAgent", "message": m})

        items: List[NewsItem] = []
        if include_news:
            m, items = self.news_agent(symbol, news_count)
            chat.append({"agent": "NewsAgent", "message": m})

        # Add a quick web search context for the symbol
        try:
            q = f"{symbol} stock technicals RSI MACD news"
            ws_msg, ws_res = self.web_search_agent(q, max_results=5)
            chat.append({"agent": "WebSearchAgent", "message": ws_msg})
        except Exception:
            ws_res = []
        m, synth = self.synthesis_agent(symbol, data_payload, items, docs)
        chat.append({"agent": "SynthesisAgent", "message": m})

        return {
            "chat": chat,
            "price": {"source": data_payload.get("source"), "hist": data_payload.get("hist")},
            "news": [i.__dict__ for i in items],
            "web": ws_res,
            "findings": synth,
        }

    def run_technical_analysis(
        self,
        tickers: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        indicators: List[str],
        length: int,
        rsi_length: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal_len: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs a detailed technical analysis for multiple tickers.
        This is the core function for the new research tab.
        """
        results = {}
        for ticker in tickers:
            
                # --- Data Agent ---
                if start_date and end_date:
                    msg, payload = self.data_agent_with_dates(
                        ticker, start=start_date, end=end_date, interval=timeframe
                    )
                else:
                    # Choose a sensible default period based on interval granularity
                    intr = (timeframe or "1d").lower()
                    if intr in {"1m","2m","5m","15m","30m","60m","90m","1h"}:
                        period = "5d"
                    elif intr in {"1d","5d"}:
                        period = "6mo"
                    elif intr in {"1wk"}:
                        period = "2y"
                    else:
                        period = "1y"
                    msg, payload = self.data_agent(ticker, period=period, interval=timeframe)
                hist = payload.get("hist")
                try:
                    from telemetry import log_event
                    log_event("technical.data", {
                        "ticker": ticker,
                        "interval": timeframe,
                        "rows": (0 if hist is None else int(getattr(hist, 'shape', [0])[0])),
                        "source": payload.get("source"),
                        "start_date": start_date,
                        "end_date": end_date,
                    })
                except Exception:
                    pass
                if hist is None or hist.empty:
                    results[ticker] = {"error": "No data found."}
                    continue

    def _get_technical_ai_summary(
        self, symbol: str, tech_summary: str, model_preference: str = "auto"
    ) -> Dict[str, Any]:
        """Calls an LLM to get a technical analysis recommendation.
        Honors AI provider selection via env var AI_PROVIDER in {auto|local|gemini|do}.
        """
        system_msg = textwrap.dedent(
            """
            You are an expert technical analyst. Based on the provided technical indicators for a stock, you will:
            1. Provide a recommendation: "Buy", "Sell", or "Hold".
            2. Provide a confidence score for your recommendation on a scale of 1 to 10.
            3. Provide a detailed justification for your recommendation, referencing the specific indicator values provided. Explain how the indicators support your conclusion. Structure this as markdown text.

            Respond in a JSON format with three keys: "recommendation", "confidence", "justification".
            Example:
            {
                "recommendation": "Hold",
                "confidence": 6,
                "justification": "The stock is currently in a consolidation phase. The price is trading between the SMA(20) and EMA(50), indicating a lack of clear trend. The RSI is near 50, further supporting a neutral stance. MACD is close to the signal line, suggesting no immediate momentum."
            }
            """
        )
        user_msg = f"Symbol: {symbol}\n\nTechnical Summary:\n{tech_summary}"

        # Read provider choice and common params
        bridge = getattr(self, "bridge", None)
        ai_provider = (os.getenv("AI_PROVIDER", model_preference) or "auto").strip().lower()
        try:
            max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", str(getattr(bridge, "local_llm_max_tokens", 4096))))
        except Exception:
            max_tokens = 4096
        try:
            temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", str(getattr(bridge, "local_llm_temperature", 0.1))))
        except Exception:
            temperature = 0.1

        def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(raw_text)
            except Exception:
                try:
                    import re as _re
                    m = _re.search(r"\{[\s\S]*\}", raw_text)
                    if m:
                        return json.loads(m.group(0))
                except Exception:
                    return None
            return None

        def _try_local_openai() -> Optional[Dict[str, Any]]:
            base_url = os.getenv("LOCAL_LLM_BASE_URL", getattr(bridge, "local_llm_base_url", "")).strip()
            model_name = os.getenv("LOCAL_LLM_MODEL", getattr(bridge, "local_llm_model", "")).strip()
            api_key = os.getenv("LOCAL_LLM_API_KEY", getattr(bridge, "local_llm_api_key", "not-needed"))
            if not base_url:
                return None
            try:
                base = base_url.rstrip("/")
                headers = {"Content-Type": "application/json"}
                if api_key and api_key != "not-needed":
                    headers["Authorization"] = f"Bearer {api_key}"
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                payload = {
                    "model": model_name or "auto",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                resp = _requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                    ) or data.get("choices", [{}])[0].get("text", "")
                    if raw_text:
                        parsed = _extract_json(raw_text)
                        if parsed:
                            return parsed
            except Exception as e:
                logging.warning(f"Local LLM (OpenAI-compatible) failed: {e}")
            return None

        def _try_do() -> Optional[Dict[str, Any]]:
            base_url = (os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1") or "").strip()
            api_key = (os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY") or "").strip()
            model_name = (os.getenv("DO_AI_MODEL") or "").strip() or "llama3-8b-instruct"
            if not base_url or not api_key:
                return None
            try:
                base = base_url.rstrip("/")
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                payload = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                resp = _requests.post(f"{base}/chat/completions", headers=headers, json=payload, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                    ) or data.get("choices", [{}])[0].get("text", "")
                    if raw_text:
                        parsed = _extract_json(raw_text)
                        if parsed:
                            return parsed
            except Exception as e:
                logging.warning(f"DigitalOcean Inference failed: {e}")
            return None

        def _try_bridge_local() -> Optional[Dict[str, Any]]:
            try:
                if self.bridge.analyst and hasattr(self.bridge.analyst, "_query_local_llm"):
                    msgs = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                    res = self.bridge.analyst._query_local_llm(msgs)
                    if isinstance(res, dict):
                        return res
            except Exception as e:
                logging.warning(f"Bridge local LLM failed: {e}")
            return None

        def _try_bridge_gemini() -> Optional[Dict[str, Any]]:
            try:
                if self.bridge.analyst and hasattr(self.bridge.analyst, "_query_gemini"):
                    msgs = [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ]
                    res = self.bridge.analyst._query_gemini(msgs)
                    if isinstance(res, dict):
                        return res
            except Exception as e:
                logging.warning(f"Bridge Gemini failed: {e}")
            return None

        # Decide provider order
        order: List[str] = []
        if ai_provider in {"local", "do", "gemini"}:
            order = [ai_provider]
        else:  # auto
            # Prefer configured local, then DO, then Gemini, then bridge local
            if os.getenv("LOCAL_LLM_BASE_URL"):
                order.append("local")
            if os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY"):
                order.append("do")
            order.extend(["gemini", "bridgelocal"])  # fallbacks

        for prov in order:
            if prov == "local":
                out = _try_local_openai()
                if out:
                    return out
            elif prov == "do":
                out = _try_do()
                if out:
                    return out
            elif prov == "gemini":
                out = _try_bridge_gemini()
                if out:
                    return out
            elif prov == "bridgelocal":
                out = _try_bridge_local()
                if out:
                    return out

        # As a final fallback try any remaining bridges
        out = _try_bridge_local() or _try_bridge_gemini()
        if out:
            return out
        return {}

    # ---------------- Fundamentals & Holistic Agents ----------------
    def company_profile_agent(self, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """Gather basic company profile info via yfinance.get_info where possible.
        Returns message and a dict with keys: name, sector, industry, website, description.
        """
        try:
            t_yf = _get_yf()
            info: Dict[str, Any] = {}
            try:
                info = t_yf.Ticker(symbol).get_info() or {}
            except Exception:
                info = {}
            out = {
                "name": info.get("longName") or info.get("shortName") or symbol,
                "sector": info.get("sector") or "",
                "industry": info.get("industry") or info.get("industryKey") or "",
                "website": info.get("website") or "",
                "description": info.get("longBusinessSummary") or "",
            }
            return f"CompanyProfileAgent: profile for {symbol}.", out
        except Exception as e:
            return f"CompanyProfileAgent error: {e}", {}

    def financial_data_agent_full(self, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """Extract raw fundamental financial data from yfinance: income, balance, cashflow (annual & quarterly)."""
        try:
            t_yf = _get_yf()
            t = t_yf.Ticker(symbol)
            payload: Dict[str, Any] = {}
            # Annual
            try:
                payload["financials"] = t.financials.copy()
            except Exception:
                payload["financials"] = None
            try:
                payload["balance_sheet"] = t.balance_sheet.copy()
            except Exception:
                payload["balance_sheet"] = None
            try:
                payload["cashflow"] = t.cashflow.copy()
            except Exception:
                payload["cashflow"] = None
            # Quarterly
            try:
                payload["q_financials"] = t.quarterly_financials.copy()
            except Exception:
                payload["q_financials"] = None
            try:
                payload["q_balance_sheet"] = t.quarterly_balance_sheet.copy()
            except Exception:
                payload["q_balance_sheet"] = None
            try:
                payload["q_cashflow"] = t.quarterly_cashflow.copy()
            except Exception:
                payload["q_cashflow"] = None
            return f"FinancialDataAgent: loaded frames for {symbol}.", payload
        except Exception as e:
            return f"FinancialDataAgent error: {e}", {}

    def _safe_get_row(self, df: Optional[pd.DataFrame], row_name_candidates: List[str]) -> Optional[pd.Series]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        idx = [str(i).strip().lower() for i in df.index]
        for name in row_name_candidates:
            n = name.strip().lower()
            if n in idx:
                try:
                    return df.iloc[idx.index(n)]
                except Exception:
                    pass
        # Try fuzzy contains
        for name in row_name_candidates:
            n = name.strip().lower()
            for i, lab in enumerate(idx):
                if n in lab:
                    try:
                        return df.iloc[i]
                    except Exception:
                        pass
        return None

    def fundamental_ratios_agent(
        self,
        symbol: str,
        fundamentals: Dict[str, Any],
        price_hist: Optional[pd.DataFrame],
    ) -> Tuple[str, Dict[str, Any]]:
        """Compute key ratios across multiple recent periods when possible. Returns dict with series (as dict) and latest snapshot."""
        try:
            fin = fundamentals.get("financials")
            bal = fundamentals.get("balance_sheet")
            q_fin = fundamentals.get("q_financials")
            q_bal = fundamentals.get("q_balance_sheet")

            # Prefer quarterly for TTM aggregates
            def _ttm(series_row: Optional[pd.Series]) -> Optional[float]:
                if series_row is None:
                    return None
                try:
                    vals = [float(x) for x in series_row.iloc[:4] if pd.notna(x)]
                    return float(sum(vals)) if vals else None
                except Exception:
                    return None

            total_rev_ttm = _ttm(self._safe_get_row(q_fin, ["Total Revenue", "Revenue"]))
            net_income_ttm = _ttm(self._safe_get_row(q_fin, ["Net Income", "NetIncome"]))
            gross_profit_ttm = _ttm(self._safe_get_row(q_fin, ["Gross Profit"]))

            # Balance sheet point-in-time (last column)
            def _last_val(series_row: Optional[pd.Series]) -> Optional[float]:
                if series_row is None:
                    return None
                try:
                    for x in series_row:
                        if pd.notna(x):
                            return float(x)
                except Exception:
                    pass
                return None

            total_assets = _last_val(self._safe_get_row(bal, ["Total Assets"]))
            total_liab = _last_val(self._safe_get_row(bal, ["Total Liab", "Total Liabilities"]))
            total_equity = _last_val(self._safe_get_row(bal, ["Total Stockholder Equity", "Total Equity", "Total Shareholder Equity"]))
            current_assets = _last_val(self._safe_get_row(bal, ["Total Current Assets", "Current Assets"]))
            current_liab = _last_val(self._safe_get_row(bal, ["Total Current Liabilities", "Current Liabilities"]))
            inventory = _last_val(self._safe_get_row(bal, ["Inventory"]))

            # Shares outstanding from fast_info if available
            shares_out = None
            try:
                t_yf = _get_yf()
                t = t_yf.Ticker(symbol)
                shares_out = getattr(t, "fast_info", {}).get("shares") if hasattr(t, "fast_info") else None
                if shares_out is None:
                    info = t.get_info() or {}
                    shares_out = info.get("sharesOutstanding")
            except Exception:
                shares_out = None

            last_price = None
            if isinstance(price_hist, pd.DataFrame) and not price_hist.empty:
                try:
                    last_price = float(price_hist["Close"].iloc[-1])
                except Exception:
                    pass

            # Ratios
            pe = None
            eps_ttm = None
            if net_income_ttm is not None and shares_out:
                try:
                    eps_ttm = float(net_income_ttm) / float(shares_out)
                    if last_price is not None and eps_ttm not in (0, None):
                        pe = float(last_price) / float(eps_ttm) if eps_ttm else None
                except Exception:
                    pass

            ps = (float(last_price) / (float(total_rev_ttm) / float(shares_out))) if (total_rev_ttm and shares_out and last_price) else None
            debt_to_equity = (float(total_liab) / float(total_equity)) if (total_liab and total_equity) else None
            roe = (float(net_income_ttm) / float(total_equity)) if (net_income_ttm and total_equity) else None
            roa = (float(net_income_ttm) / float(total_assets)) if (net_income_ttm and total_assets) else None
            gross_margin = (float(gross_profit_ttm) / float(total_rev_ttm)) if (gross_profit_ttm and total_rev_ttm) else None
            net_margin = (float(net_income_ttm) / float(total_rev_ttm)) if (net_income_ttm and total_rev_ttm) else None
            current_ratio = (float(current_assets) / float(current_liab)) if (current_assets and current_liab) else None
            quick_ratio = ((float(current_assets) - float(inventory or 0.0)) / float(current_liab)) if (current_assets and current_liab) else None

            # Secondary source: yfinance info for trailing metrics if missing
            info: Dict[str, Any] = {}
            try:
                t_yf = _get_yf()
                info = t_yf.Ticker(symbol).get_info() or {}
                if eps_ttm in (None, 0):
                    eps_ttm = info.get("trailingEps") or eps_ttm
                if pe in (None, 0):
                    pe = info.get("trailingPE") or pe
                if ps in (None, 0) and info.get("priceToSalesTrailing12Months"):
                    ps = info.get("priceToSalesTrailing12Months")
            except Exception:
                pass

            ratios = {
                "pe_ttm": pe,
                "ps_ttm": ps,
                "debt_to_equity": debt_to_equity,
                "roe": roe,
                "roa": roa,
                "gross_margin": gross_margin,
                "net_margin": net_margin,
                "current_ratio": current_ratio,
                "quick_ratio": quick_ratio,
                "eps_ttm": eps_ttm,
            }
            # Add validation notes
            notes: List[str] = []
            if ratios.get("pe_ttm") in (None, 0):
                notes.append("Missing P/E (TTM) after primary calc; attempted fallback to yfinance info.")
            if ratios.get("ps_ttm") in (None, 0):
                notes.append("Missing P/S (TTM) after primary calc; attempted fallback to yfinance info.")
            if notes:
                ratios["notes"] = "; ".join(notes)
            return "FundamentalRatiosAgent: computed ratios.", ratios
        except Exception as e:
            return f"FundamentalRatiosAgent error: {e}", {}

    def earnings_agent(self, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """Collect EPS and Revenue history using modern methods and compute growth rates."""
        try:
            t_yf = _get_yf()
            t = t_yf.Ticker(symbol)
            out: Dict[str, Any] = {}
            
            # --- Quarterly Earnings and Revenue (Modern Method) ---
            q_fin = None
            try:
                # Use .quarterly_financials, which is the income statement
                q_fin = t.quarterly_financials
            except Exception:
                q_fin = None

            if isinstance(q_fin, pd.DataFrame) and not q_fin.empty:
                # Return the DataFrame slice itself, as the caller expects it.
                out["quarterly_earnings"] = q_fin.tail(8)
                
                rev_row = self._safe_get_row(q_fin, ["Total Revenue", "Revenue"])
                earn_row = self._safe_get_row(q_fin, ["Net Income"])
                
                if rev_row is not None:
                    rev = rev_row.astype(float)
                    rev_growth = (rev.pct_change(fill_method=None) * 100.0).round(2).tolist()
                    out["revenue_growth_pct"] = rev_growth
                
                if earn_row is not None:
                    earn = earn_row.astype(float)
                    earn_growth = (earn.pct_change(fill_method=None) * 100.0).round(2).tolist()
                    out["earnings_growth_pct"] = earn_growth
            
            # --- Annual EPS (Modern Method) ---
            try:
                fin = t.financials # Use the annual income statement
                eps_row = None
                # Basic EPS is usually the most reliable figure
                for key in ["Basic EPS", "BasicEPS", "Diluted EPS", "DilutedEPS"]:
                    if key in (fin.index if isinstance(fin, pd.DataFrame) else []):
                        eps_row = fin.loc[key]
                        break
                if eps_row is not None:
                    # Drop any missing values and convert to a simple list of floats
                    out["annual_eps"] = pd.Series(eps_row).dropna().astype(float).tolist()
            except Exception:
                pass
                
            return "EarningsAgent: collected earnings data.", out
            
        except Exception as e:
            return f"EarningsAgent error: {e}", {}

    def _figure_revenue_growth(self, symbol: str, q_financials: Optional[pd.DataFrame]) -> Optional[go.Figure]:
        try:
            if q_financials is None or q_financials.empty:
                return None
            row = self._safe_get_row(q_financials, ["Total Revenue", "Revenue"]) or self._safe_get_row(q_financials, ["Operating Revenue"])
            if row is None:
                return None
            s = pd.Series(row).dropna().astype(float).iloc[::-1]
            fig = go.Figure()
            fig.add_bar(x=[str(i) for i in s.index], y=s.values, name="Revenue")
            fig.update_layout(title=f"{symbol} Quarterly Revenue", template="plotly_dark", height=300, margin=dict(l=10,r=10,t=40,b=10))
            return fig
        except Exception:
            return None

    def _figure_pe_trend(self, symbol: str, price_hist: Optional[pd.DataFrame], eps_ttm: Optional[float]) -> Optional[go.Figure]:
        try:
            if price_hist is None or price_hist.empty or not eps_ttm or eps_ttm == 0:
                return None
            close = price_hist["Close"].astype(float).tail(120)
            pe_series = close / float(eps_ttm)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=close.index, y=pe_series, name="P/E (approx)", line=dict(color="#3b82f6")))
            fig.update_layout(title=f"{symbol} Approx. P/E Trend (TTM EPS)", template="plotly_dark", height=300, margin=dict(l=10,r=10,t=40,b=10))
            return fig
        except Exception:
            return None

    def _llm_generate(self, system_msg: str, user_msg: str, ai_provider: str = "auto") -> Optional[str]:
        """
        Generic LLM text generation that RESPECTS the user's selected provider.
        Provider priority:
        1. The one explicitly passed in `ai_provider`.
        2. If `ai_provider` is 'auto', it tries Local, DO, HF, then Gemini as a last resort.
        """
        def _log_llm(event: str, payload: Dict[str, Any]) -> None:
            try:
                from telemetry import log_event
                pl = dict(payload)
                for k in ("system_msg", "user_msg", "response"):
                    if isinstance(pl.get(k), str) and len(pl[k]) > 1200:
                        pl[k] = pl[k][:1200] + "..."
                log_event(event, pl)
            except Exception:
                pass

        # --- Define all possible provider calls as separate functions ---
        def _try_local_openai() -> Optional[str]:
            try:
                base_url = (os.getenv("LOCAL_LLM_BASE_URL", "") or getattr(self.bridge, "local_llm_base_url", "")).strip().rstrip("/")
                model_name = (os.getenv("LOCAL_LLM_MODEL", "") or getattr(self.bridge, "local_llm_model", "")).strip() or "auto"
                api_key = (os.getenv("LOCAL_LLM_API_KEY", "") or getattr(self.bridge, "local_llm_api_key", "not-needed")).strip()
                if not base_url:
                    return None
                
                _log_llm("llm.request", {"provider": "local", "model": model_name, "system_msg": system_msg, "user_msg": user_msg})
                headers = {"Content-Type": "application/json"}
                if api_key and api_key != "not-needed":
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": model_name, "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2") or 0.2),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096), "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    raw = (data.get("choices", [{}])[0].get("message", {}).get("content")) or (data.get("choices", [{}])[0].get("text", ""))
                    if raw:
                        _log_llm("llm.response", {"provider": "local", "response": raw, "usage": data.get("usage")})
                        return raw
            except Exception as e:
                _log_llm("llm.error", {"provider": "local", "error": str(e)})
            return None

        def _try_do() -> Optional[str]:
            try:
                base_url = (os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1") or "").strip().rstrip("/")
                api_key = (os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY") or "").strip()
                model_name = (os.getenv("DO_AI_MODEL") or "").strip() or "llama3-8b-instruct"
                if not (base_url and api_key):
                    return None
                
                _log_llm("llm.request", {"provider": "do", "model": model_name, "system_msg": system_msg, "user_msg": user_msg})
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model_name, "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2") or 0.2),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096), "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    raw = (data.get("choices", [{}])[0].get("message", {}).get("content")) or (data.get("choices", [{}])[0].get("text", ""))
                    if raw:
                        _log_llm("llm.response", {"provider": "do", "response": raw, "usage": data.get("usage")})
                        return raw
            except Exception as e:
                _log_llm("llm.error", {"provider": "do", "error": str(e)})
            return None

        def _try_hf() -> Optional[str]:
            try:
                base_url = (os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1") or "").strip().rstrip("/")
                api_key = (os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN2") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
                model_name = (os.getenv("HF_MODEL") or "OpenAI/gpt-oss-20B").strip()
                if not (base_url and api_key):
                    return None
                
                _log_llm("llm.request", {"provider": "hf", "model": model_name, "system_msg": system_msg, "user_msg": user_msg})
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model_name, "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2") or 0.2),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096), "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    raw = (data.get("choices", [{}])[0].get("message", {}).get("content")) or (data.get("choices", [{}])[0].get("text", ""))
                    if raw:
                        _log_llm("llm.response", {"provider": "hf", "response": raw, "usage": data.get("usage")})
                        return raw
            except Exception as e:
                _log_llm("llm.error", {"provider": "hf", "error": str(e)})
            return None
        
        def _try_gemini() -> Optional[str]:
            try:
                if self.bridge and hasattr(self.bridge, "analyst") and hasattr(self.bridge.analyst, "_query_gemini"):
                    _log_llm("llm.request", {"provider": "gemini", "system_msg": system_msg, "user_msg": user_msg})
                    out = self.bridge.analyst._query_gemini([{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}])
                    if isinstance(out, str) and out.strip():
                        _log_llm("llm.response", {"provider": "gemini", "response": out})
                        return out
            except Exception as e:
                _log_llm("llm.error", {"provider": "gemini", "error": str(e)})
            return None
        
        # --- NEW, SMARTER PROVIDER SELECTION LOGIC ---
        provider_map = {
            "local": _try_local_openai,
            "do": _try_do,
            "hf": _try_hf,
            "gemini": _try_gemini
        }

        provider_order = []
        
        # If a specific provider is chosen, try it first.
        if ai_provider in provider_map:
            provider_order.append(ai_provider)
        
        # For 'auto' or as a fallback, add the rest in a sensible order (Gemini LAST)
        for p in ["local", "do", "hf", "gemini"]:
            if p not in provider_order:
                provider_order.append(p)
                
        print(f"DEBUG: LLM provider order: {provider_order}")

        # --- Execute the calls in the determined order ---
        for provider_name in provider_order:
            call_func = provider_map.get(provider_name)
            if call_func:
                result = call_func()
                if isinstance(result, str) and result.strip():
                    print(f"DEBUG: LLM call successful with provider: {provider_name}")
                    return result
        
        print("DEBUG: All LLM providers failed.")
        return None

    def holistic_analysis_agent(self, symbol: str, notebook: Dict[str, Any], ai_provider: str = "auto") -> Tuple[str, str]:
        """Create a holistic narrative from the Analyst's Notebook. Uses LLM if available; else a clear heuristic synthesis with anomaly flags."""
        # Build structured context
        prof = notebook.get("profile") or {}
        ratios = (notebook.get("financials") or {}).get("ratios") or {}
        earnings = (notebook.get("financials") or {}).get("earnings") or {}
        news = notebook.get("news_sentiment") or {}
        tech = notebook.get("technical") or {}
        errors = notebook.get("errors") or []

        # Compose a clean JSON snapshot for the model
        snapshot = {
            "symbol": symbol,
            "profile": prof,
            "financials": {"ratios": ratios, "earnings": earnings},
            "news_sentiment": news,
            "technical": tech,
            "errors": errors,
        }

        system_msg = (
            "You are a senior financial analyst. Based on the structured 'Analyst's Notebook' JSON, write a comprehensive, multi-paragraph summary. "
            "You must explicitly reference the key findings from the technical analysis section." 
            "Start with the company profile. Then synthesize the key findings from financial ratios, earnings, technicals, and news sentiment. "
            "Connect the dots between sections (e.g., AI concerns in news vs. R&D or guidance). Explicitly call out any missing or anomalous data. "
            "Be balanced and concrete. 250-400 words."
        )
        user_msg = json.dumps(snapshot, default=str)

        llm_text = self._llm_generate(system_msg, user_msg, ai_provider=ai_provider)
        if isinstance(llm_text, str) and llm_text.strip():
            return "HolisticAnalysisAgent: LLM", llm_text.strip()

        # Heuristic fallback with anomaly detection
        def _fmt_pct(x):
            try:
                return f"{float(x)*100:.1f}%"
            except Exception:
                return "n/a"
        lines: List[str] = []
        if prof:
            lines.append(f"Company: {prof.get('name', symbol)} | Sector: {prof.get('sector','')} | Industry: {prof.get('industry','')}")
        if ratios:
            lines.append(
                "Valuation & Profitability: "
                f"P/E={ratios.get('pe_ttm')} | P/S={ratios.get('ps_ttm')} | D/E={ratios.get('debt_to_equity')} | "
                f"Gross Margin={_fmt_pct(ratios.get('gross_margin'))} | Net Margin={_fmt_pct(ratios.get('net_margin'))}"
            )
        if earnings:
            rg = earnings.get('revenue_growth_pct')
            eg = earnings.get('earnings_growth_pct')
            if rg or eg:
                lines.append(f"Earnings Momentum: revenue q/q % {rg[-4:] if isinstance(rg, list) else 'n/a'}; earnings q/q % {eg[-4:] if isinstance(eg, list) else 'n/a'}")
        if tech:
            lines.append(f"Technicals: {tech.get('summary','No technical summary')}. Rec {tech.get('recommendation','N/A')} (Conf {tech.get('confidence','N/A')}).")
        if news and news.get('summary'):
            lines.append("News & Sentiment: " + str(news.get('summary'))[:500])
        if errors:
            lines.append("Data Quality Notes: " + "; ".join(errors))
        # Flag critical missing metrics
        if ratios and (ratios.get('pe_ttm') in (None, 0) or ratios.get('ps_ttm') in (None, 0)):
            lines.append("Warning: Core valuation metrics (P/E or P/S) missing; valuation conclusions may be unreliable.")
        return "HolisticAnalysisAgent: heuristic", "\n".join([l for l in lines if l])

    def recommendation_agent(self, symbol: str, notebook: Dict[str, Any], ai_provider: str = "auto") -> Tuple[str, Dict[str, Any]]:
        """Provide Buy/Hold/Sell with confidence and detailed justification using a summarized notebook."""
        # --- 1. Create a CONCISE summary of the notebook for the LLM ---
        # This is more reliable than sending the entire raw JSON object.

        ratios = (notebook.get("financials") or {}).get("ratios") or {}
        tech = notebook.get("technical") or {}
        news = notebook.get("news_sentiment") or {}
        profile = notebook.get("profile") or {}

        context_summary = f"""
        - Company: {profile.get('name', symbol)}, Sector: {profile.get('sector', 'N/A')}
        - Key Ratios: P/E={ratios.get('pe_ttm'):.2f}, P/S={ratios.get('ps_ttm'):.2f}, D/E={ratios.get('debt_to_equity'):.2f}
        - Technical Signal: Recommendation='{tech.get('recommendation', 'N/A')}', Confidence={tech.get('confidence', 'N/A')}
        - News Summary: {str(news.get('summary', 'No news summary available.'))[:800]}
        - Data Quality Notes: {notebook.get('errors', 'None')}
        """

        # --- 2. Use a more direct system prompt ---

        system_msg = (
            "You are an investment strategist. Based on the provided summary, you MUST return ONLY a single, valid JSON object. "
            "Do not include any other text, reasoning, or markdown formatting. Your entire response must be the JSON. "
            "The justification must be a coherent, multi-sentence paragraph synthesizing the provided data."
        )
        user_msg = (
            "Synthesize the following data into a final recommendation. "
            f"Return JSON with keys: 'recommendation' (Buy/Hold/Sell), 'confidence' (1-10), and 'justification' (a string).\n\n"
            f"DATA:\n{context_summary}"
        )

        # --- 3. Attempt LLM call ---

        llm_text = self._llm_generate(system_msg, user_msg, ai_provider=ai_provider)
        if llm_text:
            try:
                # Use the robust JSON extractor
                parsed = self._extract_json_object(llm_text)
                if isinstance(parsed, dict) and parsed.get("recommendation") and parsed.get("justification"):
                    # Clean up justification from potential LLM artifacts
                    parsed["justification"] = parsed["justification"].replace("Overall summary (fallback):", "").strip()
                    return "RecommendationAgent: LLM", parsed
            except Exception:
                # Fall through to the heuristic if JSON parsing fails
                pass

        # --- 4. A SMARTER Heuristic Fallback ---
        # This now attempts to synthesize the data instead of just listing it.

        just_lines = []
        confidence_score = 5 # Start at neutral

        # Factor in technicals
        if tech.get("recommendation"):
            just_lines.append(f"The technical analysis suggests a '{tech.get('recommendation')}' with a confidence of {tech.get('confidence', 'N/A')}.")
            if tech.get("recommendation", "").lower() == "buy":
                confidence_score += 2
            elif tech.get("recommendation", "").lower() == "sell":
                confidence_score -= 2

        # Factor in news sentiment
        if news.get("summary"):
            # A simple check for positive/negative keywords in the news
            news_lower = news.get("summary", "").lower()
            if any(kw in news_lower for kw in ["positive", "optimistic", "upgrade", "outperform", "strong"]):
                confidence_score += 1
                just_lines.append("News sentiment appears positive, with analysts noting optimistic outlooks.")
            elif any(kw in news_lower for kw in ["negative", "concerns", "downgrade", "risk", "weak"]):
                confidence_score -= 1
                just_lines.append("News sentiment raises some concerns, pointing to potential risks.")
    
        # Determine final recommendation based on the synthesized score
        final_rec = "Hold"
        if confidence_score >= 7:
            final_rec = "Buy"
        elif confidence_score <= 3:
            final_rec = "Sell"

        if ratios:
            just_lines.append(f"Key valuation metrics include a P/E of {ratios.get('pe_ttm'):.2f} and Debt-to-Equity of {ratios.get('debt_to_equity'):.2f}.")

        return "RecommendationAgent: heuristic", {
            "recommendation": final_rec,
            "confidence": max(1, min(10, confidence_score)), # Clamp score between 1 and 10
            "justification": " ".join(just_lines) or "A recommendation was determined by heuristic analysis of available data.",
        }

    def fundamental_synthesis_agent(self, symbol: str, notebook: Dict[str, Any], ai_provider: str = "auto") -> Tuple[str, str]:
        """
        Uses an LLM to create a narrative, easy-to-understand summary of the company's fundamental health.
        """
        profile = notebook.get("profile", {})
        ratios = notebook.get("financials", {}).get("ratios", {})
        
        # Filter out empty values for a cleaner prompt
        ratios_for_prompt = {k: v for k, v in ratios.items() if v is not None}
        
        if not ratios_for_prompt:
            return "FundamentalSynthesisAgent: No ratio data to analyze.", "Fundamental data was incomplete, so a summary could not be generated."

        system_msg = textwrap.dedent("""
            You are a skilled financial analyst who excels at explaining complex financial data to non-expert investors.
            Your tone is clear, educational, and objective. You avoid jargon where possible.
        """)
        
        user_msg = textwrap.dedent(f"""
            Please analyze the following financial data for the company: **{profile.get('name', symbol)}**.
            Provide a narrative summary in markdown format that explains what these numbers mean.

            **Financial Data Snapshot:**
            ```json
            {json.dumps(ratios_for_prompt, indent=2, default=str)}
            ```

            **Instructions:**
            1.  **Start with a brief overview** of the company's fundamental picture based on the data.
            2.  Create sections using markdown headings (e.g., `### Valuation`, `### Profitability`, `### Financial Health`).
            3.  In each section, explain the key ratios. For example, when discussing the P/E ratio, briefly state what it measures (e.g., "The P/E ratio of {ratios.get('pe_ttm', 'N/A'):.2f} suggests investors are willing to pay...").
            4.  Connect the numbers to potential insights (e.g., "A high Debt-to-Equity ratio might indicate risk, while a strong ROE shows efficient use of shareholder money.").
            5.  Conclude with a summary of the fundamental strengths and potential weaknesses you've identified.
        """)
        
        summary_text = self._llm_generate(system_msg, user_msg, ai_provider=ai_provider)
        
        if not summary_text:
            return "FundamentalSynthesisAgent: LLM call failed.", "AI-powered summary could not be generated."
            
        return "FundamentalSynthesisAgent: Generated narrative summary.", summary_text

    def execute_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a research workflow based on a configuration dictionary.
        This is the single entry point for all research tasks.
        """
        symbol = config.get("symbol")
        analysis_type = config.get("analysis_type", "combined")
        ai_provider = config.get("ai_provider", "auto")

        print(f"\nDEBUG: Executing workflow '{analysis_type}' for {symbol}...")

        # Initialize the outputs and the shared "notebook"
        outputs: Dict[str, Any] = {"plan": {"analysis_type": analysis_type, "symbol": symbol}}
        notebook: Dict[str, Any] = {"symbol": symbol, "profile": {}, "financials": {"ratios": {}, "earnings": {}}, "technical": {}, "news_sentiment": {}, "web": [], "errors": []}
        conversations: List[Dict[str, str]] = []

        # --- Agent Execution Logic ---
        
        # News & Web (Run first for context)
        if analysis_type in ("web", "combined"):
            try:
                news_items = self.bridge.fetch(symbol, days_back=7, max_articles=6, model_preference=ai_provider, analyze=True)
                outputs["news"] = [i.__dict__ for i in news_items]
                news_summary = self.bridge.summarize_overall(symbol, news_items, model_preference=ai_provider, max_chars=2500)
                outputs["news_summary"] = news_summary
                if isinstance(news_summary, dict) and news_summary.get("summary"):
                    notebook["news_sentiment"] = {"summary": news_summary.get("summary"), "provider": news_summary.get("provider")}
                conversations.append({"agent": "NewsAgent", "message": f"Fetched {len(news_items)} news items and generated summary for {symbol}"})
            except Exception as e:
                notebook["errors"].append(f"News analysis failed: {e}")
                conversations.append({"agent": "NewsAgent", "message": f"Failed to fetch news: {str(e)}"})

        # Profile & Fundamentals
        fundamentals, price_df = {}, None
        if analysis_type in ("fundamental", "combined"):
            try:
                _m, profile = self.company_profile_agent(symbol)
                outputs["company_profile"] = notebook["profile"] = profile
                _m, fundamentals = self.financial_data_agent_full(symbol)
                outputs["fundamentals"] = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else None) for k, v in fundamentals.items()}
                conversations.append({"agent": "ProfileAgent", "message": f"Retrieved company profile and financial data for {symbol}"})
            except Exception as e:
                notebook["errors"].append(f"Fundamental data fetching failed: {e}")
                conversations.append({"agent": "ProfileAgent", "message": f"Failed to fetch fundamental data: {str(e)}"})

        # Technicals
        if analysis_type in ("technical", "combined"):
            try:
                start_date = config.get("start_date")
                end_date = config.get("end_date")
                interval = config.get("interval", "1d")
                indicators = config.get("indicators", ["SMA", "EMA", "Bollinger Bands", "VWAP", "RSI", "MACD", "ATR"])
                length = config.get("length", 20)
                rsi_length = config.get("rsi_length", 14)
                macd_fast = config.get("macd_fast", 12)
                macd_slow = config.get("macd_slow", 26)
                macd_signal_len = config.get("macd_signal_len", 9)
                
                if start_date and end_date: _m, payload = self.data_agent_with_dates(symbol, start_date, end_date, interval)
                else: _m, payload = self.data_agent(symbol, period="1y", interval=interval)
                price_df = payload.get("hist")
                
                tech_fig, tech_summary = self.synthesis_agent_for_technicals(price_df, indicators, length, rsi_length, macd_fast, macd_slow, macd_signal_len)
                ai_rec = self._get_technical_ai_summary(symbol, tech_summary, model_preference=ai_provider) or {}
                _m, strategy_text = self.technical_strategist_agent(symbol, price_df, tech_summary, ai_provider=ai_provider)
                
                outputs["technical"] = notebook["technical"] = {
                    "summary": tech_summary,
                    "recommendation": ai_rec.get("recommendation", "N/A"),
                    "confidence": ai_rec.get("confidence", "N/A"),
                    "justification": strategy_text or ai_rec.get("justification", "No analysis available.")
                }
                if tech_fig: outputs.setdefault("figures", {})["technical"] = tech_fig
                conversations.append({"agent": "TechnicalAgent", "message": f"Completed technical analysis for {symbol} with recommendation: {ai_rec.get('recommendation', 'N/A')}"})
            except Exception as e:
                notebook["errors"].append(f"Technical analysis failed: {e}")
                conversations.append({"agent": "TechnicalAgent", "message": f"Technical analysis failed: {str(e)}"})

        # Ratios & Earnings (Depends on Fundamentals and sometimes Price)
        if analysis_type in ("fundamental", "combined"):
            try:
                _m, ratios = self.fundamental_ratios_agent(symbol, fundamentals, price_df)
                outputs["ratios"] = notebook["financials"]["ratios"] = ratios
                _m, earnings = self.earnings_agent(symbol)
                outputs["earnings"] = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in earnings.items()}
                notebook["financials"]["earnings"] = earnings
                conversations.append({"agent": "RatiosAgent", "message": f"Calculated financial ratios and earnings data for {symbol}"})
            except Exception as e:
                notebook["errors"].append(f"Ratios/Earnings calculation failed: {e}")
                conversations.append({"agent": "RatiosAgent", "message": f"Failed to calculate ratios/earnings: {str(e)}"})

        # Fundamental Synthesis
        if analysis_type in ("fundamental", "combined"):
            try:
                _m, fund_summary = self.fundamental_synthesis_agent(symbol, notebook, ai_provider=ai_provider)
                outputs["fundamental_summary"] = fund_summary
                conversations.append({"agent": "FundamentalSynthesisAgent", "message": f"Generated narrative fundamental summary for {symbol}"})
            except Exception as e:
                notebook["errors"].append(f"Fundamental Synthesis failed: {e}")

        # Holistic Analysis (Depends on all previous steps)
        if analysis_type in ("fundamental", "combined", "web"):
            try:
                _m, hol = self.holistic_analysis_agent(symbol, notebook, ai_provider=ai_provider)
                outputs["holistic"] = hol
                conversations.append({"agent": "HolisticAgent", "message": f"Generated holistic analysis for {symbol}"})
            except Exception as e:
                notebook["errors"].append(f"Holistic analysis failed: {e}")
                outputs["holistic"] = "Holistic analysis could not be generated due to an earlier error."
                conversations.append({"agent": "HolisticAgent", "message": f"Holistic analysis failed: {str(e)}"})

        # Recommendation (Depends on all previous steps)
        if analysis_type in ("technical", "combined", "fundamental"):
            try:
                _m, rec = self.recommendation_agent(symbol, notebook, ai_provider=ai_provider)
                outputs["recommendation"] = rec
                conversations.append({"agent": "RecommendationAgent", "message": f"Generated final recommendation for {symbol}: {rec.get('recommendation', 'N/A')}"})
            except Exception as e:
                notebook["errors"].append(f"Recommendation generation failed: {e}")
                outputs["recommendation"] = {"recommendation": "Hold", "confidence": "N/A", "justification": "Could not be generated."}
                conversations.append({"agent": "RecommendationAgent", "message": f"Recommendation generation failed: {str(e)}"})
        
        # Visualizations
        try:
            q_fin = fundamentals.get("q_financials")
            rev_fig = self._figure_revenue_growth(symbol, q_fin)
            if rev_fig: outputs.setdefault("figures", {})["revenue"] = rev_fig
            
            eps_ttm = (outputs.get("ratios") or {}).get("eps_ttm")
            pe_fig = self._figure_pe_trend(symbol, price_df, eps_ttm)
            if pe_fig: outputs.setdefault("figures", {})["pe_trend"] = pe_fig
        except Exception as e:
            print(f"WARNING: Figure generation failed: {e}")

        print("DEBUG: Analysis pipeline complete.")
        outputs["conversations"] = conversations
        return outputs



def quick_research(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Lightweight web research helper using duckduckgo_search if installed, otherwise fallback to a simple web request.

    Returns a list of dicts with keys: title, url, snippet.
    """
    out: List[Dict[str, str]] = []
    if not query:
        return out
    if _DDGS is not None:
        try:
            with _DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    out.append({
                        "title": (r.get("title") or "")[:200],
                        "url": r.get("href") or r.get("url") or "",
                        "snippet": (r.get("body") or r.get("snippet") or "")[:400],
                    })
            return out
        except Exception:
            pass
    # Fallback: try a public search API is not provided; return empty to keep it safe.
    return out