import os
import sys
import pathlib
import requests
import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
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
    ai_summary: str = ""
    sentiment_label: str = ""
    sentiment_score: float = 0.0
    analysis_provider: str = ""

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
        # Local LLM tunables from environment (for customization)
        self.local_llm_max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096)
        self.local_llm_temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1") or 0.1)
        self.local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
        self.local_llm_api_key = os.getenv("LOCAL_LLM_API_KEY", "not-needed").strip()
        self.local_llm_model = os.getenv("LOCAL_LLM_MODEL", "gpt-4o-mini-compat").strip()

    # --- helpers to sanitize LLM outputs ---
    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            j = json.loads(text)
            if isinstance(j, dict):
                return j
        except Exception:
            pass
        start = text.find("{")
        while start != -1 and start < len(text):
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : i + 1]
                        try:
                            j = json.loads(chunk)
                            if isinstance(j, dict):
                                return j
                        except Exception:
                            break
            start = text.find("{", start + 1)
        return None

    def fetch(self, asset: str, days_back: int = 7, max_articles: int = 10, *, model_preference: str = "auto", analyze: bool = True) -> List[NewsItem]:
        try:
            self.logger.info("news.fetch: asset=%s days=%s max=%s model_pref=%s analyze=%s", asset, days_back, max_articles, model_preference, analyze)
        except Exception:
            pass
        if not self.fetcher:
            # Minimal no-deps fallback
            return [
                NewsItem(title=f"{asset} placeholder item", url="#", description="Wire marketNews to enable live articles.", published_at="", source="local")
            ]
        arts = self.fetcher.fetch_news(asset, days_back, max_articles)
        # Save fetch attempts summary if available
        try:
            self._last_fetch_summary = ""
            if hasattr(self.fetcher, 'get_last_attempts_summary'):
                self._last_fetch_summary = self.fetcher.get_last_attempts_summary() or ""
        except Exception:
            self._last_fetch_summary = ""

        # Best-effort: enrich content for the first few items
        try:
            for a in arts[:5]:
                if not getattr(a, 'content', None):
                    content = self.fetcher.scrape_article_content(a.url)
                    if content:
                        a.content = content
        except Exception:
            pass

        # Run AI analysis to fill summary and sentiment
        if analyze and self.analyst:
            logs: List[str] = []
            # If user selected DigitalOcean, prefer DO-backed analysis here
            use_do = (str(model_preference or "").lower() == "do") or (os.getenv("AI_PROVIDER", "").strip().lower() == "do")
            do_base = (os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1") or "").strip().rstrip("/")
            do_key = (os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY") or "").strip()
            do_model = (os.getenv("DO_AI_MODEL") or "").strip() or "llama3-8b-instruct"

            # If user selected Hugging Face, prefer HF-backed analysis
            use_hf = (str(model_preference or "").lower() == "hf") or (os.getenv("AI_PROVIDER", "").strip().lower() == "hf")
            hf_base = (os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1") or "").strip().rstrip("/")
            hf_key = (os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN2") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
            hf_model = (os.getenv("HF_MODEL") or "OpenAI/gpt-oss-20B").strip()
            try:
                self.logger.info("news.fetch: providers use_do=%s use_hf=%s have_keys do=%s hf=%s", use_do, use_hf, bool(do_key), bool(hf_key))
                if use_do and not (do_base and do_key):
                    self.logger.warning("news.fetch: DO selected but base/key missing base=%s key_present=%s", bool(do_base), bool(do_key))
                if use_hf and not (hf_base and hf_key):
                    self.logger.warning("news.fetch: HF selected but base/key missing base=%s key_present=%s", bool(hf_base), bool(hf_key))
            except Exception:
                pass

            def _analyze_with_do(title: str, description: str, content: str) -> Optional[Dict[str, Any]]:
                if not (do_base and do_key):
                    return None
                try:
                    system_msg = (
                        "You are a financial news analyst. Given an article, return a compact JSON with keys: "
                        "'summary' (2-4 sentences), 'sentiment_label' (bullish|bearish|neutral), and 'sentiment_score' (a float between -1 and 1)."
                    )
                    user_msg = textwrap.dedent(f"""
                        Title: {title}
                        Description: {description}
                        Content: {content[:8000] if content else ''}

                        Return only a JSON object, no extra text. Do not include any chain-of-thought or analysis outside JSON.
                    """)
                    payload = {
                        "model": do_model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.2,
                        "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096),
                        "stream": False,
                    }
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {do_key}"}
                    r = _requests.post(f"{do_base}/chat/completions", headers=headers, json=payload, timeout=45)
                    try:
                        self.logger.info("news.HF/DO: DO POST /chat/completions model=%s status=%s", do_model, r.status_code)
                    except Exception:
                        pass
                    if r.status_code != 200:
                        try:
                            self.logger.warning("news.DO error: %s", (r.text or "")[:200])
                        except Exception:
                            pass
                        return None

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
                                if self.bridge and getattr(self.bridge, "analyst", None) and hasattr(self.bridge.analyst, "_query_local_llm"):
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
                                if self.bridge and getattr(self.bridge, "analyst", None) and hasattr(self.bridge.analyst, "_query_gemini"):
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
                    data = r.json()
                    raw = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                        or data.get("choices", [{}])[0].get("text", "")
                    )
                    if not raw:
                        return None
                    parsed = NewsBridge._extract_json_object(raw)
                    if isinstance(parsed, dict):
                        # Normalize keys
                        out = {
                            "summary": parsed.get("summary") or parsed.get("ai_summary") or "",
                            "sentiment_label": (parsed.get("sentiment_label") or "").lower(),
                            "sentiment_score": float(parsed.get("sentiment_score", 0) or 0.0),
                        }
                        return out
                except Exception as e:
                    try:
                        self.logger.exception("news.DO exception: %s", e)
                    except Exception:
                        pass
                    return None
                return None

            def _analyze_with_hf(title: str, description: str, content: str) -> Optional[Dict[str, Any]]:
                if not (hf_base and hf_key):
                    return None
                try:
                    system_msg = (
                        "You are a financial news analyst. Given an article, return a compact JSON with keys: "
                        "'summary' (2-4 sentences), 'sentiment_label' (bullish|bearish|neutral), and 'sentiment_score' (a float between -1 and 1)."
                    )
                    user_msg = textwrap.dedent(f"""
                        Title: {title}
                        Description: {description}
                        Content: {content[:8000] if content else ''}

                        Return only a JSON object, no extra text. Do not include any chain-of-thought or analysis outside JSON.
                    """)
                    payload = {
                        "model": hf_model,
                        "messages": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        "temperature": 0.2,
                        "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096),
                        "stream": False,
                    }
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {hf_key}"}
                    r = _requests.post(f"{hf_base}/chat/completions", headers=headers, json=payload, timeout=45)
                    try:
                        self.logger.info("news.HF/DO: HF POST /chat/completions model=%s status=%s", hf_model, r.status_code)
                    except Exception:
                        pass
                    if r.status_code != 200:
                        try:
                            self.logger.warning("news.HF error: %s", (r.text or "")[:200])
                        except Exception:
                            pass
                        return None
                    data = r.json()
                    raw = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                        or data.get("choices", [{}])[0].get("text", "")
                    )
                    if not raw:
                        return None
                    parsed = NewsBridge._extract_json_object(raw)
                    if isinstance(parsed, dict):
                        return {
                            "summary": parsed.get("summary") or parsed.get("ai_summary") or "",
                            "sentiment_label": (parsed.get("sentiment_label") or "").lower(),
                            "sentiment_score": float(parsed.get("sentiment_score", 0) or 0.0),
                        }
                except Exception as e:
                    try:
                        self.logger.exception("news.HF exception: %s", e)
                    except Exception:
                        pass
                    return None
                return None

            for a in arts:
                try:
                    if use_do:
                        res = _analyze_with_do(getattr(a, 'title', '') or '', getattr(a, 'description', '') or '', getattr(a, 'content', '') or '')
                        if res:
                            setattr(a, 'ai_summary', res.get('summary', '') or '')
                            setattr(a, 'sentiment_label', res.get('sentiment_label', '') or '')
                            try:
                                setattr(a, 'sentiment_score', float(res.get('sentiment_score', 0.0) or 0.0))
                            except Exception:
                                setattr(a, 'sentiment_score', 0.0)
                            setattr(a, 'analysis_provider', 'do')
                            logs.append(f"{asset} | DO | {(a.title or '')[:80]}")
                            continue  # done with this article
                    if use_hf:
                        res = _analyze_with_hf(getattr(a, 'title', '') or '', getattr(a, 'description', '') or '', getattr(a, 'content', '') or '')
                        if res:
                            setattr(a, 'ai_summary', res.get('summary', '') or '')
                            setattr(a, 'sentiment_label', res.get('sentiment_label', '') or '')
                            try:
                                setattr(a, 'sentiment_score', float(res.get('sentiment_score', 0.0) or 0.0))
                            except Exception:
                                setattr(a, 'sentiment_score', 0.0)
                            setattr(a, 'analysis_provider', 'hf')
                            logs.append(f"{asset} | HF | {(a.title or '')[:80]}")
                            continue
                    # Fallback to existing analyst (Gemini/Local)
                    self._last_provider = None
                    self.analyst.analyze_article(a, asset, model_preference=model_preference)
                    provider = (self._last_provider or "fallback").lower()
                    setattr(a, 'analysis_provider', provider)
                    logs.append(f"{asset} | {provider.upper()} | {(a.title or '')[:80]}")
                except Exception:
                    continue
            self.last_logs = logs
        out: List[NewsItem] = []
        for a in arts:
            out.append(NewsItem(
                title=a.title or "",
                url=a.url or "",
                description=getattr(a, 'description', '') or '',
                published_at=getattr(a, 'published_at', '') or '',
                source=getattr(a, 'source', '') or '',
                ai_summary=getattr(a, 'ai_summary', '') or '',
                sentiment_label=getattr(a, 'sentiment_label', '') or '',
                sentiment_score=float(getattr(a, 'sentiment_score', 0.0) or 0.0),
                analysis_provider=getattr(a, 'analysis_provider', '') or '',
            ))
        return out

    def get_logs(self) -> List[str]:
        return list(self.last_logs)

    def get_fetch_summary(self) -> str:
        try:
            return getattr(self, '_last_fetch_summary', '') or ''
        except Exception:
            return ''

    def summarize_overall(self, asset: str, items: List[NewsItem], *, model_preference: str = "auto", max_chars: int = 20000) -> Dict[str, str]:
        """Create a single overall AI summary from many items.
        Returns a dict with keys: summary, provider.
        """
        # Build a rich context from all items
        def _clip(txt: str, n: int) -> str:
            if not txt:
                return ""
            t = str(txt)
            return t if len(t) <= n else (t[: n - 3] + "...")

        # Helper to read from dicts or objects
        def gv(obj, key, default=""):
            try:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            except Exception:
                return default

        # Optionally enrich missing content for more breadth
        if self.fetcher:
            try:
                for a in items:
                    url = gv(a, 'url', '')
                    content = gv(a, 'content', '')
                    if url and not content:
                        content = self.fetcher.scrape_article_content(url)
                        if content:
                            try:
                                if isinstance(a, dict):
                                    a['content'] = content
                                else:
                                    setattr(a, 'content', content)
                            except Exception:
                                pass
            except Exception:
                pass

        lines: List[str] = []
        for i, it in enumerate(items, 1):
            src = gv(it, 'source', '') or ''
            dt = gv(it, 'published_at', '') or ''
            title = gv(it, 'title', '') or ''
            desc = gv(it, 'description', '') or ''
            per_summary = gv(it, 'ai_summary', '') or ''
            content = gv(it, 'content', '')
            sent = gv(it, 'sentiment_label', '') or ''
            sscore = gv(it, 'sentiment_score', None)
            sscore_s = (f" {float(sscore):+.2f}" if sscore is not None else "")
            lines.append(textwrap.dedent(f"""
            ### Item {i}
            Title: {title}
            Source/Date: {src} • {dt}
            Sentiment: {sent}{sscore_s}
            Summary: {_clip(per_summary, 2000)}
            Description: {_clip(desc, 1000)}
            Content: {_clip(content, 4000)}
            URL: {gv(it, 'url', '')}
            """))
        context = "\n".join(lines)
        # Hard cap to avoid local LLM crashes on very large prompts
        try:
            effective_max = min(int(max_chars or 20000), 12000)
        except Exception:
            effective_max = 12000
        if len(context) > effective_max:
            context = context[: effective_max] + "\n...[truncated]"

        system_msg = (
            "You are a seasoned financial markets analyst. Given multiple news items about a symbol, "
            "produce a concise executive summary. Reply as strict JSON only, no prose, using this schema: "
            "{\n"
            "  \"executive_summary\": string  # 200-300 words, 6-10 bullets or short paragraphs, no fluff, no chain-of-thought,\n"
            "  \"overall_sentiment\": string  # one of: bullish | bearish | neutral,\n"
            "  \"confidence\": number        # 0.0-1.0, confidence in sentiment,\n"
            "  \"key_points\": [string]      # 4-8 terse bullets, optional\n"
            "}\n"
            "Keep the summary tight and non-repetitive."
        )
        user_msg = f"Symbol: {asset}\n\nCorpus:\n{context}"
        result_obj = None
        summary_text = None
        provider = ""
        try:
            self.logger.info("news.summary: asset=%s model_pref=%s items=%s", asset, model_preference, len(items or []))
        except Exception:
            pass

        # Query preference
        def _try_gem():
            try:
                if self.analyst and hasattr(self.analyst, "_query_gemini"):
                    self._last_provider = None
                    res = self.analyst._query_gemini([
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ])
                    if res is not None:
                        return res
            except Exception:
                return None
            return None

        def _try_local():
            try:
                if self.analyst and hasattr(self.analyst, "_query_local_llm"):
                    self._last_provider = None
                    res = self.analyst._query_local_llm([
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg}
                    ])
                    if res is not None:
                        return res
            except Exception:
                return None
            return None

        def _try_local_openai():
            """Direct call to LOCAL_LLM_BASE_URL /chat/completions (OpenAI-compatible)."""
            try:
                base_url = (os.getenv("LOCAL_LLM_BASE_URL", "") or getattr(self, "local_llm_base_url", "")).strip().rstrip("/")
                model_name = (os.getenv("LOCAL_LLM_MODEL", "") or getattr(self, "local_llm_model", "")).strip() or "auto"
                api_key = (os.getenv("LOCAL_LLM_API_KEY", "") or getattr(self, "local_llm_api_key", "not-needed")).strip()
                if not base_url:
                    return None
                headers = {"Content-Type": "application/json"}
                if api_key and api_key != "not-needed":
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1") or 0.1),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096),
                    "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                if r.status_code != 200:
                    try:
                        self._get_app_logger().warning("news.summary.local: HTTP %s: %s", r.status_code, (r.text or "")[:200])
                    except Exception:
                        pass
                    return None
                data = r.json() or {}
                raw = (
                    data.get("choices", [{}])[0].get("message", {}).get("content")
                ) or data.get("choices", [{}])[0].get("text", "")
                return raw
            except Exception:
                return None

        def _try_do():
            try:
                base_url = (os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1") or "").strip().rstrip("/")
                api_key = (os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY") or "").strip()
                model_name = (os.getenv("DO_AI_MODEL") or "").strip() or "llama3-8b-instruct"
                if not (base_url and api_key):
                    return None
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1") or 0.1),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096),
                    "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                try:
                    self.logger.info("news.summary.DO: POST /chat/completions model=%s status=%s", model_name, r.status_code)
                except Exception:
                    pass
                if r.status_code != 200:
                    try:
                        self.logger.warning("news.summary.DO error: %s", (r.text or "")[:200])
                    except Exception:
                        pass
                    return None
                data = r.json()
                raw = (
                    data.get("choices", [{}])[0].get("message", {}).get("content")
                ) or data.get("choices", [{}])[0].get("text", "")
                return raw
            except Exception:
                return None

        def _try_hf():
            try:
                base_url = (os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1") or "").strip().rstrip("/")
                api_key = (os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN2") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
                model_name = (os.getenv("HF_MODEL") or "OpenAI/gpt-oss-20B").strip()
                if not (base_url and api_key):
                    return None
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1") or 0.1),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096),
                    "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                try:
                    self.logger.info("news.summary.HF: POST /chat/completions model=%s status=%s", model_name, r.status_code)
                except Exception:
                    pass
                if r.status_code != 200:
                    try:
                        self.logger.warning("news.summary.HF error: %s", (r.text or "")[:200])
                    except Exception:
                        pass
                    return None
                data = r.json()
                raw = (
                    data.get("choices", [{}])[0].get("message", {}).get("content")
                ) or data.get("choices", [{}])[0].get("text", "")
                return raw
            except Exception:
                return None

        if model_preference == "local":
            result_obj = _try_local() or _try_local_openai()
            provider = "local" if result_obj is not None else ""
        elif model_preference == "gemini":
            result_obj = _try_gem()
            provider = "gemini" if result_obj is not None else ""
        elif model_preference == "do":
            result_obj = _try_do()
            provider = "do" if result_obj is not None else ""
        elif model_preference == "hf":
            result_obj = _try_hf()
            provider = "hf" if result_obj is not None else ""
        else:  # auto
            result_obj = _try_gem()
            provider = "gemini" if result_obj is not None else provider
            if result_obj is None:
                result_obj = _try_local()
                provider = "local" if result_obj is not None else provider
            if result_obj is None:
                result_obj = _try_local_openai()
                provider = "local" if result_obj is not None else provider
            if result_obj is None:
                result_obj = _try_do()
                provider = "do" if result_obj is not None else provider
            if result_obj is None:
                result_obj = _try_hf()
                provider = "hf" if result_obj is not None else provider

        payload: Optional[Dict[str, Any]] = None
        # Normalize result into text and optional payload
        def _strip_code_fences(s: str) -> str:
            if not s:
                return s
            try:
                st = s.strip()
                # Remove leading ```json or ``` and trailing ```
                if st.startswith("```"):
                    # drop first fence line
                    st = st.split("\n", 1)[1] if "\n" in st else st.replace("```", "")
                if st.endswith("```"):
                    st = st[::-1].replace("```"[::-1], "", 1)[::-1]
                # Also remove a leading 'json' label if present
                if st.lower().startswith("json\n"):
                    st = st[5:]
                return st.strip()
            except Exception:
                return s
        if isinstance(result_obj, dict):
            payload = result_obj
        elif isinstance(result_obj, str):
            try:
                parsed = json.loads(_strip_code_fences(result_obj))
                if isinstance(parsed, dict):
                    payload = parsed
                else:
                    summary_text = result_obj
            except Exception:
                # Try to extract a JSON object if present within text
                try:
                    cleaned = _strip_code_fences(result_obj)
                    start = cleaned.find('{')
                    end = cleaned.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        maybe = cleaned[start:end+1]
                        parsed = json.loads(maybe)
                        if isinstance(parsed, dict):
                            payload = parsed
                        else:
                            summary_text = cleaned
                    else:
                        summary_text = cleaned
                except Exception:
                    summary_text = result_obj
        elif result_obj is not None:
            # Unknown type, stringify
            summary_text = str(result_obj)

        if payload and not summary_text:
            # Prefer executive_summary if available
            summary_text = payload.get('executive_summary') or payload.get('summary') or ''

        # If summary_text still looks like JSON, attempt a last parse
        if (summary_text or '').strip().startswith('{') and ('"executive_summary"' in summary_text or '"overall_sentiment"' in summary_text):
            try:
                pj = json.loads(_strip_code_fences(summary_text))
                if isinstance(pj, dict):
                    payload = payload or pj
                    summary_text = pj.get('executive_summary') or pj.get('summary') or summary_text
            except Exception:
                pass

        # Enforce concise length (trim to ~300 words)
        def _trim_words(txt: str, max_words: int = 300) -> str:
            try:
                words = (txt or '').split()
                if len(words) <= max_words:
                    return txt or ''
                return ' '.join(words[:max_words]) + ' …'
            except Exception:
                return (txt or '')

        if not summary_text:
            # Very simple fallback: synthesize from titles and per-article summaries
            bullets = []
            for it in items[:8]:
                title = (gv(it, 'title', '') or '')
                per = (gv(it, 'ai_summary', '') or gv(it, 'description', '') or '')
                bullets.append(f"- {title}: {_clip(per, 180)}")
            summary_text = "Overall summary (fallback):\n" + "\n".join(bullets)
            provider = provider or "fallback"

        summary_text = _trim_words(summary_text, 300)

        # Surface sentiment if provided, else derive from article sentiments
        def _normalize_sent_label(s: Optional[str]) -> Optional[str]:
            if not s:
                return None
            s = str(s).strip().lower()
            if s in ("bullish","bearish","neutral"):
                return s
            # common variants
            if s in ("pos","positive","up"): return "bullish"
            if s in ("neg","negative","down"): return "bearish"
            if s in ("flat","mixed"): return "neutral"
            return None

        # Try to read from payload with several key variants
        overall_sent = None
        conf = None
        if payload:
            for k in ("overall_sentiment","overallSentiment","sentiment","sentiment_label","label"):
                v = payload.get(k)
                lab = _normalize_sent_label(v)
                if lab:
                    overall_sent = lab
                    break
            for k in ("confidence","sentiment_confidence","score"):
                v = payload.get(k)
                try:
                    conf = float(v) if v is not None else None
                except Exception:
                    conf = None

        # Fallback aggregation across items if missing
        if not overall_sent:
            # Map labels to numeric for averaging
            def _val(lbl: Optional[str]) -> Optional[float]:
                lbl = (lbl or "").lower()
                if lbl == "bullish": return 1.0
                if lbl == "bearish": return -1.0
                if lbl == "neutral": return 0.0
                return None
            vs: List[float] = []
            ws: List[float] = []
            for it in items:
                try:
                    lbl = gv(it, 'sentiment_label', '')
                    v = _val(lbl)
                    if v is None:
                        continue
                    score = gv(it, 'sentiment_score', None)
                    try:
                        w = float(score) if score is not None else 1.0
                    except Exception:
                        w = 1.0
                    # clamp weight 0..1
                    if not (0.0 <= w <= 1.0):
                        w = 1.0
                    vs.append(v * w)
                    ws.append(w)
                except Exception:
                    continue
            if ws:
                try:
                    avg = sum(vs) / max(1e-9, sum(ws))
                    if avg > 0.2:
                        overall_sent = "bullish"
                    elif avg < -0.2:
                        overall_sent = "bearish"
                    else:
                        overall_sent = "neutral"
                    conf = max(0.0, min(1.0, abs(avg)))
                except Exception:
                    pass

        out = {"summary": summary_text, "provider": provider}
        if overall_sent:
            out["overall_sentiment"] = overall_sent
        if isinstance(conf, (int, float)):
            out["confidence"] = float(conf)
        if payload:
            out["payload"] = payload
        return out


# ---------------- Multi‑Agent Research (free-data) ----------------
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

    # Utils
    def _read_project_documents(self) -> Dict[str, str]:
        docs: Dict[str, str] = {}
        # project description (txt)
        try:
            txt = (self.base_dir / "dash" / "project description.txt")
            if txt.exists():
                docs["project_description"] = txt.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
        # requirements.pdf -> text
        try:
            pdfp = (self.base_dir / "dash" / "requirements.pdf")
            if pdfp.exists():
                global PdfReader
                if PdfReader is None:
                    try:
                        import importlib as _il
                        PdfReader = getattr(_il.import_module("pypdf"), "PdfReader", None)
                    except Exception:
                        PdfReader = None
                if PdfReader is None:
                    raise RuntimeError("pypdf not installed")
                text_parts = []
                reader = PdfReader(str(pdfp))
                for page in reader.pages:
                    try:
                        text_parts.append(page.extract_text() or "")
                    except Exception:
                        continue
                docs["requirements_pdf"] = "\n".join(text_parts)
        except Exception:
            pass
        return docs

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
            try:
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

                # --- Synthesis Agent (Indicator Calculation) ---
                fig, tech_summary = self.synthesis_agent_for_technicals(
                    hist,
                    indicators,
                    length,
                    rsi_length,
                    macd_fast,
                    macd_slow,
                    macd_signal_len,
                )

                # --- AI Analyst Agent ---
                ai_analysis = self._get_technical_ai_summary(ticker, tech_summary)
                try:
                    from telemetry import log_event
                    log_event("technical.figure", {
                        "ticker": ticker,
                        "interval": timeframe,
                        "indicators": indicators,
                        "has_figure": True,
                    })
                except Exception:
                    pass
                results[ticker] = {
                    "figure": fig,
                    "recommendation": ai_analysis.get("recommendation", "N/A"),
                    "confidence": ai_analysis.get("confidence", "N/A"),
                    "justification": ai_analysis.get(
                        "justification", "No justification provided."
                    ),
                }
            except Exception as e:
                try:
                    from telemetry import log_event
                    log_event("technical.error", {"ticker": ticker, "error": str(e)})
                except Exception:
                    pass
                results[ticker] = {"error": f"An error occurred: {str(e)}"}

        return results

    

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
        """Collect EPS and Revenue history where available and compute growth rates."""
        try:
            t_yf = _get_yf()
            t = t_yf.Ticker(symbol)
            out: Dict[str, Any] = {}
            # Quarterly earnings (yfinance provides 'Earnings' and 'Revenue')
            q_e = None
            try:
                q_e = t.quarterly_earnings.copy()
            except Exception:
                q_e = None
            if isinstance(q_e, pd.DataFrame) and not q_e.empty:
                out["quarterly_earnings"] = q_e.tail(8)  # keep last 8 quarters
                try:
                    rev = q_e["Revenue"].astype(float)
                    earn = q_e["Earnings"].astype(float)
                    rev_growth = (rev.pct_change() * 100.0).round(2).tolist()
                    earn_growth = (earn.pct_change() * 100.0).round(2).tolist()
                    out["revenue_growth_pct"] = rev_growth
                    out["earnings_growth_pct"] = earn_growth
                except Exception:
                    pass
            # Annual EPS from financials if available
            try:
                fin = t.financials
                eps_row = None
                for key in ["Basic EPS", "BasicEPS", "Diluted EPS", "DilutedEPS"]:
                    if key in (fin.index if isinstance(fin, pd.DataFrame) else []):
                        eps_row = fin.loc[key]
                        break
                if eps_row is not None:
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

    def _llm_generate(self, system_msg: str, user_msg: str) -> Optional[str]:
        """Generic LLM text generation using available providers (Gemini, Local, DO, HF). Returns text or None."""
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
        # Try Gemini via bridge
        try:
            if self.bridge and getattr(self.bridge, "analyst", None) and hasattr(self.bridge.analyst, "_query_gemini"):
                _log_llm("llm.request", {"provider": "gemini", "system_msg": system_msg, "user_msg": user_msg})
                out = self.bridge.analyst._query_gemini([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ])
                if isinstance(out, str) and out.strip():
                    _log_llm("llm.response", {"provider": "gemini", "response": out})
                    return out
        except Exception:
            _log_llm("llm.error", {"provider": "gemini", "error": "exception"})
            pass
        # Try Local via bridge
        try:
            if self.bridge and getattr(self.bridge, "analyst", None) and hasattr(self.bridge.analyst, "_query_local_llm"):
                _log_llm("llm.request", {"provider": "local", "system_msg": system_msg, "user_msg": user_msg})
                out = self.bridge.analyst._query_local_llm([
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ])
                if isinstance(out, str) and out.strip():
                    _log_llm("llm.response", {"provider": "local", "response": out})
                    return out
        except Exception:
            _log_llm("llm.error", {"provider": "local", "error": "exception"})
            pass
        # Try DO
        try:
            base_url = (os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1") or "").strip().rstrip("/")
            api_key = (os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY") or "").strip()
            model_name = (os.getenv("DO_AI_MODEL") or "").strip() or "llama3-8b-instruct"
            if base_url and api_key:
                _log_llm("llm.request", {"provider": "do", "model": model_name, "system_msg": system_msg, "user_msg": user_msg})
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2") or 0.2),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096),
                    "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    raw = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                    ) or data.get("choices", [{}])[0].get("text", "")
                    if raw:
                        _log_llm("llm.response", {"provider": "do", "response": raw, "usage": data.get("usage")})
                        return raw
        except Exception as e:
            _log_llm("llm.error", {"provider": "do", "error": str(e)})
            pass
        # Try HF
        try:
            base_url = (os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1") or "").strip().rstrip("/")
            api_key = (os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN2") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
            model_name = (os.getenv("HF_MODEL") or "OpenAI/gpt-oss-20B").strip()
            if base_url and api_key:
                _log_llm("llm.request", {"provider": "hf", "model": model_name, "system_msg": system_msg, "user_msg": user_msg})
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.2") or 0.2),
                    "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096") or 4096),
                    "stream": False,
                }
                r = _requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    raw = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                    ) or data.get("choices", [{}])[0].get("text", "")
                    if raw:
                        _log_llm("llm.response", {"provider": "hf", "response": raw, "usage": data.get("usage")})
                        return raw
        except Exception as e:
            _log_llm("llm.error", {"provider": "hf", "error": str(e)})
            pass
        return None

    def holistic_analysis_agent(
        self,
        symbol: str,
        notebook: Dict[str, Any],
    ) -> Tuple[str, str]:
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

        llm_text = self._llm_generate(system_msg, user_msg)
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

    def recommendation_agent(self, symbol: str, notebook: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
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

        llm_text = self._llm_generate(system_msg, user_msg)
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

    def run_full_analysis(
        self,
        *,
        symbol: str,
        analysis_type: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str = "1d",
        indicators: Optional[List[str]] = None,
        length: int = 20,
        rsi_length: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal_len: int = 9,
    ) -> Dict[str, Any]:
        """Enhanced orchestrator covering technical, fundamental, combined, and web-search modes.
        Returns a dict with keys: plan, profile, fundamentals, ratios, earnings, technical, news, holistic, recommendation, figures
        """
        sym = (symbol or "").strip().upper()
        a_type = (analysis_type or "combined").lower()
        indicators = indicators or ["SMA", "EMA", "Bollinger Bands", "VWAP", "RSI", "MACD", "ATR"]

        plan_steps = []
        if a_type == "technical":
            plan_steps = ["Technical Data Agent", "Technical Analysis Agent", "Recommendation Agent", "Visualization Agent"]
        elif a_type == "fundamental":
            plan_steps = ["Company Profile Agent", "Financial Data Agent", "Fundamental Ratios Agent", "Earnings Analysis Agent", "Holistic Analysis Agent", "Recommendation Agent", "Visualization Agent"]
        elif a_type == "web":
            plan_steps = ["General Web Search Agent", "News & Sentiment Agent", "Holistic Analysis Agent"]
        else:
            plan_steps = [
                "UI Agent",
                "Planner",
                "Company Profile Agent",
                "Financial Data Agent",
                "Fundamental Ratios Agent",
                "Earnings Analysis Agent",
                "Technical Data Agent",
                "Technical Analysis Agent",
                "News & Sentiment Agent",
                "Holistic Analysis Agent",
                "Recommendation Agent",
                "Visualization Agent",
            ]

        outputs: Dict[str, Any] = {"plan": {"analysis_type": a_type, "steps": plan_steps}}
        # Initialize Analyst's Notebook (shared context)
        notebook: Dict[str, Any] = {
            "symbol": sym,
            "profile": {},
            "financials": {"ratios": {}, "earnings": {}},
            "technical": {},
            "news_sentiment": {},
            "web": [],
            "errors": [],
        }
        conversations: List[Dict[str, str]] = []
        # UI Agent structured request (logged)
        try:
            conversations.append({
                "agent": "UI Agent",
                "message": json.dumps({
                    "symbol": sym,
                    "analysis_type": a_type,
                    "interval": interval,
                    "start_date": start_date,
                    "end_date": end_date,
                    "indicators": indicators,
                    "params": {"length": length, "rsi": rsi_length, "macd": [macd_fast, macd_slow, macd_signal_len]},
                })
            })
            conversations.append({
                "agent": "Planner",
                "message": " -> ".join(plan_steps)
            })
        except Exception:
            pass

        # Optional News & Web
        news_summary = None
        try:
            if a_type in ("web", "combined"):
                news_items = self.bridge.fetch(sym, days_back=7, max_articles=6, model_preference="auto", analyze=True)
                outputs["news"] = [i.__dict__ for i in news_items]
                conversations.append({"agent": "News & Sentiment Agent", "message": f"Analyzed {len(news_items)} articles."})
                # Prefer local model if available
                local_llm_base = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
                model_pref = "local" if local_llm_base else "auto"
                news_summary = self.bridge.summarize_overall(sym, news_items, model_preference=model_pref, max_chars=2500)
                outputs["news_summary"] = news_summary
                if isinstance(news_summary, dict) and news_summary.get("summary"):
                    conversations.append({"agent": "News & Sentiment Agent", "message": news_summary.get("summary", "")[:500] + ("..." if len(news_summary.get("summary",""))>500 else "")})
                    notebook["news_sentiment"] = {"summary": news_summary.get("summary"), "provider": news_summary.get("provider")}
        except Exception:
            outputs["news"] = []
        # General web search (qualitative context)
        try:
            if a_type in ("web", "combined"):
                q = f"{sym} company analysis fundamentals news"
                _m, web_res = self.web_search_agent(q, max_results=20)
                outputs["web_search"] = web_res
                notebook["web"] = web_res
                conversations.append({"agent": "Web Search Agent", "message": f"Query: {q} -> {len(web_res)} results"})
                if len(web_res) == 0:
                    notebook["errors"].append("CRITICAL: General web search returned 0 results; check search tool/network.")
        except Exception:
            outputs["web_search"] = []

        # Company profile and fundamentals
        profile = {}
        fundamentals = {}
        ratios = {}
        earnings = {}
        try:
            if a_type in ("fundamental", "combined"):
                _m, profile = self.company_profile_agent(sym)
                outputs["company_profile"] = profile
                notebook["profile"] = profile
                conversations.append({"agent": "Company Profile Agent", "message": f"Loaded profile for {profile.get('name', sym)}"})
                _m, fundamentals = self.financial_data_agent_full(sym)
                outputs["fundamentals"] = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else None) for k, v in fundamentals.items()}
                conversations.append({"agent": "Financial Data Agent", "message": "Loaded financial statements (annual/quarterly)."})
        except Exception:
            pass

        # Technicals
        technical = {}
        tech_fig = None
        price_df = None
        tech_summary = ""
        try:
            if a_type in ("technical", "combined"):
                # Fetch with explicit dates if given, else 1y
                if start_date and end_date:
                    _m, payload = self.data_agent_with_dates(sym, start_date, end_date, interval)
                else:
                    _m, payload = self.data_agent(sym, period="1y", interval=interval)
                price_df = payload.get("hist")
                tech_fig, tech_summary = self.synthesis_agent_for_technicals(
                    price_df,
                    indicators,
                    length,
                    rsi_length,
                    macd_fast,
                    macd_slow,
                    macd_signal_len,
                )
                ai = self._get_technical_ai_summary(sym, tech_summary) or {}
                technical = {"summary": tech_summary, **ai}
                outputs["technical"] = technical
                notebook["technical"] = technical
                conversations.append({"agent": "Technical Data Agent", "message": f"Fetched {0 if price_df is None else len(price_df)} rows; interval={interval}"})
                conversations.append({"agent": "Technical Analysis Agent", "message": (ai.get('justification') or tech_summary or '')[:600] + ('...' if len((ai.get('justification') or tech_summary or ''))>600 else '')})
                try:
                    from telemetry import log_event
                    log_event("technical.figure_full", {
                        "symbol": sym,
                        "interval": interval,
                        "rows": (0 if price_df is None else int(getattr(price_df, 'shape', [0])[0])),
                        "has_figure": bool(tech_fig is not None),
                    })
                except Exception:
                    pass
        except Exception:
            outputs["technical"] = {"error": "Technical analysis failed."}

        # Ratios/Earnings
        try:
            if a_type in ("fundamental", "combined"):
                _m, ratios = self.fundamental_ratios_agent(sym, fundamentals, price_df)
                conversations.append({"agent": "Fundamental Ratios Agent", "message": json.dumps({k: ratios.get(k) for k in ['pe_ttm','ps_ttm','debt_to_equity','gross_margin','net_margin']}, default=str)})
                _m, earnings = self.earnings_agent(sym)
                outputs["ratios"] = ratios
                outputs["earnings"] = {k: (v.to_dict() if isinstance(v, pd.DataFrame) else v) for k, v in earnings.items()}
                notebook["financials"]["ratios"] = ratios
                notebook["financials"]["earnings"] = outputs["earnings"]
                conversations.append({"agent": "Earnings Analysis Agent", "message": "Collected earnings and growth metrics."})
                # Validate critical metrics; if missing, try targeted web queries to recover
                def _extract_first_number(snippet: str) -> Optional[float]:
                    if not snippet:
                        return None
                    m = re.search(r"(\d{1,3}(?:\.\d+)?)(?:\s*[xX])?", snippet)
                    try:
                        return float(m.group(1)) if m else None
                    except Exception:
                        return None
                if ratios.get("pe_ttm") in (None, 0):
                    queries = [
                        f"{sym} current P/E ratio",
                        f"{profile.get('name', sym)} P/E ratio",
                        f"{sym} trailing P/E",
                    ]
                    for q in queries:
                        _wm, res = self.web_search_agent(q, max_results=5)
                        for r in res:
                            cand = _extract_first_number((r.get("snippet") or "") + " " + (r.get("title") or ""))
                            if cand and cand > 1 and cand < 200:
                                ratios["pe_ttm"] = cand
                                ratios.setdefault("notes", "")
                                ratios["notes"] = (ratios["notes"] + "; " if ratios["notes"] else "") + f"PE recovered from web: {r.get('url','')}"
                                break
                        if ratios.get("pe_ttm") not in (None, 0):
                            break
                    if ratios.get("pe_ttm") in (None, 0):
                        notebook["errors"].append("CRITICAL: Could not retrieve P/E (TTM).")
                if ratios.get("ps_ttm") in (None, 0):
                    queries = [
                        f"{sym} price to sales ratio",
                        f"{profile.get('name', sym)} P/S ratio",
                    ]
                    for q in queries:
                        _wm, res = self.web_search_agent(q, max_results=5)
                        for r in res:
                            cand = _extract_first_number((r.get("snippet") or "") + " " + (r.get("title") or ""))
                            if cand and cand > 0 and cand < 100:
                                ratios["ps_ttm"] = cand
                                ratios.setdefault("notes", "")
                                ratios["notes"] = (ratios["notes"] + "; " if ratios["notes"] else "") + f"PS recovered from web: {r.get('url','')}"
                                break
                        if ratios.get("ps_ttm") not in (None, 0):
                            break
                    if ratios.get("ps_ttm") in (None, 0):
                        notebook["errors"].append("CRITICAL: Could not retrieve P/S (TTM).")
        except Exception:
            pass

        # Holistic + Recommendation
        try:
            if a_type in ("fundamental", "combined", "web"):
                _m, hol = self.holistic_analysis_agent(sym, notebook)
                outputs["holistic"] = hol
                conversations.append({"agent": "Holistic Analysis Agent", "message": (hol or '')[:600] + ('...' if hol and len(hol)>600 else '')})
        except Exception:
            outputs["holistic"] = ""
        try:
            if a_type in ("technical", "combined", "fundamental"):
                _m, rec = self.recommendation_agent(sym, notebook)
                outputs["recommendation"] = rec
                conversations.append({"agent": "Recommendation Agent", "message": json.dumps(rec)})
        except Exception:
            outputs["recommendation"] = {"recommendation": "Hold", "confidence": 5, "justification": "n/a"}

        # Visualizations
        figs: Dict[str, Any] = {}
        try:
            if tech_fig is not None:
                figs["technical"] = tech_fig
        except Exception:
            pass
        try:
            q_fin = fundamentals.get("q_financials") if fundamentals else None
            rev_fig = self._figure_revenue_growth(sym, q_fin)
            if rev_fig is not None:
                figs["revenue"] = rev_fig
        except Exception:
            pass
        try:
            pe_fig = self._figure_pe_trend(sym, price_df, (ratios or {}).get("eps_ttm"))
            if pe_fig is not None:
                figs["pe_trend"] = pe_fig
        except Exception:
            pass
        outputs["figures"] = figs
        outputs["conversations"] = conversations
        outputs["notebook"] = notebook
        try:
            from telemetry import log_event
            log_event("analysis.completed", {
                "symbol": sym,
                "type": a_type,
                "steps": plan_steps,
                "news_count": len(outputs.get("news", [])) if isinstance(outputs.get("news"), list) else 0,
                "web_results": len(outputs.get("web_search", [])) if isinstance(outputs.get("web_search"), list) else 0,
                "has_tech": bool(outputs.get("technical")),
                "figures": list((outputs.get("figures") or {}).keys()),
                "errors": notebook.get("errors", []),
            })
        except Exception:
            pass
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


class TestmailClient:
    """Minimal Testmail client. Docs: https://testmail.app/api

    Environment:
    - TESTMAIL_API_KEY: your API key
    - TESTMAIL_NAMESPACE: optional namespace (defaults to 'dashdemo')
    """

    BASE = "https://api.testmail.app/api/json"

    def __init__(self, api_key: Optional[str] = None, namespace: Optional[str] = None):
        self.api_key = (api_key or os.getenv("TESTMAIL_API_KEY", "")).strip()
        self.namespace = (namespace or os.getenv("TESTMAIL_NAMESPACE", "dashdemo")).strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate_address(self, tag: str = "inbox") -> Dict[str, Any]:
        if not self.is_configured():
            return {"error": "Missing TESTMAIL_API_KEY"}
        # Testmail uses pattern: <namespace>.<tag>@inbox.testmail.app
        addr = f"{self.namespace}.{tag}@inbox.testmail.app"
        return {"address": addr}

    def get_messages(self, tag: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        if not self.is_configured():
            return {"error": "Missing TESTMAIL_API_KEY"}
        params = {
            "apikey": self.api_key,
            "namespace": self.namespace,
            "limit": int(limit or 10),
        }
        if tag:
            params["tag"] = tag
        try:
            r = _requests.get(self.BASE, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()  # includes 'emails': [...]
            else:
                try:
                    j = r.json()
                    return {"error": j}
                except Exception:
                    return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            return {"error": str(e)}

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
