import os
import re
import json
import requests
import logging
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv() # Ensures environment variables are loaded at the very start
import dash
from dash import Dash, dcc, html, Input, Output, State, ALL
# Import services with package-safe relative import first, then fallback for script execution
try:
    from .services import NewsBridge, ResearchOrchestrator, strip_markdown_fences
except Exception:
    from services import NewsBridge, ResearchOrchestrator, strip_markdown_fences
import plotly.graph_objects as go
# Import tab layouts; handle both package and script execution paths
try:
    from .ui.layouts.news import layout as news_tab_layout
    from .ui.layouts.strategy import layout as strategy_tab_layout
except Exception:
    # When running as a script (python dash/app.py), use local import
    from ui.layouts.news import layout as news_tab_layout
    from ui.layouts.strategy import layout as strategy_tab_layout

from pathlib import Path as _Path
_BASE = _Path(__file__).resolve().parent
_ASSETS_PATH = str(_BASE / "ui" / "assets")

# Configurable research history path (supports Render Disks or other mounts)
def _history_file() -> _Path:
    """Resolve path to research history JSON file.
    Priority:
    1) RESEARCH_HISTORY_PATH (full file path)
    2) RESEARCH_STORAGE_DIR (directory to place research_history.json)
    3) default: <project>/logs/research_history.json
    """
    p = os.getenv("RESEARCH_HISTORY_PATH")
    if p:
        return _Path(p)
    d = os.getenv("RESEARCH_STORAGE_DIR")
    if d:
        return _Path(d) / "research_history.json"
    return _BASE / "logs" / "research_history.json"

# Simple disk cache for News tab (reuses recent fetched cards)
def _news_cache_file() -> _Path:
    d = _BASE / "logs"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d / "news_cache.json"

def _news_cache_load() -> dict:
    try:
        p = _news_cache_file()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _news_cache_save(cache: dict) -> None:
    try:
        p = _news_cache_file()
        p.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass

def _news_cache_get(key: str, ttl_minutes: int) -> tuple[list[dict], dict, dict] | None:
    try:
        # Allow disabling cache via env for quick recovery/debugging
        if str(os.getenv("NEWS_CACHE_DISABLE", "")).strip().lower() in ("1", "true", "yes", "on"):
            return None
        cache = _news_cache_load()
        entry = (cache or {}).get(key)
        if not entry:
            return None
        ts = entry.get("meta", {}).get("ts")
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts.replace('Z','+00:00'))
        except Exception:
            return None
        age = datetime.now(timezone.utc) - (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc))
        if age.total_seconds() > max(1, ttl_minutes) * 60:
            return None
        items = entry.get("items") or []
        overall = entry.get("overall") or {}
        meta = entry.get("meta") or {}
        # basic shape checks
        if not isinstance(items, list) or not isinstance(overall, dict) or not isinstance(meta, dict):
            return None
        return items, overall, meta
    except Exception:
        return None

def _news_cache_put(key: str, items: list[dict], overall: dict, meta: dict) -> None:
    try:
        # Allow disabling cache via env for quick recovery/debugging
        if str(os.getenv("NEWS_CACHE_DISABLE", "")).strip().lower() in ("1", "true", "yes", "on"):
            return
        # Skip caching obviously bad results to avoid freezing the UI in an error state
        try:
            provider = (overall or {}).get("provider", "")
            summary_txt = (overall or {}).get("summary", "")
            missing_ai = sum(1 for it in (items or []) if not (it or {}).get("ai_summary"))
            total = len(items or [])
            # Consider result low-quality if overall failed or majority of items lack ai_summary
            low_quality = (provider == "error") or (not summary_txt) or (total > 0 and missing_ai >= max(1, total // 2))
        except Exception:
            low_quality = False

        if low_quality:
            # Don't cache low-quality results; let next call try again soon
            return

        cache = _news_cache_load()
        cache[key] = {"items": items, "overall": overall, "meta": meta}
        # Optional: cap cache size
        if len(cache) > 100:
            # Drop oldest by ts
            try:
                cache = dict(sorted(cache.items(), key=lambda kv: (kv[1].get("meta",{}).get("ts") or ""), reverse=True)[:80])
            except Exception:
                pass
        _news_cache_save(cache)
    except Exception:
        pass

# Instantiate Dash app early (before callbacks)
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Financial Intelligence Dashboard",
    assets_folder=_ASSETS_PATH,
)
server = app.server

# Optionally silence noisy access logs like "POST /_dash-update-component 204".
try:
    _silence = str(os.getenv("SILENCE_ACCESS_LOGS", "1")).strip().lower() in ("1", "true", "yes", "on")
    if _silence:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        # Also quiet Dash/Flask app loggers
        try:
            app.logger.setLevel(logging.ERROR)
            server.logger.setLevel(logging.ERROR)
        except Exception:
            pass
except Exception:
    pass

# Explicit set of crypto tickers for TradingView mapping
CRYPTO_ASSETS = {"BTC", "ETH", "SOL"}

# Low-end model filter keywords (DRY across callbacks)
LOW_END_MODEL_KEYWORDS = [
    "7b", "8b", "9b", "10b", "12b", "13b", "15b", "20b", "mini", "tiny", "small"
]

# Curated HF models shown in dropdowns
CURATED_HF_MODELS = [
    {"label": "Qwen/Qwen3-4B-Base (default)", "value": "Qwen/Qwen3-4B-Base"},
]

# --- Symbol autocomplete support ---
# Default popular symbols shown when there is no search input
DEFAULT_SYMBOL_OPTIONS = [
    {"label": x, "value": x}
    for x in [
        # Mega-cap tech
        "AAPL", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "NVDA", "AMD", "NFLX", "INTC",
        # Popular ETFs and commodities
        "VOO", "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV",
        # Top crypto pairs
        "BTC-USD", "ETH-USD",
        # Semis and TSMC/ASML frequently searched
        "TSM", "ASML"
    ]
]

# Research-specific defaults (symbol only, no names needed)
RESEARCH_SYMBOL_OPTIONS = DEFAULT_SYMBOL_OPTIONS

_SYMBOL_INDEX_CACHE: list[dict] | None = None

def _symbols_index_path() -> _Path:
    return _BASE / "dev" / "docs" / "symbols.json"

def _load_symbol_index() -> list[dict]:
    """Load a local symbol index JSON if present.

    Expected format: list of {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"}
    Falls back to an empty list if not found.
    """
    global _SYMBOL_INDEX_CACHE
    if _SYMBOL_INDEX_CACHE is not None:
        return _SYMBOL_INDEX_CACHE
    try:
        p = _symbols_index_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                _SYMBOL_INDEX_CACHE = data
                return _SYMBOL_INDEX_CACHE
    except Exception:
        pass
    _SYMBOL_INDEX_CACHE = []
    return _SYMBOL_INDEX_CACHE

def _search_symbols(query: str, limit: int = 20) -> list[dict]:
    """Return dropdown options matching the query from the local index.

    Matches symbol prefix aggressively; also matches by name substring.
    """
    q = (query or "").strip()
    if not q:
        return DEFAULT_SYMBOL_OPTIONS
    q_upper = q.upper()
    q_lower = q.lower()
    idx = _load_symbol_index() or []
    # If no index, still offer default popular list filtered by prefix
    if not idx:
        base = [opt for opt in DEFAULT_SYMBOL_OPTIONS if opt["value"].startswith(q_upper)]
        return base[: max(5, min(limit, 20))]
    out: list[dict] = []
    for row in idx:
        sym = (row.get("symbol") or "").upper()
        name = (row.get("name") or "").strip()
        exch = (row.get("exchange") or "").strip()
        # Prioritize symbol prefix match, then name contains
        if sym.startswith(q_upper) or (name and q_lower in name.lower()):
            label_parts = [sym]
            if name:
                label_parts.append(f"— {name}")
            if exch:
                label_parts.append(f"({exch})")
            label = " ".join(label_parts)
            out.append({"label": label, "value": sym})
            if len(out) >= limit:
                break
    # Fallback: if no results, show default filtered by prefix
    if not out:
        base = [opt for opt in DEFAULT_SYMBOL_OPTIONS if opt["value"].startswith(q_upper)]
        return base[: max(5, min(limit, 20))]
    return out

# Service singletons
bridge = NewsBridge()
research = ResearchOrchestrator(news_bridge=bridge)

# Defaults for Local LLM fields pulled from environment
DEFAULT_LLM_BASE = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
DEFAULT_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL") or os.getenv("MODEL_NAME") or "llama-3.1-8b-instruct"
try:
    DEFAULT_LLM_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096"))
except Exception:
    DEFAULT_LLM_TOKENS = 4096
try:
    DEFAULT_LLM_TEMP = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1"))
except Exception:
    DEFAULT_LLM_TEMP = 0.1

def tv_symbol(asset: str) -> str:
    a = (asset or '').strip().upper()
    if not a:
        return "AAPL"
    if ":" in a:
        return a
    if a in CRYPTO_ASSETS:
        return f"CRYPTO:{a}USD"
    if a.isalpha() and 1 <= len(a) <= 5:
        return a
    for suf in ("USDT", "USD"):
        if a.endswith(suf) and a[:-len(suf)].isalpha() and 1 <= len(a[:-len(suf)]) <= 5:
            return a[:-len(suf)]
    return a

def _get_do_model_options(base_url: str) -> list[dict]:
    """Fetch DO models and return filtered low-end options; returns empty on error.

    Uses DO_AI_API_KEY or DIGITALOCEAN_AI_API_KEY from env.
    """
    try:
        base = (base_url or "https://inference.do-ai.run/v1").rstrip("/")
        key = os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY")
        if not key:
            return []
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        r = requests.get(f"{base}/models", headers=headers, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json() or {}
        opts = []
        for m in (data.get("data") or [])[:200]:
            mid = (m.get("id") or m.get("name") or "").lower()
            if mid and any(k in mid for k in LOW_END_MODEL_KEYWORDS):
                lab = m.get("id") or m.get("name")
                opts.append({"label": lab, "value": lab})
        return opts
    except Exception:
        return []

# Layout with enhanced visual design
HISTORY_LIMIT = 100  # maximum number of research history entries retained

def _load_initial_history():
    try:
        import json as _json
        p = _history_file()
        if p.exists():
            return _json.loads(p.read_text(encoding='utf-8'))[:HISTORY_LIMIT]
    except Exception:
        pass
    return []

_INITIAL_HISTORY = _load_initial_history()

def serve_layout():
    return html.Div(
        className="container",
        children=[
            # Enhanced tabs with icons
            dcc.Tabs(
                id="tabs",
                value="tab-news",
                className="tab-parent",
                children=[
                    dcc.Tab(label="📊 News & Analysis", value="tab-news", className="tab"),
                    dcc.Tab(label="💼 Strategy (Wallet)", value="tab-strategy", className="tab"),
                    dcc.Tab(label="🔬 Research", value="tab-research", className="tab"),
                    dcc.Tab(label="🕒 History", value="tab-history", className="tab"),
                ],
            ),

            # Data stores
            dcc.Store(id="news-data"),
            dcc.Store(id="active-asset"),
            dcc.Store(id="selected-idx"),
            dcc.Store(id="overall-summary"),
            dcc.Store(id="update-meta"),
            # Research history (persist recent research runs; max HISTORY_LIMIT entries)
            dcc.Store(id="research-history", data=_INITIAL_HISTORY),
            # Persist the last selected tab across reloads (e.g., dev hot reload)
            dcc.Store(id="tabs-memory", storage_type="local"),
            # Selected history item (expanded)
            dcc.Store(id="history-selected"),
            # One-shot interval to pre-list DigitalOcean models on first load
            dcc.Interval(id="do-models-primer", interval=300, n_intervals=0, max_intervals=1),
            # One-shot primer to restore the last selected tab
            dcc.Interval(id="tabs-primer", interval=300, n_intervals=0, max_intervals=1),

            # Tab content container (keep tab children mounted and toggle visibility via style)
            html.Div(
                id="tab-content",
                children=[
                    html.Div(
                        id="tab-news-container",
                        children=news_tab_layout(),
                        style={"display": "block"},
                    ),
                    html.Div(
                        id="tab-strategy-container",
                        children=strategy_tab_layout(),
                        style={"display": "none"},
                    ),
                    html.Div(
                        id="tab-research-container",
                        children=research_layout,
                        style={"display": "none"},
                    ),
                    html.Div(
                        id="tab-history-container",
                        children=history_tab_layout(),
                        style={"display": "none"},
                    ),
                ],
            ),
        ],
    )

# Tab 1 and 2 layouts are imported from layouts/ modules

# Tab 3: Research (multi-agent) layout (redesigned for specified agents)
research_layout = html.Div(
    style={"display": "grid", "gridTemplateColumns": "400px 1fr", "gap": "16px"},
    children=[
        # --- WIZARD-STYLE SIDEBAR ---
        html.Div(
            className="card",
            style={"padding": "16px"},
            children=[
                html.H3("Research Configuration", className="section-title"),
                
                dcc.Tabs(id="config-wizard-tabs", value='tab-mission', className="wizard-tabs", children=[
                    
                    # --- STEP 1: DEFINE THE MISSION ---
                    dcc.Tab(label='1. Mission ✨', value='tab-mission', className="wizard-tab", children=[
                        html.Div(className="wizard-step", children=[
                            html.Label("Ticker Symbol"),
                            html.Div([
                                dcc.Dropdown(
                                    id="research-symbol-dd",
                                    options=RESEARCH_SYMBOL_OPTIONS,
                                    value="AAPL",
                                    clearable=False,
                                    searchable=True,
                                    style={"width": "60%", "display": "inline-block", "marginRight": "8px"}
                                ),
                                dcc.Input(
                                    id="research-symbol-input",
                                    type="text",
                                    placeholder="Or custom...",
                                    style={"width": "38%", "display": "inline-block"}
                                ),
                            ], style={"display": "flex", "flexDirection": "row", "gap": "8px"}),

                            html.Label("Analysis Type"),
                            dcc.Dropdown(
                                id="analysis-type",
                                options=[
                                    {"label": "Combined (Fundamental + Technical)", "value": "combined"},
                                    {"label": "Technical Analysis Only", "value": "technical"},
                                    {"label": "Fundamental Analysis Only", "value": "fundamental"},
                                    {"label": "Web & News Summary", "value": "web"},
                                ],
                                value="combined",
                                clearable=False,
                            ),
                            
                            html.Label("Timeframe"),
                            dcc.Dropdown(
                                id="research-timeframe",
                                options=[
                                    {"label": "Daily", "value": "1d"},
                                    {"label": "Hourly", "value": "1h"},
                                    {"label": "15 Minute", "value": "15m"},
                                ],
                                value="1d",
                                clearable=False,
                            ),

                            html.Label("Lookback Period"),
                            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"8px"}, children=[
                                dcc.Input(id="lookback-value", type="number", value=180, min=1, step=1),
                                dcc.Dropdown(
                                    id="lookback-unit",
                                    options=[
                                        {"label": "Days", "value": "days"},
                                        {"label": "Weeks", "value": "weeks"},
                                        {"label": "Months", "value": "months"},
                                    ],
                                    value="days",
                                    clearable=False,
                                ),
                            ])
                        ])
                    ]),
                    
                    # --- STEP 2: CONFIGURE THE AI ENGINE ---
                    dcc.Tab(label='2. AI Engine', value='tab-engine', className="wizard-tab", children=[
                        html.Div(className="wizard-step", children=[
                            html.Label("Execution Engine"),
                            dcc.RadioItems(
                                id="execution-engine",
                                options=[
                                    {'label': ' LangChain (Fast & Focused)', 'value': 'lang'},
                                    #{'label': ' AutoGen (Deep & Collaborative)', 'value': 'multi'}, # Disabled for now 
                                ],
                                value='lang',
                                className="radio-items"
                            ),
                            
                            html.Hr(),
                            
                            html.Label("Allow Web Search?"),
                            dcc.RadioItems(
                                id="allow-web-search",
                                options=[{'label': ' On', 'value': 'on'}, {'label': ' Off', 'value': 'off'}],
                                value='on',
                                labelStyle={'display': 'inline-block', 'marginRight': '20px'}
                            ),

                            html.Div(id='web-results-container', children=[
                                html.Label("Max Web Results"),
                                dcc.Dropdown(
                                    id="web-max-results",
                                    options=[
                                        {"label": "3 (Fast)", "value": 3},
                                        {"label": "5 (Balanced)", "value": 5},
                                        {"label": "10 (Thorough)", "value": 10},
                                        {"label": "15 (In-depth)", "value": 15},
                                        {"label": "20 (Exhaustive)", "value": 20},
                                    ],
                                    value=5,
                                    clearable=False,
                                ),
                            ]),
                            
                            html.Hr(),
                            
                            html.Label("Speed vs. Thoroughness"),
                            dcc.Dropdown(
                                id="speed-mode",
                                options=[
                                    {"label": "Fast", "value": "fast"},
                                    {"label": "Normal", "value": "normal"},
                                    {"label": "Thorough", "value": "thorough"},
                                ],
                                value="fast",
                                clearable=False,
                            ),
                        ])
                    ]),

                    # --- STEP 3: ADVANCED SETTINGS (OPTIONAL) ---
                    dcc.Tab(label='3. Advanced', value='tab-advanced', className="wizard-tab", children=[
                        html.Div(className="wizard-step", children=[
                            html.Details([
                                html.Summary("Technical Indicator Parameters"),
                                html.Div(className="details-content", children=[
                                    html.Label("Technical Indicators"),
                                    dcc.Dropdown(
                                        id="technical-indicators",
                                        options=[{"label": val, "value": val} for val in ["SMA", "EMA", "Bollinger Bands", "VWAP", "RSI", "MACD", "ATR"]],
                                        value=["SMA", "EMA", "Bollinger Bands", "VWAP", "RSI", "MACD", "ATR"],
                                        multi=True,
                                    ),
                                    html.Label("Length (SMA/EMA/Bollinger)"),
                                    dcc.Input(id="indicator-length", type="number", value=20, style={"width": "100%"}),
                                    html.Label("RSI Length"),
                                    dcc.Input(id="rsi-length", type="number", value=14, style={"width": "100%"}),
                                    html.Label("MACD Fast/Slow/Signal"),
                                    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"8px"}, children=[
                                        dcc.Input(id="macd-fast", type="number", value=12),
                                        dcc.Input(id="macd-slow", type="number", value=26),
                                        dcc.Input(id="macd-signal", type="number", value=9),
                                    ])
                                ])
                            ]),
                            
                            html.Details([
                                html.Summary("AI Agent Parameters"),
                                html.Div(className="details-content", children=[
                                    html.Label("Agent Rounds"),
                                    dcc.Input(id="agent-rounds", type="number", value=2, min=1, max=6, step=1, style={"width": "100%"}),
                                ])
                            ]),

                            html.Details([
                                html.Summary("LLM Provider Settings"),
                                html.Div(className="details-content", style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "10px"}, children=[
                                    html.Div([
                                        html.Label("AI Provider"),
                                        dcc.Dropdown(
                                            id="ai-provider",
                                            options=[
                                                {"label": "Auto", "value": "auto"},
                                                {"label": "Local (LM STUDIO)", "value": "local"},
                                                {"label": "Gemini", "value": "gemini"},
                                                {"label": "DigitalOcean Inference", "value": "do"},
                                                {"label": "Hugging Face (Router)", "value": "hf"},
                                            ],
                                            value=os.getenv("AI_PROVIDER", "local"),
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Local Base URL"),
                                        dcc.Input(id="llm-base-url", type="text", value=DEFAULT_LLM_BASE, placeholder="http://localhost:11434/v1", style={"width": "100%"}),
                                        html.Button("Test Local LLM", id="llm-test", className="btn", style={"marginTop": "6px"}),
                                        html.Div(id="llm-test-status", style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "6px"}),
                                    ]),
                                    html.Div([
                                        html.Label("Local Model"),
                                        dcc.Input(id="llm-model", type="text", value=DEFAULT_LLM_MODEL, placeholder="qwen/qwen3-4b-2507", style={"width": "100%"}),
                                        dcc.Dropdown(id="llm-model-dd", options=[], placeholder="Pick Local model (optional)", style={"marginTop": "8px", "width": "100%"}),
                                        html.Div(id="llm-model-status", style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "6px"}),
                                    ]),
                                    dcc.Interval(id="llm-models-primer", n_intervals=0, max_intervals=1, interval=1000),
                                    html.Div([
                                        html.Label("Max tokens"),
                                        dcc.Input(id="llm-max-tokens", type="number", value=DEFAULT_LLM_TOKENS, placeholder="512", style={"width": "100%"}),
                                        html.Div(id="llm-token-status", style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "6px"}),
                                    ]),
                                    html.Div([
                                        html.Label("Temperature"),
                                        dcc.Input(id="llm-temp", type="number", value=DEFAULT_LLM_TEMP, placeholder="0.1", step=0.05, style={"width": "100%"}),
                                    ]),
                                    html.Hr(),
                                    html.Div([
                                        html.Label("DO Base URL"),
                                        dcc.Input(id="do-base-url", type="text", value=os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1"), style={"width": "100%"}),
                                    ]),
                                    html.Div([
                                        html.Label("DO Model"),
                                        dcc.Input(id="do-model", type="text", value=os.getenv("DO_AI_MODEL", "llama3-8b-instruct"), style={"width": "100%"}),
                                        dcc.Dropdown(id="do-model-dd", options=[], placeholder="Pick DO model (optional)", style={"marginTop": "6px"}),
                                        html.Div(id="do-model-status", style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "6px"}),
                                    ]),
                                    html.Div([
                                        html.Label("HF Base URL"),
                                        dcc.Input(id="hf-base-url", type="text", value=os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1"), style={"width": "100%"}),
                                    ]),
                                    html.Div([
                                        html.Label("HF Model"),
                                        dcc.Input(id="hf-model", type="text", value=os.getenv("HF_MODEL", "OpenAI/gpt-oss-20B"), style={"width": "100%"}),
                                        dcc.Dropdown(id="hf-model-dd", options=[], placeholder="Pick HF model (optional)", style={"marginTop": "6px"}),
                                        html.Div(id="hf-model-status", style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "6px"}),
                                    ]),
                                ])
                            ]),
                        ])
                    ]),
                ]),

                # The "Run" button is now outside the tabs, always visible
                html.Button("🚀 Run Analysis", id="research-run", className="btn", style={"width": "100%", "marginTop": "20px", "padding": "12px"}),
                html.Div(id="research-status", style={"textAlign": "center", "marginTop": "10px"}),
            ],
        ),
        
        # --- MAIN CONTENT AREA (UNCHANGED) ---
        html.Div(id="research-output-container", style={"paddingTop": "200px"}, children=[
            dcc.Loading(id="research-loader", type="cube", color="#3b82f6", children=html.Div(id="research-content")),
        ]),
    ],
)

app.layout = serve_layout

# History tab layout
def history_tab_layout():
    return html.Div(id="history-tab-content", style={"padding":"20px"})

# History tab renderer (placed before server start so it registers)
@app.callback(
    Output("history-tab-content", "children"),
    Input("research-history", "data"),
    Input("tabs", "value"),
    Input("history-selected", "data")
)
def render_history_tab(history, active_tab, selected_idx):
    if active_tab != 'tab-history':
        raise dash.exceptions.PreventUpdate
    items = history or []
    if not items:
        return html.Div(className="card", style={"textAlign":"center","padding":"40px"}, children=[
            html.Div("🕒", style={"fontSize":"48px","marginBottom":"12px"}),
            html.Div("No research history yet", style={"fontSize":"18px","fontWeight":"500","marginBottom":"8px"}),
            html.Div("Run some analysis to see your research timeline here", style={"color":"var(--text-muted)"})
        ])

    def _fmt_ts(ts):
        try:
            dt = datetime.fromisoformat(ts.replace('Z','+00:00'))
            return dt.strftime('%b %d, %Y at %H:%M')
        except Exception:
            return ts

    def _get_analysis_badge(analysis_type):
        badge_styles = {
            'combined': {'bg':'linear-gradient(135deg, #6366f1, #8b5cf6)', 'icon':'🔬'},
            'technical': {'bg':'linear-gradient(135deg, #10b981, #059669)', 'icon':'📈'},
            'fundamental': {'bg':'linear-gradient(135deg, #f59e0b, #d97706)', 'icon':'💰'},
            'web': {'bg':'linear-gradient(135deg, #0ea5e9, #0284c7)', 'icon':'🌐'},
        }
        style = badge_styles.get(analysis_type, {'bg':'linear-gradient(135deg, #64748b, #475569)', 'icon':'🔍'})
        return html.Span([
            style['icon'], ' ', (analysis_type or 'Unknown').title()
        ], style={
            "background": style['bg'],
            "color": "#fff",
            "fontSize": "11px",
            "padding": "4px 10px",
            "borderRadius": "16px",
            "fontWeight": "500",
            "letterSpacing": "0.3px",
            "textShadow": "0 1px 2px rgba(0,0,0,0.3)"
        })

    def _get_rec_badge(rec, conf):
        if not rec:
            return None
        rec_colors = {
            'buy': '#10b981', 'strong buy': '#059669',
            'sell': '#ef4444', 'strong sell': '#dc2626', 
            'hold': '#f59e0b'
        }
        color = rec_colors.get(rec.lower(), '#6366f1')
        return html.Span([
            "🎯 ", f"{rec} ({conf})"
        ], style={
            "background": color,
            "color": "#fff",
            "fontSize": "11px",
            "padding": "4px 8px",
            "borderRadius": "12px",
            "fontWeight": "500"
        })

    nodes = []
    for idx, h in enumerate(items):
        is_expanded = (selected_idx == idx)
        full_summary = h.get('summary', '')
        # Create preview (first ~320 chars now, but break at sentence/word boundary)
        # Increase preview length for more context before expansion
        preview_length = 430
        if len(full_summary) <= preview_length:
            preview_text = full_summary
        else:
            # Try to break at sentence end first
            preview_text = full_summary[:preview_length]
            sentence_end = preview_text.rfind('. ')
            if sentence_end > preview_length * 0.6:  # Only use sentence break if it's not too short
                preview_text = preview_text[:sentence_end + 1]
            else:
                # Break at word boundary
                space_pos = preview_text.rfind(' ')
                if space_pos > preview_length * 0.8:
                    preview_text = preview_text[:space_pos] + '...'
                else:
                    preview_text = preview_text + '...'
        
        card_id = {"type": "history-card", "index": idx}
        
        nodes.append(html.Div(
            className="news-card clickable",
            id=card_id,
            n_clicks=0,
            style={"marginBottom":"0"},  # Remove margin since we're using grid gap
            children=[
                html.Div(className="news-head", children=[
                    html.Div("📊", style={"fontSize":"20px","marginRight":"8px"}),
                    html.Div([
                        html.Div(f"{h.get('symbol','?')} Analysis", className="news-title", style={"marginBottom":"4px"}),
                        html.Div([
                            _get_analysis_badge(h.get('analysis_type')),
                            _get_rec_badge(h.get('recommendation'), h.get('rec_confidence')),
                        ], style={"display":"flex","gap":"8px","alignItems":"center","flexWrap":"wrap"})
                    ], style={"minWidth": "0"}),
                    html.Button(
                        "🗑️",
                        id={"type": "history-delete", "index": idx},
                        title="Delete this entry",
                        n_clicks=0,
                        className="btn",
                        style={
                            "padding": "4px 8px",
                            "border": "1px solid var(--border-secondary)",
                            "background": "var(--bg-elevated)",
                            "borderRadius": "6px",
                            "cursor": "pointer",
                            "gridColumn": "3",
                            "gridRow": "1",
                            "justifySelf": "end",
                            "alignSelf": "start",
                        }
                    ),
                ]),
                html.Div(className="news-meta", children=[
                    "🕒 ", _fmt_ts(h.get('ts', '')),
                ]),
                # Always show preview text (like news description)
                html.Div(className="news-desc", children=preview_text),
                # Expandable summary section (only shows when clicked)
                html.Div(
                    id={"type": "history-summary", "index": idx},
                    className=("summary" + (" open" if is_expanded else "")),
                    children=[
                        html.Div("🤖 Full AI Analysis", className="summary-title"),
                        # Render markdown so **bold** and headings show properly
                        dcc.Markdown(
                            full_summary,
                            className="markdown-content",
                            style={"lineHeight": "1.55", "whiteSpace": "pre-wrap"}
                        )
                    ]
                )
            ]
        ))

    return html.Div([
        html.Div(style={"display":"flex","alignItems":"center","justifyContent":"space-between","marginBottom":"20px","gap":"12px"}, children=[
            html.H2("📚 Research History", style={"margin":"0","fontSize":"24px","fontWeight":"600"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"10px"}, children=[
                html.Div(f"{len(items)} of {HISTORY_LIMIT} entries", style={"color":"var(--text-muted)","fontSize":"14px"}),
                html.Button(
                    "Clear All",
                    id="history-clear-all",
                    className="btn",
                    n_clicks=0,
                    title="Delete all history",
                    style={
                        "padding": "6px 10px",
                        "border": "1px solid var(--border-secondary)",
                        "background": "var(--bg-elevated)",
                        "borderRadius": "6px",
                        "cursor": "pointer"
                    }
                ),
            ])
        ]),
        html.Div(className="news-list", children=nodes, style={"display":"grid","gap":"16px"})
    ])
@app.callback(
    Output("web-results-container", "style"),
    Input("allow-web-search", "value")
)
def toggle_web_results_visibility(allow_search):
    if allow_search == 'on':
        return {'display': 'block', 'marginTop': '15px'} # Show it
    else:
        return {'display': 'none'} # Hide it
@app.callback(
    Output("tab-news-container", "style"),
    Output("tab-strategy-container", "style"),
    Output("tab-research-container", "style"),
    Output("tab-history-container", "style"),
    Input("tabs", "value"),
)
def render_tab(tab):
    def _style_for(target):
        return {"display": "block"} if tab == target else {"display": "none"}

    return (
        _style_for("tab-news"),
        _style_for("tab-strategy"),
        _style_for("tab-research"),
        _style_for("tab-history"),
    )

# Save the currently selected tab into local storage
@app.callback(
    Output("tabs-memory", "data"),
    Input("tabs", "value"),
    prevent_initial_call=True,
)
def remember_tab(tab_val):
    return tab_val

# Restore the last selected tab on first load (or after dev hot reload)
@app.callback(
    Output("tabs", "value"),
    Input("tabs-primer", "n_intervals"),
    State("tabs-memory", "data"),
    prevent_initial_call=True,
)
def restore_tab(_tick, saved):
    return saved or "tab-news"
# Test local LLM endpoint quickly
@app.callback(
    Output("llm-test-status", "children"),
    Input("llm-test", "n_clicks"),
    State("llm-base-url", "value"),
    prevent_initial_call=True,
)
def test_local_llm(n, base_url):
    base = (base_url or "").strip()
    if not base:
        return "Enter a base URL."
    try:
        r = requests.get(base.rstrip("/") + "/models", timeout=6)
        if r.status_code != 200:
            return f"/models: HTTP {r.status_code}"
        j = r.json()
        count = len(j.get("data") or j.get("models") or [])
        return f"/models OK. Found {count} model(s)."
    except Exception as e:
        return f"/models error: {str(e)[:120]}"

# Clamp extreme token values to avoid server rejection (display-only helper)
@app.callback(
    Output("llm-token-status", "children"),
    Input("llm-max-tokens", "value"),
)
def clamp_tokens(v):
    try:
        val = int(v)
    except Exception:
        return ""
    if val > 8192:
        return "Note: Many local servers cap output tokens around 2k–8k. Values above 8192 may be ignored or rejected."
    return ""

# Fetch local model list from OpenAI-compatible endpoint
@app.callback(
    Output("llm-model-dd", "options"),
    Output("llm-model-dd", "value"),
    Output("llm-model-status", "children"),
    Input("llm-base-url", "value"),
    Input("llm-models-primer", "n_intervals"),
)
def refresh_local_models(base_url, _primer):
    base = (base_url or "").strip()
    if not base:
        return [], None, ""
    try:
        url = base.rstrip("/") + "/models"
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return [], None, f"Local models fetch failed: HTTP {r.status_code}"
        data = r.json()
        items = data.get("data") or data.get("models") or []
        names = []
        for it in items:
            # OpenAI schema: id is the model name
            name = it.get("id") if isinstance(it, dict) else None
            if isinstance(it, str):
                name = it
            if name:
                names.append(str(name))
        names = sorted(set(names))
        options = [{"label": n, "value": n} for n in names]
        # Auto-select current DEFAULT if present
        sel = DEFAULT_LLM_MODEL if DEFAULT_LLM_MODEL in names else (names[0] if names else None)
        status = f"Found {len(names)} models from local endpoint." if names else "No models found at endpoint."
        return options, sel, status
    except Exception as e:
        return [], None, f"Local models fetch error: {str(e)[:120]}"

# Sync dropdown choice into the text input so downstream code reads a single source
@app.callback(
    Output("llm-model", "value"),
    Input("llm-model-dd", "value"),
    State("llm-model", "value"),
)
def sync_llm_model(selected, current):
    if selected:
        return selected
    return current

# Coalesce dropdown and custom input into a single active asset value
@app.callback(
    Output("active-asset", "data"),
    Input("asset-dd", "value"),
    Input("asset-custom", "value")
)
def choose_asset(dd_val, custom_val):
    s = (custom_val or "").strip()
    return s or dd_val or "BTC"

@app.callback(
    Output("tv-container", "children"),
    Input("active-asset", "data")
)
def update_tv(asset):
    symbol = tv_symbol(asset)
    h = 720  # Fixed chart height
    inter = "60"
    tv_html = f"""
    <div id='tvchart' style='height:{h}px;'></div>
    <script type='text/javascript' src='https://s3.tradingview.com/tv.js'></script>
    <script type='text/javascript'>
        new TradingView.widget({{
            autosize: true,
            symbol: '{symbol}',
            interval: '{inter}',
            timezone: 'Etc/UTC',
            theme: 'dark',
            style: '1',
            locale: 'en',
            toolbar_bg: '#0b1220',
            enable_publishing: false,
            allow_symbol_change: true,
            hide_top_toolbar: false,
            hide_side_toolbar: false,
            withdateranges: true,
            container_id: 'tvchart'
        }});
    </script>
    """
    return html.Iframe(srcDoc=tv_html, style={"width": "100%", "height": f"{h+30}px", "border": "0"})

# Fetch news + analysis (heavy)
@app.callback(
    Output("news-data", "data"),
    Output("overall-summary", "data"),
    Output("update-meta", "data"),
    Output("news-loading-probe", "children"),
    Input("active-asset", "data"),
    Input({"type": "news-refresh", "scope": ALL}, "n_clicks"),
    Input("model-pref", "value"),
    Input("news-count", "value"),
    Input("news-days", "value"),
    Input("tabs", "value"),
    State("news-do-model-dd", "value"),
    State("news-hf-model-dd", "value"),
    State("news-data", "data"),
    State("overall-summary", "data"),
    State("update-meta", "data"),
    prevent_initial_call=True,
)
def fetch_news(asset, refresh_clicks, model_pref, count, days_back, current_tab, do_model_choice, hf_model_choice, existing_data, existing_overall, existing_meta):
    # Do not perform heavy fetches when user isn't on News tab; just keep existing state
    if current_tab != 'tab-news':
        raise dash.exceptions.PreventUpdate
    
    ctx = dash.callback_context
    trig_id = (ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "initial")

    # Do not refetch just because the user switched to the News tab
    if trig_id == "tabs":
        return (
            existing_data or [],
            (existing_overall or {}),
            (existing_meta or {}),
            (existing_meta or {}).get("ts", ""),
        )

    if trig_id in {"news-count", "news-days"} and existing_data:
        return existing_data, (existing_overall or {}), (existing_meta or {}), (existing_meta or {}).get("ts", "")

    prev_asset = (existing_meta or {}).get("asset")
    if trig_id == "active-asset" and existing_data and prev_asset == asset:
        return existing_data, (existing_overall or {}), (existing_meta or {}), (existing_meta or {}).get("ts", "")

    # Try disk cache first to avoid refetching (cache key uses asset+days+count)
    try:
        cache_key = f"{asset}|d{int(days_back or 7)}|n{int(count or 0)}"
    except Exception:
        cache_key = f"{asset}|d7|n{int(count or 0) if str(count).isdigit() else 0}"
    cache_ttl_min = int(os.getenv("NEWS_CACHE_TTL_MIN", "30") or 30)
    cached = _news_cache_get(cache_key, ttl_minutes=cache_ttl_min)
    if cached and trig_id not in {"model-pref"}:  # allow re-summarize on model change
        items_for_summary, overall, meta = cached
        # Quality gate: if cached result looks low quality, ignore cache and refetch+reanalyze
        try:
            missing_ai = sum(1 for it in (items_for_summary or []) if not (it or {}).get("ai_summary"))
            total = len(items_for_summary or [])
            provider = (overall or {}).get("provider", "")
            summary_txt = (overall or {}).get("summary", "")
            low_quality = (provider == "error") or (not summary_txt) or (total > 0 and missing_ai >= max(1, total // 2))
        except Exception:
            low_quality = False

        if not low_quality:
            return items_for_summary, overall, meta, meta.get("ts", "")
        else:
            print("DEBUG (app.py): Cached entry deemed low-quality (will refetch):",
                  f"missing_ai={missing_ai}/{total}", f"overall_provider={provider}")

    max_articles = int(count or 0)
    if max_articles <= 0:
        meta = {"asset": asset, "ts": datetime.now(timezone.utc).isoformat(), "fetch_summary": "News disabled (count=0)"}
        return [], {"summary": "", "provider": ""}, meta, meta["ts"]

    if trig_id == "model-pref" and existing_data and prev_asset == asset:
        print("DEBUG (app.py): AI model changed. Re-using fetched articles to generate a new summary.")
        items_for_summary = existing_data
    else:
        print(f"DEBUG (app.py): Performing a full news fetch for '{asset}'.")
        # Set AI provider env vars
        if (model_pref or "auto") == "do" and (do_model_choice or os.getenv("DO_AI_MODEL")):
            os.environ["AI_PROVIDER"] = "do"
            if do_model_choice: os.environ["DO_AI_MODEL"] = str(do_model_choice)
        if (model_pref or "auto") == "hf":
            os.environ["AI_PROVIDER"] = "hf"
            chosen = (hf_model_choice or os.getenv("HF_MODEL") or "Qwen/Qwen3-4B-Base")
            os.environ["HF_MODEL"] = str(chosen)
            if not os.getenv("HF_BASE_URL"): os.environ["HF_BASE_URL"] = "https://router.huggingface.co/v1"
        
        days = min(max(int(days_back or 7), 1), 60)
        
        items = bridge.fetch(asset, days_back=days, max_articles=max_articles, model_preference=model_pref or "auto", analyze=True)
        # Ensure JSON-serializable dicts: dataclasses to dict, drop any non-serializable fields
        try:
            from dataclasses import asdict as _asdict
            items_for_summary = [_asdict(i) for i in (items or [])]
        except Exception:
            items_for_summary = [i.__dict__ for i in (items or [])]

    # --- FIX #2: IMPROVED ERROR HANDLING FOR THE SUMMARY CALL ---
    # This block will now always run unless an early exit happened.
    overall = {"summary": "", "provider": ""}
    try:
        # The 'data' passed to summarize_overall must be in the correct format.
        # NewsBridge now returns NewsItem objects, so we pass that directly if available.
        # If we reused 'existing_data', it's already a list of dicts.
        if items_for_summary and isinstance(items_for_summary[0], dict):
             # Convert list of dicts to list of NewsItem objects for the function
            try:
                from .services import NewsItem
            except Exception:
                from services import NewsItem
            news_item_objects = [NewsItem(**d) for d in items_for_summary]
        else:
            news_item_objects = items_for_summary
        
        # Now, call the summarize function
        if news_item_objects:
             print(f"DEBUG (app.py): Calling summarize_overall with {len(news_item_objects)} items.")
             overall = bridge.summarize_overall(asset, news_item_objects, model_preference=model_pref or "auto", max_chars=60000)
        else:
             print("DEBUG (app.py): No items to summarize.")

    except Exception as e:
        # This will print any error from the summarizer to your console!
        print(f"!!!!!!!!!!!!!! ERROR IN SUMMARIZE_OVERALL !!!!!!!!!!!!!!")
        print(f"Asset: {asset}, Model Pref: {model_pref}")
        import traceback
        traceback.print_exc()
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        overall = {"summary": f"Error during summary generation: {e}", "provider": "error"}

    # Finalize and return data
    fetch_summary = bridge.get_fetch_summary() or ""
    sast = timezone(timedelta(hours=2))
    meta = {"asset": asset, "ts": datetime.now(sast).isoformat(), "fetch_summary": fetch_summary}
    probe = meta["ts"]

    # 'items_for_summary' is already the list of dicts we need for the news-data store
    try:
        _news_cache_put(cache_key, items_for_summary or [], overall or {}, meta or {})
    except Exception:
        pass
    return items_for_summary, overall, meta, probe


# Render news list + inline summaries (light)
@app.callback(
    Output("news-feed", "children"),
    Input("news-data", "data"),
    Input("selected-idx", "data"),
    Input({"type": "news-refresh", "scope": ALL}, "n_clicks"),
    Input("update-meta", "data"),
    prevent_initial_call=True
)
def render_news(data, selected_idx, refresh_clicks, meta):
    # If refresh button triggered, clear list immediately and reset scroll
    try:
        ctx = dash.callback_context
        if ctx and ctx.triggered:
            prop = ctx.triggered[0]["prop_id"].split(".")[0]
            is_refresh = False
            if prop.startswith("{"):
                try:
                    payload = json.loads(prop)
                    is_refresh = (payload.get("type") == "news-refresh")
                except Exception:
                    is_refresh = ("news-refresh" in prop)
            else:
                is_refresh = ("news-refresh" in prop)
            # Only clear if no new data payload is present yet
            if is_refresh and not (data and isinstance(data, list) and len(data) > 0):
                # Return empty list wrapped in a new container id to force remount
                # refresh_clicks will be a list when using ALL; coerce to an int counter
                try:
                    rc = 0 if refresh_clicks is None else (refresh_clicks[0] if isinstance(refresh_clicks, list) and refresh_clicks else int(refresh_clicks))
                except Exception:
                    rc = 0
                wid = f"news-list-refresh-{int(rc or 0)}"
                return html.Div(id=wid)
    except Exception:
        pass
    if not data:
        wid = f"news-list-ts-{(meta or {}).get('ts','na')}"
        return html.Div([html.Div("No recent articles.", className="card")], id=wid)

    def _badge(item):
        label = (item.get('sentiment_label') or '').lower()
        parts = []
        if label:
            if label == 'bullish':
                cls = 'badge badge-bullish'
                icon = '📈'
            elif label == 'bearish':
                cls = 'badge badge-bearish'
                icon = '📉'
            else:
                cls = 'badge badge-neutral'
                icon = '➖'
            score = item.get('sentiment_score')
            score_txt = f" {score:+.2f}" if (score is not None) else ""
            parts.append(html.Span([icon, " ", label + score_txt], className=cls))
        # Always show provider chip; default to FALLBACK when missing
        provider = (item.get('analysis_provider') or 'fallback').upper()
        parts.append(html.Span(f"🤖 {provider}", className="provider"))
        if not parts:
            return None
        return html.Span(parts, className="badge-wrap")

    def _domain(u):
        try:
            return urlparse(u).netloc
        except Exception:
            return ''

    def _fmt_date(s):
        try:
            return datetime.fromisoformat((s or '').replace('Z','+00:00')).strftime('%Y-%m-%d %H:%M')
        except Exception:
            return s or ''

    # The selected_idx now directly corresponds to the visual item's index.
    # No complex URL lookup is needed.

    # Round-robin across domains with an adaptive per-domain cap
    from collections import OrderedDict, deque
    groups = OrderedDict()
    order = []
    for item in data:
        d = _domain(item.get('url')) or 'other'
        if d not in groups:
            groups[d] = deque()
            order.append(d)
        groups[d].append(item)
    used = {d: 0 for d in groups}
    # If there are 1-2 domains, don't cap so all requested cards render. Otherwise cap to 3 per domain.
    if len(groups) <= 2:
        max_per_domain = len(data)
    else:
        max_per_domain = 3
    ordered_items = []
    # Interleave one-by-one
    while True:
        progressed = False
        for d in list(order):
            if used.get(d, 0) >= max_per_domain:
                continue
            q = groups.get(d)
            if q and len(q) > 0:
                ordered_items.append(q.popleft())
                used[d] = used.get(d, 0) + 1
                progressed = True
        if not progressed:
            break

    # Build nodes from ordered_items; expansion is now a direct index match
    nodes = []
    for idx, item in enumerate(ordered_items):
        is_expanded = (idx == selected_idx)
        summary_content = (item.get('ai_summary') or item.get('description') or "")
        card_id = {"type": "news-card", "index": idx}
        nodes.append(html.Div(className="news-card clickable", id=card_id, n_clicks=0, children=[
            html.Div(className="news-head", children=[
                html.Img(src=f"https://www.google.com/s2/favicons?domain={_domain(item.get('url'))}&sz=32", className="favicon"),
                html.A(item.get('title'), href=item.get('url') or '#', target="_blank", className="news-title"),
                _badge(item)
            ]),
            html.Div(className="news-meta", children=[
                "🕒 ", _fmt_date(item.get('published_at')), " • ",
                "📰 ", item.get('source') or _domain(item.get('url'))
            ]),
            html.Div(className="news-desc", children=(item.get('description') or "")),
            html.Div(id={"type": "summary-area", "index": idx}, className=("summary" + (" open" if is_expanded else "")), children=[
                html.Div("🤖 AI Analysis", className="summary-title"),
                html.Div(summary_content, style={"lineHeight": "1.6"})
            ])
        ]))

    wid = f"news-list-ts-{(meta or {}).get('ts','na')}"
    return html.Div(html.Div(nodes, className="news-list"), id=wid)

@app.callback(
    Output("overall-card", "children"),
    Input("overall-summary", "data"),
    Input("active-asset", "data"),
    Input("tabs", "value"),
    Input("update-meta", "data"),
    Input("news-data", "data"),
)
def render_overall(overall, asset, tab_value, meta, news_data):
    if not overall or not overall.get("summary"):
        return html.Div(className="card", children=[
            html.Div(style={"textAlign": "center", "padding": "20px"}, children=[
                html.Div("🤖", style={"fontSize": "48px", "marginBottom": "12px"}),
                html.Div(f"Overall AI Summary for {asset}", style={"fontWeight": 600, "marginBottom": "8px", "fontSize": "16px"}),
                html.Div("No overall summary available yet. Try adjusting the AI model or refreshing the data.", 
                        style={"color": "var(--text-secondary)"})
            ])
        ])
    provider = (overall.get("provider") or '').upper()
    chip = html.Span(f"🤖 {provider}", className="provider") if provider else html.Span("🛟 FALLBACK", className="provider")

    # Sentiment badge from payload if available
    payload = overall.get("payload") or {}
    olabel = (payload.get("overall_sentiment") or overall.get("overall_sentiment") or "").lower()
    conf = payload.get("confidence")
    if isinstance(conf, str):
        try:
            conf = float(conf)
        except Exception:
            conf = None
    conf_txt = f" {conf:.0%}" if isinstance(conf, (int, float)) else ""
    if olabel == 'bullish':
        s_cls, s_icon, s_txt = 'badge badge-bullish', '📈', f"Bullish{conf_txt}"
    elif olabel == 'bearish':
        s_cls, s_icon, s_txt = 'badge badge-bearish', '📉', f"Bearish{conf_txt}"
    elif olabel == 'neutral':
        s_cls, s_icon, s_txt = 'badge badge-neutral', '➖', f"Neutral{conf_txt}"
    else:
        s_cls, s_icon, s_txt = 'badge', '', ''
    sent_badge = html.Span([html.Span([s_icon, " ", s_txt], className=s_cls)]) if s_txt else None
    # Build a small sources footer (providers + visible domains) to place under the summary
    prov = None
    try:
        prov = (meta or {}).get("fetch_summary") or ""
    except Exception:
        prov = ""
    # Domain counts follow the same cap used in render_news
    def _domain(u):
        try:
            return urlparse(u or "").netloc
        except Exception:
            return ""
    max_per_domain = 3
    per = {}
    if news_data:
        for it in news_data:
            d = _domain(it.get("url"))
            if d:
                c = per.get(d, 0)
                if c >= max_per_domain:
                    continue
                per[d] = c + 1
    footer = None
    if prov or per:
        parts = []
        if prov:
            parts.append(f"Sources (providers): {prov}")
        if per:
            top = sorted(per.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            dom_txt = ", ".join([f"{k}:{v}" for k, v in top])
            total = sum(per.values())
            parts.append(f"Visible domains (round-robin): {dom_txt} — Shown: {total}")
        footer = html.Div(" — ".join(parts), style={
            "fontSize": "12px",
            "color": "var(--text-muted)",
            "marginTop": "12px",
            "borderTop": "1px solid var(--border-secondary)",
            "paddingTop": "8px"
        })

    return html.Div(className="card", children=[
        html.Div(style={"display":"flex", "alignItems":"center", "gap":"12px", "marginBottom": "12px"}, children=[
            html.Div("🧠", style={"fontSize": "24px"}),
            html.Div(f"Overall AI Summary for {asset}", style={"fontWeight": 600, "fontSize": "16px", "flex": "1"}),
            sent_badge,
            chip
        ]),
        html.Div(overall.get("summary"), style={"lineHeight": "1.6", "color": "var(--text-secondary)"}),
        footer,
    ])

@app.callback(
    Output("selected-idx", "data"),
    Input({"type": "news-card", "index": ALL}, "n_clicks"),
    Input({"type": "history-delete", "index": ALL}, "n_clicks"),
    State("selected-idx", "data"),
    prevent_initial_call=True,
)
def toggle_selected(n_clicks, delete_clicks, current_selected_idx):
    # If no cards have been clicked yet, do nothing
    if not any(n_clicks):
        raise dash.exceptions.PreventUpdate

    # Use dash.callback_context to find out exactly which card was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # The ID of the clicked element is a dictionary, e.g., {'type': 'news-card', 'index': 1}
    triggered_id = ctx.triggered_id
    # Ignore delete button triggers here so they don't toggle expansion
    if isinstance(triggered_id, dict) and triggered_id.get("type") == "history-delete":
        raise dash.exceptions.PreventUpdate
    clicked_index = triggered_id['index'] if isinstance(triggered_id, dict) else None
    
    # If the user clicks the same card that is already open, close it
    if clicked_index == current_selected_idx:
        return None
    
    # Otherwise, store the index of the newly clicked card
    return clicked_index

# Simplify toolbar: show provider-specific controls only when selected
@app.callback(
    Output("do-controls", "style"),
    Output("hf-controls", "style"),
    Input("model-pref", "value"),
)
def _toggle_provider_controls(model_pref):
    mp = (model_pref or "auto").lower()
    show = {"display": "block"}
    hide = {"display": "none"}
    return (show if mp == "do" else hide, show if mp == "hf" else hide)

# --- START: WALLET STRATEGY SECTION ---

def generate_llm_strategy(chain, balance, token_count, tokens_list):
    """
    Uses the local LLM to generate a more dynamic wallet strategy.
    """
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip()
    if not base_url:
        return ["LLM not configured. Set LOCAL_LLM_BASE_URL to enable AI strategy."]

    prompt = f"""
    You are an expert crypto portfolio analyst. A user has provided a snapshot of their wallet.
    Based on the data below, provide a concise, actionable analysis and strategy in 3-4 bullet points.

    Wallet Data:
    - Blockchain: {chain}
    - Native Coin Balance ({'BTC' if chain == 'bitcoin' else 'ETH'}): {balance:.6f}
    - Number of Different Tokens: {token_count}
    - Notable Tokens: {', '.join(tokens_list[:5]) if tokens_list else 'N/A'}

    Your analysis should consider the wallet's posture (e.g., concentrated, diversified, passive holding)
    and suggest potential next steps or areas to research. Be encouraging and insightful.
    Example:
    - Your portfolio appears to be a passive holding with a focus on the native asset. This is a solid, lower-risk strategy.
    - With {token_count} different assets, you have a good level of diversification. Consider reviewing underperforming assets quarterly.
    - To increase potential upside, you could explore adding a small position in a trending narrative like AI or DePIN.
    """
    
    system_message = "You are a helpful crypto analyst that provides brief, actionable advice."
    
    try:
        payload = {
            "model": os.getenv("LOCAL_LLM_MODEL", "llama-3.1-8b-instruct"),
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 300,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
        if api_key and api_key != "not-needed":
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        
        content = response.json()['choices'][0]['message']['content']
        strategy_points = [p.strip().lstrip('-* ').capitalize() for p in content.split('\n') if p.strip()]
        return strategy_points

    except Exception as e:
        print(f"LLM strategy generation failed: {e}")
        return ["AI strategy could not be generated at this time."]

@app.callback(
    Output("strategy-output", "children"),
    Input("analyze-btn", "n_clicks"),
    State("chain-dd", "value"),
    State("wallet-input", "value"),
    prevent_initial_call=True
)
def analyze_wallet(n_clicks, chain, addr):
    if not addr:
        return html.Div("Enter an address.")
    
    blockchair_api_key = os.getenv("BLOCKCHAIR_API_KEY", "")

    try:
        def _api_error_msg(resp, provider_name: str):
            try:
                body = resp.json()
                ctx = body.get("context", {}) if isinstance(body, dict) else {}
                detail = ctx.get("error") or body.get("message") or str(body)[:200]
            except Exception:
                detail = (resp.text or "").strip()[:200]
            hint = []
            if resp.status_code in (402, 403, 429):
                if not blockchair_api_key:
                    hint.append("No BLOCKCHAIR_API_KEY set. You may be rate limited.")
                else:
                    hint.append("Your API key may be invalid or out of quota.")
            return f"{provider_name} error {resp.status_code}: {detail}" + (" — " + "; ".join(hint) if hint else "")

        if chain == "bitcoin":
            url = f"https://api.blockchair.com/bitcoin/dashboards/address/{addr}"
        elif chain == "ethereum":
            url = f"https://api.blockchair.com/ethereum/dashboards/address/{addr}?erc_20=true" # Add parameter to ensure tokens are included
        else:
            return html.Div("Unsupported chain.")
        
        data = None
        r = None
        blockchair_status_note = None

        if blockchair_api_key:
            try:
                params = {"key": blockchair_api_key}
                r = requests.get(url, params=params, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                elif r.status_code >= 500:
                    blockchair_status_note = "Blockchair is temporarily unavailable (Server Error). Using fallback data source."
                    print(f"Blockchair API Error: Status {r.status_code} - {r.text[:200]}")
            except Exception as e:
                print(f"Blockchair request failed: {e}")
                r = None

        used_fallback = False
        fallback_note = None
        if data is None:
            used_fallback = True
            fallback_note = blockchair_status_note or "Used fallback data source. For more detailed analysis, please add a Blockchair API key."
            
          
        # --- DATA PARSING AND STRATEGY GENERATION ---
        idea = []
        if chain == "bitcoin":
            # Bitcoin parsing logic
            bal_sats = 0
            if data:
                if data.get('_fallback'):
                    bal_sats = data.get('btc_balance_sats', 0)
                else:
                    data_data = data.get('data')
                    if data_data and addr in data_data:
                        address_info = data_data[addr].get('address') if data_data[addr] else None
                        if address_info:
                            bal_sats = address_info.get('balance', 0)
            bal_btc = bal_sats / 1e8
            idea.append(f"BTC balance: {bal_btc:.8f} BTC")
            token_count = 0
            tokens_list = []
        else: # Ethereum parsing logic
            eth_bal = 0
            token_count = 0
            tokens_list = []
            if data:
                if data.get("_fallback"):
                    eth_bal = data.get("eth_balance", 0)
                    token_count = data.get("erc20_count", 0)
                else:
                    data_data = data.get('data')
                    if data_data and addr in data_data:
                        address_info = data_data[addr].get('address') if data_data[addr] else None
                        if address_info:
                            balance_in_wei = address_info.get('balance', 0)
                            eth_bal = int(balance_in_wei) / 1e18
                        erc20_tokens = data_data[addr].get('tokens', []) if data_data[addr] else []
                        if erc20_tokens:
                            token_count = len(erc20_tokens)
                            tokens_list = [t.get('token_symbol', 'Unknown') for t in erc20_tokens]
            idea.append(f"ETH balance: {eth_bal:.6f} ETH")
            idea.append(f"ERC-20 tokens: {token_count}")
        
        # AI-Powered Strategy Generation
        ai_strategy_points = generate_llm_strategy(
            chain=chain,
            balance=eth_bal if chain == "ethereum" else bal_btc,
            token_count=token_count,
            tokens_list=tokens_list
        )
        strategy = [f"🚀 {point}" for point in ai_strategy_points]

        # Assemble the final output display
        nodes = [
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
                html.Div(children=[
                    html.Div("💰 Wallet Insights", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px", "color": "var(--accent-primary)"}),
                    html.Ul([html.Li(x, style={"marginBottom": "8px", "lineHeight": "1.5"}) for x in idea], style={"paddingLeft": "20px"})
                ]),
                html.Div(children=[
                    html.Div("🤖 AI Strategy Recommendations", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px", "color": "var(--accent-primary)"}),
                    html.Ul([html.Li(x, style={"marginBottom": "8px", "lineHeight": "1.5"}) for x in strategy], style={"paddingLeft": "20px"})
                ])
            ])
        ]
        if used_fallback and fallback_note:
            nodes.append(html.Div(["ℹ️ ", fallback_note], style={
                "color": "var(--text-muted)", "fontSize": "13px", "marginTop": "16px",
                "padding": "12px", "background": "var(--warning-bg)", "border": "1px solid var(--warning-border)", "borderRadius": "8px"
            }))
        
        return html.Div(nodes)
        
    except Exception as e:
        return html.Div(f"An unexpected error occurred: {e}")

# --- END: WALLET STRATEGY SECTION ---

# --- UX: Refresh status & button state (server-side, simple) ---
@app.callback(
    Output("refresh-status", "children"),
    Output({"type": "news-refresh", "scope": "news"}, "disabled"),
    Input("update-meta", "data"),
    Input("tabs", "value"),
    prevent_initial_call=True,
)
def show_refresh_status(meta, tab):
    # Only update when News tab is active, otherwise prevent update
    if tab != "tab-news":
        raise dash.exceptions.PreventUpdate
        
    try:
        ts = (meta or {}).get("ts")
        if ts:
            try:
                t = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                label = f"Updated {t.strftime('%H:%M:%S')} (UTC+2)"
            except Exception:
                label = "Updated"
            return label, False
        return "", False
    except Exception:
        return "", False

# Use the built-in deterministic orchestrator shim; AutoGen team is deprecated/removed
try:
    from .agents.agents import run_research_workflow
except Exception:
    from agents.agents import run_research_workflow
from telemetry import log_event, Timer

# ---------------- Research callbacks ----------------

# List DO models for News tab (top-level)
@app.callback(
    Output("news-do-model-dd", "options"),
    Input("do-models-primer", "n_intervals"),
    prevent_initial_call=True,
)
def list_do_models_news(_primer):
    base = os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1")
    return _get_do_model_options(base)

# Preload DO models for Research tab as well (top-level)
@app.callback(
    Output("do-model-dd", "options"),
    Output("do-model-status", "children"),
    Input("do-models-primer", "n_intervals"),
    State("do-base-url", "value"),
    prevent_initial_call=True,
)
def list_do_models_research(_primer, base_url):
    opts = _get_do_model_options(base_url or os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1"))
    if not opts:
        # Distinguish between missing key and just no matches
        if not (os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY")):
            return [], "Set DO_AI_API_KEY in .env to list models."
        return [], "No low-end models matched or request failed."
    return opts, f"Loaded {len(opts)} low-end models."

# Curated HF list for News tab (low-end only)
@app.callback(
    Output("news-hf-model-dd", "options"),
    Input("do-models-primer", "n_intervals"),
    prevent_initial_call=True,
)
def list_hf_models_news(_primer):
    return CURATED_HF_MODELS

# Curated HF list for Research tab
@app.callback(
    Output("hf-model-dd", "options"),
    Output("hf-model-status", "children"),
    Input("do-models-primer", "n_intervals"),
    prevent_initial_call=True,
)
def list_hf_models_research(_primer):
    return CURATED_HF_MODELS, f"Loaded {len(CURATED_HF_MODELS)} curated model(s)."

# Autocomplete for research ticker dropdown (symbol-only, no names)
@app.callback(
    Output("research-symbol-dd", "options"),
    Input("research-symbol-dd", "search_value"),
    Input("tabs-primer", "n_intervals"),
    State("research-symbol-dd", "value"),
    prevent_initial_call=False,
)
def update_symbol_options(search_value, _tabs_primer, current_value):
    try:
        q = (search_value or "").strip()
        if not q:
            # No search - return symbol-only defaults
            return RESEARCH_SYMBOL_OPTIONS
        
        # Search through index but return symbol-only labels
        q_upper = q.upper()
        q_lower = q.lower()
        idx = _load_symbol_index() or []
        
        out = []
        for row in idx:
            sym = (row.get("symbol") or "").upper()
            name = (row.get("name") or "").strip()
            # Match by symbol or name
            if sym.startswith(q_upper) or (name and q_lower in name.lower()):
                out.append({"label": sym, "value": sym})
                if len(out) >= 20:
                    break
        
        # Fallback to filtered defaults if no matches
        if not out:
            base = [opt for opt in RESEARCH_SYMBOL_OPTIONS if opt["value"].startswith(q_upper)]
            return base[:20] if base else RESEARCH_SYMBOL_OPTIONS
        
        return out
    except Exception:
        return RESEARCH_SYMBOL_OPTIONS

# Autocomplete for News asset dropdown; also primes on page load via research primer interval
@app.callback(
    Output("asset-dd", "options"),
    Input("asset-dd", "search_value"),
    Input("llm-models-primer", "n_intervals"),
    prevent_initial_call=False,
)
def update_news_asset_options(search_value, _primer):
    try:
        return _search_symbols(search_value or "")
    except Exception:
        return DEFAULT_SYMBOL_OPTIONS
@app.callback(
    Output("research-content", "children"),
    Output("research-status", "children"),
    Output("research-history", "data"),
    Input("research-run", "n_clicks"),
    State("analysis-type", "value"),
    State("execution-engine", "value"),
    State("research-symbol-dd", "value"),
    State("research-symbol-input", "value"),
    State("research-timeframe", "value"),
    State("lookback-value", "value"),
    State("lookback-unit", "value"),
    State("technical-indicators", "value"),
    State("indicator-length", "value"),
    State("rsi-length", "value"),
    State("macd-fast", "value"),
    State("macd-slow", "value"),
    State("macd-signal", "value"),
    State("speed-mode", "value"),
    State("allow-web-search", "value"),
    State("web-max-results", "value"),
    State("ai-provider", "value"),
    State("llm-base-url", "value"),
    State("llm-model", "value"),
    State("llm-max-tokens", "value"),
    State("llm-temp", "value"),
    State("agent-rounds", "value"),
    State("do-base-url", "value"),
    State("do-model", "value"),
    State("do-model-dd", "value"),
    State("hf-base-url", "value"),
    State("hf-model", "value"),
    State("hf-model-dd", "value"),
    State("research-history", "data"),
    prevent_initial_call=True,
)
def run_research(
    n, analysis_type, engine, symbol_dd, symbol_input, timeframe, lookback_value,
    lookback_unit, indicators, length, rsi_length, macd_fast, macd_slow, macd_signal,
    speed_mode, allow_web_search, web_max_results, ai_provider,
    llm_base_url, llm_model, llm_max_tokens, llm_temp, agent_rounds, do_base_url,
    do_model_text, do_model_choice, hf_base_url, hf_model_text, hf_model_choice,
    existing_history
):
    # Guard: only run when button actually clicked
    if not n:
        raise dash.exceptions.PreventUpdate
    symbol = (symbol_input or "").strip() if symbol_input else (symbol_dd or "").strip()
    if not symbol:
        return html.Div("Enter a ticker."), "Idle", (existing_history or [])

    try:
        if ai_provider: os.environ["AI_PROVIDER"] = str(ai_provider)
        if llm_base_url: os.environ["LOCAL_LLM_BASE_URL"] = str(llm_base_url)
        if llm_model: os.environ["LOCAL_LLM_MODEL"] = str(llm_model)
        if llm_max_tokens: os.environ["LOCAL_LLM_MAX_TOKENS"] = str(int(llm_max_tokens))
        if llm_temp: os.environ["LOCAL_LLM_TEMPERATURE"] = str(float(llm_temp))
        if do_base_url: os.environ["DO_AI_BASE_URL"] = str(do_base_url)
        if (do_model_choice or do_model_text): os.environ["DO_AI_MODEL"] = str(do_model_choice or do_model_text)
        if hf_base_url: os.environ["HF_BASE_URL"] = str(hf_base_url)
        if (hf_model_choice or hf_model_text): os.environ["HF_MODEL"] = str(hf_model_choice or hf_model_text)
    except Exception as e:
        print(f"Error setting env vars: {e}")
        pass

    try:
        from datetime import datetime, timedelta
        lb = int(lookback_value or 180)
        if speed_mode == "fast": lb = min(lb, 60)
        elif speed_mode == "normal": lb = min(lb, 120)
        unit = (lookback_unit or "days").lower()
        now = datetime.now()
        delta = timedelta(days=lb * {'days': 1, 'weeks': 7, 'months': 30}.get(unit, 1))
        start_date = (now - delta).date().isoformat()
        end_date = now.date().isoformat()
    except Exception:
        start_date, end_date = None, None

    # 1. Build a single configuration dictionary
    config = {
        "symbol": symbol,
        "analysis_type": analysis_type,
        "start_date": start_date,
        "end_date": end_date,
        "interval": timeframe,
        "indicators": indicators,
        "length": int(length or 20),
        "rsi_length": int(rsi_length or 14),
        "macd_fast": int(macd_fast or 12),
        "macd_slow": int(macd_slow or 26),
        "macd_signal_len": int(macd_signal or 9),
        "ai_provider": ai_provider,
        "web_max_results": (int(web_max_results) if str(web_max_results).isdigit() else 6) if web_max_results is not None else 6,
    }

    try:
        with Timer() as t:
            out = research.execute_workflow(config)
    except Exception as e:
        log_event("research_error", {"error": str(e)})
        return html.Div(f"Analysis failed unexpectedly: {e}"), "Failed", (existing_history or [])

    # --- UI Rendering Section ---
    # Deprecated: Previously showed a plan-of-agents card in the overview; removed per request

    profile = out.get("company_profile") or {}
    profile_card = None
    if profile and profile.get("name") and profile.get("name") != symbol:
        profile_card = html.Div([
            html.H3(profile.get("name", symbol)),
            html.P(f"Sector: {profile.get('sector','N/A')} | Industry: {profile.get('industry','N/A')}", style={"margin":"0"}),
            html.P(profile.get("description",""), style={"fontSize":"14px", "color":"var(--text-muted)", "marginTop":"10px"}),
        ], className="card", style={"padding":"12px"})

    tech = out.get("technical") or {}
    tech_fig = out.get("figures", {}).get("technical")
    tech_section = None
    if analysis_type in ("technical", "combined"):
        
        # Clean the justification text from the agent's output
        justification_text = tech.get('justification', 'No strategic summary was generated.')
        cleaned_justification = strip_markdown_fences(justification_text)

        tech_section = html.Div([
            html.H4("Technical Analysis"),
            dcc.Graph(figure=tech_fig) if tech_fig else html.Div("No technical chart generated."),
            html.Div([
                html.P(f"Recommendation: {tech.get('recommendation','N/A')} | Confidence: {tech.get('confidence','N/A')}"),
                html.H5("Technical Strategy & Key Levels"),
                dcc.Markdown(
                    cleaned_justification #
                )
            ])
        ], className="card", style={"padding":"12px"})

    ratios = out.get("ratios") or {}
    figs = out.get("figures") or {}
    fundamental_summary_text = out.get("fundamental_summary")
    
    fundamentals_section = None
    if analysis_type in ("fundamental", "combined") and ratios:
        kv_pairs = [("P/E (TTM)", ratios.get("pe_ttm")), ("P/S (TTM)", ratios.get("ps_ttm")), ("Debt/Equity", ratios.get("debt_to_equity")),
                    ("ROE", ratios.get("roe")), ("ROA", ratios.get("roa")), ("Gross Margin", ratios.get("gross_margin")),
                    ("Net Margin", ratios.get("net_margin"))]
        ratio_rows = [html.Div([
            html.Span(k), html.Span(f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
        ], style={"display":"flex", "justifyContent":"space-between", "gap":"12px", "padding":"4px 0"}) for k,v in kv_pairs if v is not None]
        
        fund_children = []

        # --- AI-powered summary card at the top ---
        if fundamental_summary_text:
            fund_children.append(
                html.Div([
                    html.H4("AI-Powered Fundamental Analysis"),
                    # Use dcc.Markdown to render the formatted text from the LLM
                    dcc.Markdown(fundamental_summary_text, className="markdown-content")
                ], className="card", style={"padding": "12px"})
            )

        # Append the original cards for ratios and charts
        if ratio_rows:
            fund_children.append(html.Div([html.H4("Key Ratios"), html.Div(ratio_rows)], className="card", style={"padding":"12px"}))
        if figs.get("revenue"): fund_children.append(html.Div([html.H4("Revenue (Quarterly)"), dcc.Graph(figure=figs["revenue"])], className="card", style={"padding":"12px"}))
        if figs.get("pe_trend"): fund_children.append(html.Div([html.H4("P/E Trend (approx)"), dcc.Graph(figure=figs["pe_trend"])], className="card", style={"padding":"12px"}))
        fundamentals_section = html.Div(fund_children)

    news_items = out.get("news", [])
    news_summary = out.get("news_summary", {})
    # Determine how many news to display: prefer the same limit used during fetch
    desired_news_max = None
    cfg_for_news = out.get("config", {})
    # Prefer explicit fetched/requested counts if present
    requested_raw = cfg_for_news.get("web_requested") or cfg_for_news.get("web_max_results") or cfg_for_news.get("news_max_results")
    try:
        desired_news_max = int(requested_raw) if requested_raw is not None else 0
    except Exception:
        desired_news_max = 0
    fetched_count = len(news_items)
    if desired_news_max <= 0:
        desired_news_max = fetched_count
    news_section = None
    if analysis_type in ("web", "combined") and news_items:
        cards = [html.Div([
            html.A(it.get("title", "(no title)"), href=it.get("url"), target="_blank", className="news-title-link"),
            html.P(it.get("source",""), style={"fontSize":"12px", "color":"var(--text-muted)", "margin":"4px 0"}),
            html.P(it.get("ai_summary") or it.get("description") or "", style={"fontSize":"13px"})
        ], className="card", style={"padding":"10px"}) for it in news_items[:desired_news_max]]
        showing = min(desired_news_max, fetched_count)
        req_note = ""
        req = None
        try:
            req = int(requested_raw) if requested_raw is not None else None
        except Exception:
            req = None
        if req is not None and req > fetched_count:
            req_note = f" (requested {req})"
        news_section = html.Div([
            html.H4(f"News & Sentiment (showing {showing} of {fetched_count}{req_note})"),
            html.Div(cards, style={"display":"grid", "gap":"10px"}),
            dcc.Markdown(news_summary.get("summary",""), className="markdown-content", style={"marginTop":"15px"})
        ])

    holistic_text = out.get("holistic")
    hol_section = None
    if holistic_text:
        hol_section = html.Div([html.H4("Holistic Analysis"), dcc.Markdown(holistic_text, className="markdown-content")], className="card", style={"padding":"12px"})

    rec = out.get("recommendation") or {}
    rec_section = None
    if rec.get("justification"):
        rec_section = html.Div([
            html.H4("Final Recommendation"),
            html.P(f"{rec.get('recommendation','Hold')} (Confidence: {rec.get('confidence','N/A')})", style={"fontWeight":"bold"}),
            dcc.Markdown(rec.get("justification",""), className="markdown-content")
        ], className="card", style={"padding":"12px"})

    # --- Assemble Tabs ---
    tabs, overview_children = [], []
    overview_children.extend([c for c in [profile_card, hol_section, rec_section] if c is not None])
    tabs.append(dcc.Tab(label="Overview", value="overview", children=html.Div(overview_children, style={"display":"grid", "gap":"15px"})))
    if tech_section: tabs.append(dcc.Tab(label="Technical", value="tech", children=tech_section))
    if fundamentals_section: tabs.append(dcc.Tab(label="Fundamentals", value="fund", children=fundamentals_section))
    if news_section: tabs.append(dcc.Tab(label="News", value="news", children=news_section))

    conversations = out.get("conversations") or []
    if conversations:
        # Helper to extract an AI-generated text snippet for a given agent
        def _extract_text_snippet(text: str, max_len: int = 240) -> str:
            if not text:
                return ""
            try:
                cleaned = strip_markdown_fences(str(text))
            except Exception:
                cleaned = str(text)
            cleaned = cleaned.strip().replace("\n\n", " \u00b7 ").replace("\n", " ")
            if len(cleaned) <= max_len:
                return cleaned
            cutoff = cleaned.rfind(" ", 0, max_len)
            cutoff = cutoff if cutoff != -1 else max_len
            return cleaned[:cutoff].rstrip() + "..."

        def _get_agent_snippet(agent_name: str) -> str:
            agent = (agent_name or "").strip()
            try:
                if agent == "FundamentalSynthesisAgent":
                    return _extract_text_snippet(out.get("fundamental_summary") or "")
                if agent == "TechnicalAgent":
                    return _extract_text_snippet(((out.get("technical") or {}).get("justification")) or "")
                if agent == "HolisticAgent":
                    return _extract_text_snippet(out.get("holistic") or "")
                if agent == "RecommendationAgent":
                    return _extract_text_snippet(((out.get("recommendation") or {}).get("justification")) or "")
                if agent == "NewsAgent":
                    return _extract_text_snippet(((out.get("news_summary") or {}).get("summary")) or "")
            except Exception:
                pass
            return ""

        conv_nodes = []
        for item in conversations:
            a = item.get("agent") or "Agent"
            m = item.get("message") or ""
            snippet = _get_agent_snippet(a)
            children = [
                html.Div(a, style={"fontWeight":600, "color":"var(--accent-primary)"}),
                html.Pre(m, style={"whiteSpace":"pre-wrap", "margin":"4px 0 8px 0", "fontSize": "13px", "fontFamily": "var(--font-mono)"})
            ]
            if snippet:
                children.append(html.Div([
                    html.Span("AI snippet", style={"fontWeight":500, "fontSize":"12px", "color":"var(--text-muted)"}),
                    dcc.Markdown(snippet, className="markdown-content", style={"margin":"4px 0 0 0", "fontSize":"13px"})
                ]))
            conv_nodes.append(html.Div(children, className="card", style={"padding":"10px"}))

        tabs.append(dcc.Tab(label="Conversations", value="conv", children=html.Div(conv_nodes, style={"display":"grid", "gap":"10px"})))
    
    current_utc_time = datetime.now(timezone.utc)
    print(f"[DEBUG] UTC time for status banner: {current_utc_time}")
    main = html.Div([
        html.Div(f"Completed at {current_utc_time.strftime('%H:%M:%S')} UTC", className="status-banner"),
        dcc.Tabs(id="research-results-tabs", value="overview", children=tabs)
    ])
    
    log_event("research_success", {"symbol": symbol, "duration_sec": getattr(t, 'elapsed', None)})

    # --- History entry creation ---
    try:
        from datetime import datetime as _dt
        hist_entry = {
            "ts": _dt.now(timezone.utc).isoformat(),
            "symbol": symbol.upper(),
            "analysis_type": analysis_type,
            "recommendation": (rec or {}).get("recommendation"),
            "rec_confidence": (rec or {}).get("confidence"),
            "summary": strip_markdown_fences(holistic_text or (rec or {}).get("justification") or fundamental_summary_text or ""),
        }
    except Exception:
        hist_entry = {"ts": datetime.now(timezone.utc).isoformat(), "symbol": symbol.upper(), "analysis_type": analysis_type, "summary": "(unable to build summary)"}

    current_history = existing_history or []
    # Prepend new entry; enforce max HISTORY_LIMIT
    try:
        _limit_tail = max(HISTORY_LIMIT - 1, 0)
    except Exception:
        _limit_tail = 24  # fallback if constant missing
    new_history = [hist_entry] + [h for h in current_history if isinstance(h, dict)][: _limit_tail]

    # Persist to disk (best-effort)
    try:
        import json as _json, os as _os
        hist_path = _history_file()
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(hist_path, 'w', encoding='utf-8') as _f:
            _json.dump(new_history, _f, indent=2)
    except Exception as _e:  # non-fatal
        print(f"[HISTORY] Failed to persist to {str(hist_path)}: {_e}")

    return main, "Done", new_history

# Expansion toggle for history items (now using clickable cards)
@app.callback(
    Output("history-selected", "data"),
    Input({"type":"history-card","index": ALL}, "n_clicks"),
    State("history-selected", "data"),
    prevent_initial_call=True
)
def toggle_history_expansion(clicks, current):
    if not clicks or not any(clicks):
        raise dash.exceptions.PreventUpdate
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trig = ctx.triggered_id
    if isinstance(trig, dict) and 'index' in trig:
        idx = trig['index']
        # Collapse if same index clicked again
        if current == idx:
            return None
        return idx
    raise dash.exceptions.PreventUpdate

# --- Deletion controls for history ---
@app.callback(
    Output("research-history", "data", allow_duplicate=True),
    Output("history-selected", "data", allow_duplicate=True),
    Input({"type": "history-delete", "index": ALL}, "n_clicks"),
    State("research-history", "data"),
    State("history-selected", "data"),
    prevent_initial_call=True,
)
def delete_history_item(delete_clicks, history, selected_idx):
    if not delete_clicks or not any(delete_clicks):
        raise dash.exceptions.PreventUpdate
    try:
        ctx = dash.callback_context
        trig = ctx.triggered_id
        if isinstance(trig, dict) and "index" in trig:
            idx = trig["index"]
        else:
            raise dash.exceptions.PreventUpdate
        new_hist = []
        for i, item in enumerate(history or []):
            if i == idx:
                continue
            new_hist.append(item)
        # Persist
        try:
            import json as _json
            p = _history_file()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                _json.dump(new_hist[:HISTORY_LIMIT], f, indent=2)
        except Exception as e:
            print(f"[HISTORY] Delete persist failed: {e}")
        # Adjust selected index if needed
        new_selected = None
        if isinstance(selected_idx, int):
            if selected_idx == idx:
                new_selected = None
            elif selected_idx > idx:
                new_selected = selected_idx - 1
            else:
                new_selected = selected_idx
        return new_hist[:HISTORY_LIMIT], new_selected
    except Exception:
        raise dash.exceptions.PreventUpdate

@app.callback(
    Output("research-history", "data", allow_duplicate=True),
    Input("history-clear-all", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all_history(n):
    if not n:
        raise dash.exceptions.PreventUpdate
    try:
        import json as _json
        p = _history_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            _json.dump([], f)
    except Exception as e:
        print(f"[HISTORY] Clear-all persist failed: {e}")
    return []

# --- Local dev entrypoint ---
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    try:
        port = int(os.getenv("PORT", os.getenv("DASH_PORT", "8050") or 8050))
    except Exception:
        port = 8050
    debug_env = str(os.getenv("DEBUG", "0")).strip().lower()
    debug = debug_env in ("1", "true", "yes", "on")

    print(f"Starting Dash app on http://{host}:{port} (debug={debug})")
    app.run_server(host=host, port=port, debug=debug)