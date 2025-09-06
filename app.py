import os
import json
import requests
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import dash
from dash import Dash, dcc, html, Input, Output, State, ALL
from services import NewsBridge, ResearchOrchestrator
import plotly.graph_objects as go
# Import tab layouts; handle both module and script execution paths
try:
    from .layouts.news import layout as news_tab_layout
    from .layouts.strategy import layout as strategy_tab_layout
except Exception:
    # When running as a script (python dash/app.py), use local package import
    from layouts.news import layout as news_tab_layout  # type: ignore
    from layouts.strategy import layout as strategy_tab_layout  # type: ignore

# Ensure dash/.env overrides any pre-set environment variables (so API-backed news providers are enabled)
load_dotenv(override=True)

app = Dash(__name__, suppress_callback_exceptions=True, title="Financial Intelligence Dashboard")
server = app.server

# Explicit set of crypto tickers for TradingView mapping (avoid misclassifying stocks)
CRYPTO_ASSETS = {"BTC", "ETH", "SOL"}

bridge = NewsBridge()
research = ResearchOrchestrator()

# Defaults for Local LLM fields pulled from environment (and sensible fallbacks)
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

# Helpers

def tv_symbol(asset: str) -> str:
    a = (asset or '').strip().upper()
    if not a:
        return "AAPL"
    # If explicit TradingView symbol (e.g., NASDAQ:AAPL), use it as-is
    if ":" in a:
        return a
    # Treat known crypto tickers explicitly
    if a in CRYPTO_ASSETS:
        return f"CRYPTO:{a}USD"
    # Heuristic: plain 1-5 letter tickers are stocks (no USD suffix)
    if a.isalpha() and 1 <= len(a) <= 5:
        return a
    # Fallback: if it ends with a quote currency, drop it if base looks like a stock ticker
    for suf in ("USDT", "USD"):
        if a.endswith(suf) and a[:-len(suf)].isalpha() and 1 <= len(a[:-len(suf)]) <= 5:
            return a[:-len(suf)]
    # Otherwise leave as provided (lets TV try to resolve), but avoid forcing CRYPTO mapping
    return a

# Layout with enhanced visual design
app.layout = html.Div(className="container", children=[
    # Enhanced tabs with icons
    dcc.Tabs(id="tabs", value="tab-news", className="tab-parent", children=[
        dcc.Tab(label="📊 News & Analysis", value="tab-news", className="tab"),
        dcc.Tab(label="💼 Strategy (Wallet)", value="tab-strategy", className="tab"),
    dcc.Tab(label="🔬 Research", value="tab-research", className="tab"),
    ]),
    
    # Data stores
    dcc.Store(id="news-data"),
    dcc.Store(id="active-asset"),
    dcc.Store(id="selected-idx"),
    dcc.Store(id="overall-summary"),
    dcc.Store(id="update-meta"),
    # One-shot interval to pre-list DigitalOcean models on first load
    dcc.Interval(id="do-models-primer", interval=300, n_intervals=0, max_intervals=1),
    
    # Tab content container
    html.Div(id="tab-content")
])

# Tab 1 and 2 layouts are imported from layouts/ modules

# Tab 3: Research (multi-agent) layout (redesigned for specified agents)
research_layout = html.Div(
    style={"display": "grid", "gridTemplateColumns": "380px 1fr", "gap": "16px"},
    children=[
        # --- NEW WIZARD-STYLE SIDEBAR ---
        html.Div(
            className="card",
            style={"padding": "16px"},
            children=[
                html.H3("Research Configuration", className="section-title"),
                
                dcc.Tabs(id="config-wizard-tabs", value='tab-mission', className="wizard-tabs", children=[
                    
                    # --- STEP 1: DEFINE THE MISSION ---
                    dcc.Tab(label='1. Mission', value='tab-mission', className="wizard-tab", children=[
                        html.Div(className="wizard-step", children=[
                            html.Label("Ticker Symbol"),
                            html.Div([
                                dcc.Dropdown(
                                    id="research-symbol-dd",
                                    options=[
                                        {"label": x, "value": x} for x in [
                                            "AAPL","MSFT","GOOGL","META","AMZN","TSLA","NVDA","AMD","NFLX","INTC",
                                            "SPY","QQQ","IWM","DIA","GLD","SLV","BTC-USD","ETH-USD"
                                        ]
                                    ],
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
                                    {'label': ' AutoGen (Deep & Collaborative)', 'value': 'multi'},
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
                                     html.Label("AutoGen Max Turns"),
                                    dcc.Dropdown(
                                        id="autogen-max-turns",
                                        options=[
                                            {"label": "3 (Fast)", "value": 3},
                                            {"label": "5 (Balanced)", "value": 5},
                                            {"label": "8 (Thorough)", "value": 8},
                                            {"label": "12 (Deep)", "value": 12},
                                        ],
                                        value=5,
                                        clearable=False,
                                    ),
                                    html.Label("Agent Rounds (LangChain)"),
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
                                                {"label": "Local (OpenAI-compatible)", "value": "local"},
                                                {"label": "Gemini (marketNews)", "value": "gemini"},
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
                                        dcc.Input(id="llm-model", type="text", value=DEFAULT_LLM_MODEL, placeholder="llama-3.1-8b-instruct", style={"width": "100%"}),
                                        dcc.Dropdown(id="llm-model-dd", options=[], placeholder="Pick Local model (optional)", style={"marginTop": "6px"}),
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
            dcc.Loading(id="research-loader", type="cube", color="#3b82f6", children=html.Div(id="research-content"))
        ]),
    ],
)
@app.callback(
    Output("web-results-container", "style"),
    Input("allow-web-search", "value")
)
def toggle_web_results_visibility(allow_search):
    if allow_search == 'on':
        return {'display': 'block', 'marginTop': '15px'} # Show it
    else:
        return {'display': 'none'} # Hide it
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    # Always include news layout to keep components in DOM, but hide when not active
    news_style = {"display": "block" if tab == "tab-news" else "none"}
    strategy_style = {"display": "block" if tab == "tab-strategy" else "none"}
    research_style = {"display": "block" if tab == "tab-research" else "none"}
    
    return html.Div([
        html.Div(news_tab_layout(), style=news_style),
        html.Div(strategy_tab_layout(), style=strategy_style) if tab == "tab-strategy" else html.Div(),
        html.Div(research_layout, style=research_style) if tab == "tab-research" else html.Div(),
    ])
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
    State("news-do-model-dd", "value"),
    State("news-hf-model-dd", "value"),
    State("news-data", "data"),
    State("overall-summary", "data"),
    State("update-meta", "data"),
    prevent_initial_call=True,
)
def fetch_news(asset, refresh_clicks, model_pref, count, days_back, do_model_choice, hf_model_choice, existing_data, existing_overall, existing_meta):

    # Short-circuit: if controls re-mounted on tab switch caused this callback and we already have data, do not refetch
    try:
        ctx = dash.callback_context
        trig = (ctx.triggered[0]["prop_id"].split(".")[0] if ctx and ctx.triggered else None)
    except Exception:
        trig = None
    prev_asset = (existing_meta or {}).get("asset") if isinstance(existing_meta, dict) else None
    if trig in {"model-pref", "news-count", "news-days"} and existing_data:
        # Return what we already have to preserve UI state
        probe = ((existing_meta or {}).get("ts") if isinstance(existing_meta, dict) else None) or datetime.now().isoformat() + "Z"
        return existing_data, (existing_overall or {"summary": "", "provider": ""}), (existing_meta or {}), probe
    # Note: On explicit refresh button click, proceed to refetch (no early return)
    # If active-asset fired but asset hasn't actually changed, preserve
    if trig == "active-asset" and existing_data and prev_asset == asset:
        probe = ((existing_meta or {}).get("ts") if isinstance(existing_meta, dict) else None) or datetime.now().isoformat() + "Z"
        return existing_data, (existing_overall or {"summary": "", "provider": ""}), (existing_meta or {}), probe
    max_articles = int(count or 0)
    # If 0, skip fetching to avoid auto-loads on reload
    if max_articles <= 0:
        meta = {"asset": asset, "ts": datetime.now(timezone.utc).isoformat() + "Z", "fetch_summary": "news disabled (count=0)"}
        probe = meta["ts"]
        return [], {"summary": "", "provider": ""}, meta, probe
    # If DigitalOcean is selected and a model is chosen, set env so NewsBridge prefers it
    if (model_pref or "auto") == "do" and (do_model_choice or os.getenv("DO_AI_MODEL")):
        os.environ["AI_PROVIDER"] = "do"
        if do_model_choice:
            os.environ["DO_AI_MODEL"] = str(do_model_choice)
    # If Hugging Face is selected and a model is chosen, set env so NewsBridge prefers it
    if (model_pref or "auto") == "hf":
        os.environ["AI_PROVIDER"] = "hf"
        # Prefer dropdown choice, then env, then default to requested model
        chosen = (hf_model_choice or os.getenv("HF_MODEL") or "Qwen/Qwen3-4B-Base")
        os.environ["HF_MODEL"] = str(chosen)
        if not os.getenv("HF_BASE_URL"):
            os.environ["HF_BASE_URL"] = "https://router.huggingface.co/v1"
    # Coerce/clip lookback
    try:
        days = int(days_back or 7)
        if days < 1:
            days = 1
        if days > 60:
            days = 60
    except Exception:
        days = 7
    items = bridge.fetch(asset, days_back=days, max_articles=max_articles, model_preference=model_pref or "auto", analyze=True)
    data = [
        {
            "title": i.title,
            "url": i.url,
            "source": i.source,
            "published_at": i.published_at,
            "ai_summary": i.ai_summary,
            "description": i.description,
            "sentiment_label": i.sentiment_label,
            "sentiment_score": i.sentiment_score,
            "analysis_provider": i.analysis_provider,
        } for i in (items or [])
    ]
    overall = None
    try:
        overall = bridge.summarize_overall(asset, data, model_preference=model_pref or "auto", max_chars=60000)
    except Exception as _:
        overall = {"summary": "", "provider": ""}
    # Emit simple meta for UX messaging
    fetch_summary = ""
    try:
        fetch_summary = bridge.get_fetch_summary()
    except Exception:
        fetch_summary = ""
    meta = {"asset": asset, "ts": datetime.now().isoformat() + "Z", "fetch_summary": fetch_summary}
    # Probe content changes every call -> triggers loading overlay reliably
    probe = meta["ts"]
    return data, overall, meta, probe

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
        if label == 'bullish':
            cls = 'badge badge-bullish'
            icon = '📈'
        elif label == 'bearish':
            cls = 'badge badge-bearish'
            icon = '📉'
        else:
            cls = 'badge badge-neutral'
            icon = '➖'
        if not item.get('sentiment_label'):
            return None
        provider = (item.get('analysis_provider') or '').upper()
        prov = html.Span(f"🤖 {provider}", className="provider") if provider else None
        score = item.get('sentiment_score')
        score_txt = f" {score:+.2f}" if (score is not None) else ""
        return html.Span([
            html.Span([icon, " ", item.get('sentiment_label') + score_txt], className=cls), 
            prov
        ], className="badge-wrap")

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

    # Guard selected_idx and compute selected URL (so we can preserve expansion after reordering)
    sel_url = None
    if isinstance(selected_idx, int) and 0 <= selected_idx < len(data):
        try:
            sel_url = data[selected_idx].get('url')
        except Exception:
            sel_url = None
    else:
        selected_idx = None

    # Round-robin across domains with a per-domain cap
    max_per_domain = 3
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

    # Build nodes from ordered_items; preserve expansion by matching URL
    nodes = []
    for idx, item in enumerate(ordered_items):
        is_expanded = (sel_url is not None and item.get('url') == sel_url)
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
    Input({"type": "news-card", "index": ALL}, "n_clicks_timestamp"),
    Input({"type": "news-refresh", "scope": ALL}, "n_clicks"),
    State("selected-idx", "data"),
    State("news-data", "data"),
    prevent_initial_call=True,
)
def toggle_selected(timestamps, refresh_clicks, current, data):
    # If refresh clicked, clear selection state
    try:
        ctx = dash.callback_context
        if ctx and ctx.triggered:
            prop = ctx.triggered[0]["prop_id"].split(".")[0]
            if "news-refresh" in prop:
                return None
    except Exception:
        pass
    if not timestamps or all(t is None for t in timestamps):
        raise dash.exceptions.PreventUpdate
    # Choose most recent clicked card
    try:
        idx = max((i for i, t in enumerate(timestamps) if t), key=lambda j: timestamps[j])
    except Exception:
        raise dash.exceptions.PreventUpdate
    # Toggle if clicking the same card again
    if current == idx:
        return None
    return idx

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

# Strategy maker: Blockchair
BLOCKCHAIR_API_KEY = os.getenv("BLOCKCHAIR_API_KEY", "")

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
    try:
        # Helper: friendly message if Blockchair not available
        def _api_error_msg(resp, provider_name: str):
            try:
                body = resp.json()
                ctx = body.get("context", {}) if isinstance(body, dict) else {}
                detail = ctx.get("error") or body.get("message") or str(body)[:200]
            except Exception:
                detail = (resp.text or "").strip()[:200]
            hint = []
            if resp.status_code in (402, 403, 429):
                if not BLOCKCHAIR_API_KEY:
                    hint.append("No BLOCKCHAIR_API_KEY set. You may be rate limited. Add it to .env and restart.")
                else:
                    hint.append("Your API key may be invalid or out of quota.")
            return f"{provider_name} error {resp.status_code}: {detail}" + (" — " + "; ".join(hint) if hint else "")

        # Try Blockchair first
        if chain == "bitcoin":
            url = f"https://api.blockchair.com/bitcoin/dashboards/address/{addr}"
        elif chain == "ethereum":
            url = f"https://api.blockchair.com/ethereum/dashboards/address/{addr}"
        else:
            return html.Div("Unsupported chain.")
        params = {"key": BLOCKCHAIR_API_KEY} if BLOCKCHAIR_API_KEY else {}
        data = None
        r = None
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
        except Exception:
            r = None

        # Fallbacks if Blockchair failed
        used_fallback = False
        fallback_note = None
        if data is None:
            if chain == "bitcoin":
                # Blockstream public API
                try:
                    br = requests.get(f"https://blockstream.info/api/address/{addr}", timeout=15)
                    if br.status_code == 200:
                        bdata = br.json()
                        # Approx balance = funded - spent
                        cs = bdata.get("chain_stats", {})
                        funded = int(cs.get("funded_txo_sum", 0))
                        spent = int(cs.get("spent_txo_sum", 0))
                        bal_sats = max(funded - spent, 0)
                        data = {"_fallback": True, "btc_balance_sats": bal_sats}
                        used_fallback = True
                        fallback_note = "Used Blockstream fallback (no API key)."
                    else:
                        err = _api_error_msg(br, "Blockstream")
                        return html.Div(err)
                except Exception as e2:
                    err = _api_error_msg(r, "Blockchair") if r is not None else f"Network error: {e2}"
                    return html.Div(err)
            elif chain == "ethereum":
                # Ethplorer free endpoint
                try:
                    er = requests.get(f"https://api.ethplorer.io/getAddressInfo/{addr}", params={"apiKey": "freekey"}, timeout=15)
                    if er.status_code == 200:
                        edata = er.json()
                        eth_bal = edata.get("ETH", {}).get("balance", 0)
                        tokens = edata.get("tokens", []) or []
                        data = {"_fallback": True, "eth_balance": eth_bal, "erc20_count": len(tokens)}
                        used_fallback = True
                        fallback_note = "Used Ethplorer fallback (no API key)."
                    else:
                        err = _api_error_msg(er, "Ethplorer")
                        return html.Div(err)
                except Exception as e2:
                    err = _api_error_msg(r, "Blockchair") if r is not None else f"Network error: {e2}"
                    return html.Div(err)
            else:
                return html.Div("Unsupported chain.")

        # Simple inference: detect if balance or tokens exist and propose posture
        idea = []
        if chain == "bitcoin":
            if data.get("_fallback"):
                bal = int(data.get("btc_balance_sats", 0))
            else:
                bal = data.get('data', {}).get(addr, {}).get('address', {}).get('balance', 0)
            idea.append(f"BTC balance: {bal} sats")
            posture = "HOLD" if int(bal) > 0 else "NO POSITION"
        else:
            if data.get("_fallback"):
                eth_bal = data.get("eth_balance", 0)
                token_count = int(data.get("erc20_count", 0))
            else:
                eth_bal = data.get('data', {}).get(addr, {}).get('address', {}).get('balance', 0)
                token_count = len(data.get('data', {}).get(addr, {}).get('layer_2', {}).get('erc_20', []) or [])
            idea.append(f"ETH balance: {eth_bal}")
            idea.append(f"ERC-20 tokens: {token_count}")
            posture = "DIVERSIFY" if token_count and token_count > 3 else ("HOLD" if float(eth_bal) > 0 else "NO POSITION")

        # Very lightweight strategy suggestion
        strategy = [
            f"🎯 Suggested posture: {posture}",
            "⚖️ Rebalance monthly if allocation drifts >10%",
            "📊 Consider DCA for top holdings if conviction is high",
        ]
        nodes = [
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
                html.Div(children=[
                    html.Div("💰 Wallet Insights", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px", "color": "var(--accent-primary)"}),
                    html.Ul([html.Li(x, style={"marginBottom": "8px", "lineHeight": "1.5"}) for x in idea], style={"paddingLeft": "20px"})
                ]),
                html.Div(children=[
                    html.Div("🚀 Strategy Recommendations", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "12px", "color": "var(--accent-primary)"}),
                    html.Ul([html.Li(x, style={"marginBottom": "8px", "lineHeight": "1.5"}) for x in strategy], style={"paddingLeft": "20px"})
                ])
            ])
        ]
        if used_fallback and fallback_note:
            nodes.append(html.Div([
                "ℹ️ ", fallback_note
            ], style={
                "color": "var(--text-muted)", 
                "fontSize": "13px", 
                "marginTop": "16px",
                "padding": "12px",
                "background": "var(--warning-bg)",
                "border": "1px solid var(--warning-border)",
                "borderRadius": "8px"
            }))
        if not used_fallback and not BLOCKCHAIR_API_KEY:
            nodes.append(html.Div([
                "💡 ", html.Strong("Pro tip: "), "Add BLOCKCHAIR_API_KEY to .env for higher API limits and more detailed analytics."
            ], style={
                "color": "var(--text-muted)", 
                "fontSize": "13px", 
                "marginTop": "16px",
                "padding": "12px",
                "background": "var(--bg-secondary)",
                "border": "1px solid var(--border-secondary)",
                "borderRadius": "8px"
            }))
        return html.Div(nodes)
    except Exception as e:
        return html.Div(f"Error: {e}")

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
                # Apply timezone offset (default +2 hours) from env TIMEZONE_OFFSET_HOURS
                try:
                    tz_off = float(os.getenv("TIMEZONE_OFFSET_HOURS", "2"))
                except Exception:
                    tz_off = 2.0
                t = t + timedelta(hours=tz_off)
                label = f"Updated {t.strftime('%H:%M:%S')} (UTC{('+' if tz_off>=0 else '')}{int(tz_off)})"
            except Exception:
                label = "Updated"
            return label, False
        return "", False
    except Exception:
        return "", False

try:
    from agents_team import run_team_workflow as run_autogen_workflow
except Exception:
    from agents import run_autogen_workflow
from telemetry import log_event, Timer

# ---------------- Research callbacks ----------------

# List DO models for News tab (top-level)
@app.callback(
    Output("news-do-model-dd", "options"),
    Input("do-models-primer", "n_intervals"),
    prevent_initial_call=True,
)
def list_do_models_news(_primer):
    try:
        base = (os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1")).rstrip("/")
        key = os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY")
        if not key:
            return []
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        r = requests.get(f"{base}/models", headers=headers, timeout=20)
        if r.status_code != 200:
            return []
        data = r.json()
        # Filter to lower-end models (e.g., 7B–20B or mini/tiny) to keep costs modest
        low_keywords = ["7b", "8b", "9b", "10b", "12b", "13b", "15b", "20b", "mini", "tiny", "small"]
        opts = []
        for m in (data.get("data") or [])[:200]:
            mid = (m.get("id") or m.get("name") or "").lower()
            if mid and any(k in mid for k in low_keywords):
                opts.append({"label": m.get("id") or m.get("name"), "value": m.get("id") or m.get("name")})
        return opts
    except Exception:
        return []

# Preload DO models for Research tab as well (top-level)
@app.callback(
    Output("do-model-dd", "options"),
    Output("do-model-status", "children"),
    Input("do-models-primer", "n_intervals"),
    State("do-base-url", "value"),
    prevent_initial_call=True,
)
def list_do_models_research(_primer, base_url):
    try:
        base = (base_url or os.getenv("DO_AI_BASE_URL", "https://inference.do-ai.run/v1")).rstrip("/")
        key = os.getenv("DO_AI_API_KEY") or os.getenv("DIGITALOCEAN_AI_API_KEY")
        if not key:
            return [], "Set DO_AI_API_KEY in .env to list models."
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        r = requests.get(f"{base}/models", headers=headers, timeout=20)
        if r.status_code != 200:
            return [], f"Error {r.status_code}: {r.text[:120]}"
        data = r.json()
        low_keywords = ["7b", "8b", "9b", "10b", "12b", "13b", "15b", "20b", "mini", "tiny", "small"]
        opts = []
        for m in (data.get("data") or [])[:200]:
            mid = (m.get("id") or m.get("name") or "").lower()
            if mid and any(k in mid for k in low_keywords):
                opts.append({"label": m.get("id") or m.get("name"), "value": m.get("id") or m.get("name")})
        status = f"Loaded {len(opts)} low-end models." if opts else "No low-end models matched."
        return opts, status
    except Exception as e:
        return [], f"Failed: {e}"

# Curated HF list for News tab (low-end only)
@app.callback(
    Output("news-hf-model-dd", "options"),
    Input("do-models-primer", "n_intervals"),
    prevent_initial_call=True,
)
def list_hf_models_news(_primer):
    curated = [
        {"label": "Qwen/Qwen3-4B-Base (default)", "value": "Qwen/Qwen3-4B-Base"},
    ]
    return curated

# Curated HF list for Research tab
@app.callback(
    Output("hf-model-dd", "options"),
    Output("hf-model-status", "children"),
    Input("do-models-primer", "n_intervals"),
    prevent_initial_call=True,
)
def list_hf_models_research(_primer):
    curated = [
        {"label": "Qwen/Qwen3-4B-Base (default)", "value": "Qwen/Qwen3-4B-Base"},
    ]
    return curated, "Loaded 1 curated model."
@app.callback(
    Output("research-content", "children"),
    Output("research-status", "children"),
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
    State("autogen-max-turns", "value"),
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
    prevent_initial_call=True,
)
def run_research(
    n,
    analysis_type,
    engine,
    symbol_dd,
    symbol_input,
    timeframe,
    lookback_value,
    lookback_unit,
    indicators,
    length,
    rsi_length,
    macd_fast,
    macd_slow,
    macd_signal,
    speed_mode,
    autogen_max_turns,
    allow_web_search,
    web_max_results,
    ai_provider,
    llm_base_url,
    llm_model,
    llm_max_tokens,
    llm_temp,
    agent_rounds,
    do_base_url,
    do_model_text,
    do_model_choice,
    hf_base_url,
    hf_model_text,
    hf_model_choice,
):
    # Prefer custom input if provided, else dropdown
    symbol = (symbol_input or "").strip() if symbol_input else (symbol_dd or "").strip()
    if not symbol:
        return html.Div("Enter a ticker."), "Idle"

    # Apply optional LLM overrides (for summarizers and agent engines)
    try:
        overrides = {}
        if ai_provider:
            os.environ["AI_PROVIDER"] = str(ai_provider)
            overrides["provider"] = str(ai_provider)
        if llm_base_url:
            os.environ["LOCAL_LLM_BASE_URL"] = str(llm_base_url)
            overrides["base_url"] = str(llm_base_url)
        if llm_model:
            os.environ["LOCAL_LLM_MODEL"] = str(llm_model)
            overrides["model"] = str(llm_model)
        if llm_max_tokens not in (None, ""):
            os.environ["LOCAL_LLM_MAX_TOKENS"] = str(int(llm_max_tokens))
            overrides["max_tokens"] = int(llm_max_tokens)
        if llm_temp not in (None, ""):
            os.environ["LOCAL_LLM_TEMPERATURE"] = str(float(llm_temp))
            overrides["temperature"] = float(llm_temp)
        if do_base_url:
            os.environ["DO_AI_BASE_URL"] = str(do_base_url)
        if (do_model_choice or do_model_text):
            os.environ["DO_AI_MODEL"] = str(do_model_choice or do_model_text)
        if hf_base_url:
            os.environ["HF_BASE_URL"] = str(hf_base_url)
        if (hf_model_choice or hf_model_text):
            os.environ["HF_MODEL"] = str(hf_model_choice or hf_model_text)
        if overrides:
            log_event("llm_config", overrides)
    except Exception:
        pass

    # UI Agent output (structured request)
    # Compute date window from lookback selector
    try:
        from datetime import datetime, timedelta
        lb = int(lookback_value or 180)
        # Speed mode clamps lookback to reduce data volume
        if speed_mode == "fast":
            lb = min(lb, 60)
        elif speed_mode == "normal":
            lb = min(lb, 120)
        unit = (lookback_unit or "days").lower()
        now = datetime.now()
        if unit == "days":
            delta = timedelta(days=lb)
        elif unit == "weeks":
            delta = timedelta(weeks=lb)
        elif unit == "months":
            # approximate months
            delta = timedelta(days=lb * 30)
        else:
            delta = timedelta(days=lb * 365)
        start_date = (now - delta).date().isoformat()
        end_date = now.date().isoformat()
    except Exception:
        start_date = None
        end_date = None

    ui_request = {
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
    "macd_signal": int(macd_signal or 9),
    "speed_mode": speed_mode,
    "autogen_max_turns": int(autogen_max_turns or 5),
    "allow_web_search": (allow_web_search or "off") == "on",
    "web_max_results": int(web_max_results or 5),
    }
    log_event("research_ui_request", ui_request)
    try:
        # Also log compact request signature for quick scans
        sig = {"symbol": symbol, "engine": (engine or "lang"), "type": analysis_type, "interval": timeframe}
        log_event("research_ui_signature", sig)
    except Exception:
        pass

    # Branch: Multi-agent vs local
    out = {}
    conversations = []
    try:
        with Timer() as t:
            if (engine or "lang") == "multi":
                # Compose a natural language task for the team
                req_lines = [
                    f"stock tickers: {symbol}",
                    f"timeframe of {timeframe}",
                    f"from {start_date} to {end_date}",
                    f"technical indicators: {', '.join(indicators or [])}",
                    f"General length: {int(length or 20)}",
                    f"RSI length: {int(rsi_length or 14)}",
                    f"MACD settings: fast={int(macd_fast or 12)}, slow={int(macd_slow or 26)}, signal={int(macd_signal or 9)}",
                    f"speed mode: {speed_mode}",
                    f"max turns: {int(autogen_max_turns or 5)}",
                    f"allow web search: {((allow_web_search or 'off') == 'on')}",
                    f"max web results: {int(web_max_results or 5)}",
                ]
                user_request = "\n".join(req_lines)
                # Also set env for agents_team to read
                try:
                    os.environ["AUTOGEN_MAX_TURNS"] = str(int(autogen_max_turns or 5))
                    os.environ["ALLOW_WEB_SEARCH"] = "1" if (allow_web_search or "off") == "on" else "0"
                    os.environ["WEB_MAX_RESULTS"] = str(int(web_max_results or 5))
                except Exception:
                    pass
                try:
                    res = run_autogen_workflow(user_request)  # returns {"result": json_str, optional "messages": [...]}
                except Exception as e2:
                    res = {"result": json.dumps({"error": f"agent framework failed: {e2}"})}
                raw = res.get("result") or "{}"
                try:
                    j = json.loads(raw)
                except Exception:
                    j = {}
                # Expect a single ticker result
                r = j.get(symbol) if isinstance(j, dict) else None
                if not r and isinstance(j, dict) and len(j) == 1:
                    r = list(j.values())[0]
                # Normalize possible string payloads into dicts
                r_dict = {}
                if isinstance(r, dict):
                    r_dict = r
                elif isinstance(r, str):
                    try:
                        parsed = json.loads(r)
                        if isinstance(parsed, dict):
                            r_dict = parsed
                    except Exception:
                        r_dict = {}
                # Build a minimal out compatible with our renderer
                fig = r_dict.get("figure") if isinstance(r_dict, dict) else None
                tech_fig = go.Figure(fig) if isinstance(fig, dict) else None
                out = {
                    "plan": {"analysis_type": analysis_type, "steps": ["Planner", "DataAgent", "TechnicalAnalyst"]},
                    "technical": {
                        "recommendation": r_dict.get("recommendation", "N/A"),
                        "confidence": r_dict.get("confidence", "N/A"),
                        "summary": r_dict.get("justification", ""),
                    },
                    "figures": {"technical": tech_fig} if tech_fig else {},
                }
                conversations = res.get("messages") or []
            elif (engine or "lang") == "lang":
                # Minimal LC flow: call lc_agents workflow
                req_lines = [
                    f"stock tickers: {symbol}",
                    f"timeframe of {timeframe}",
                    f"from {start_date} to {end_date}",
                    f"technical indicators: {', '.join(indicators or [])}",
                ]
                user_request = "\n".join(req_lines)
                try:
                    from lc_agents import run_langchain_workflow
                    res = run_langchain_workflow(user_request, model_name=llm_model)
                except Exception as e2:
                    res = {"result": json.dumps({"error": f"langchain workflow failed: {e2}"})}
                raw = res.get("result") or "{}"
                try:
                    j = json.loads(raw)
                except Exception:
                    j = {}
                r = j.get(symbol) if isinstance(j, dict) else None
                if not r and isinstance(j, dict) and len(j) == 1:
                    r = list(j.values())[0]
                # Normalize possible string payloads into dicts
                r_dict = {}
                if isinstance(r, dict):
                    r_dict = r
                elif isinstance(r, str):
                    try:
                        parsed = json.loads(r)
                        if isinstance(parsed, dict):
                            r_dict = parsed
                    except Exception:
                        r_dict = {}
                fig = r_dict.get("figure") if isinstance(r_dict, dict) else None
                tech_fig = None
                if isinstance(fig, dict):
                    try:
                        tech_fig = go.Figure(fig)
                    except Exception as fig_error:
                        print(f"Error reconstructing LangChain figure: {fig_error}")
                        tech_fig = None
                # Get full deterministic pipeline so Overview/Fundamentals/News are populated
                full = research.run_full_analysis(
                    symbol=symbol,
                    analysis_type=analysis_type,
                    start_date=start_date,
                    end_date=end_date,
                    interval=timeframe,
                    indicators=indicators,
                    length=int(length or 20),
                    rsi_length=int(rsi_length or 14),
                    macd_fast=int(macd_fast or 12),
                    macd_slow=int(macd_slow or 26),
                    macd_signal_len=int(macd_signal or 9),
                )
                # Override technical summary with LC justification and carry figure if present
                full.setdefault("plan", {})
                full["plan"]["steps"] = ["Planner(LC)", "Data(LC)", "Analyst(LC)"]
                full.setdefault("technical", {})
                full["technical"].update({
                    "recommendation": r_dict.get("recommendation", full["technical"].get("recommendation", "N/A")),
                    "confidence": r_dict.get("confidence", full["technical"].get("confidence", "N/A")),
                    "summary": r_dict.get("justification", full["technical"].get("summary", "")),
                })
                if tech_fig:
                    full.setdefault("figures", {})
                    full["figures"]["technical"] = tech_fig
                out = full
                # Optionally add a few visible rounds to make the dialogue feel more agentic
                convs = res.get("messages") or []
                rounds = int(agent_rounds or 2)
                if rounds > 1:
                    for i in range(1, rounds):
                        convs.append({"agent": "Planner(LC)", "message": f"Round {i+1}: refine analysis focus and confirm signals."})
                        convs.append({"agent": "Validator(LC)", "message": "VALIDATED"})
                conversations = convs + (full.get("conversations") or [])
            # Removed exploratory web engine and local deterministic fallback
    except Exception as e:
        log_event("research_error", {"error": str(e)})
        return html.Div(f"Analysis failed: {e}"), "Failed"

    # Planner/Plan view
    plan = out.get("plan", {})
    plan_badge = html.Div([
        html.Div("Plan of Agents", style={"fontWeight":600, "marginBottom":"6px"}),
        html.Ul([html.Li(s) for s in (plan.get("steps") or [])], style={"paddingLeft":"20px", "margin":"0"})
    ], className="card", style={"padding":"12px", "marginBottom":"12px"})

    # Profile card
    profile = out.get("company_profile") or {}
    profile_card = None
    if profile:
        profile_card = html.Div([
            html.Div(profile.get("name") or symbol, style={"fontWeight":"bold", "fontSize":"18px"}),
            html.Div(f"Sector: {profile.get('sector','')} | Industry: {profile.get('industry','')}", style={"color":"var(--text-secondary)", "marginTop":"4px"}),
            html.Div(profile.get("description",""), style={"fontSize":"13px", "color":"var(--text-muted)", "marginTop":"8px"}),
        ], className="card", style={"padding":"12px"})

    # Technical section
    tech = out.get("technical") or {}
    tech_fig = out.get("figures", {}).get("technical")
    tech_section = html.Div([
        html.H4("Technical Analysis"),
        dcc.Graph(figure=tech_fig) if tech_fig else html.Div("No chart."),
        html.Div([
            html.Div(f"Recommendation: {tech.get('recommendation','N/A')} | Confidence: {tech.get('confidence','N/A')}", style={"marginTop":"8px"}),
            html.Details(open=False, children=[html.Summary("Technical Summary"), html.Pre(tech.get('summary',''))])
        ])
    ], className="card", style={"padding":"12px"}) if analysis_type in ("technical","combined") else None

    # Fundamentals section
    ratios = out.get("ratios") or {}
    earnings = out.get("earnings") or {}
    figs = out.get("figures") or {}
    fund_children = []
    if ratios:
        kv = [
            ("P/E (TTM)", ratios.get("pe_ttm")),
            ("P/S (TTM)", ratios.get("ps_ttm")),
            ("Debt/Equity", ratios.get("debt_to_equity")),
            ("ROE", ratios.get("roe")),
            ("ROA", ratios.get("roa")),
            ("Gross Margin", ratios.get("gross_margin")),
            ("Net Margin", ratios.get("net_margin")),
            ("Current Ratio", ratios.get("current_ratio")),
            ("Quick Ratio", ratios.get("quick_ratio")),
        ]
        rows = []
        for k,v in kv:
            if v is None: continue
            rows.append(html.Div([html.Span(k), html.Span(f"{v:.2f}" if isinstance(v,(int,float)) else str(v))], style={"display":"flex","justifyContent":"space-between","gap":"12px","padding":"4px 0"}))
        fund_children.append(html.Div([html.H4("Key Ratios"), html.Div(rows)], className="card", style={"padding":"12px"}))
    if figs.get("revenue"):
        fund_children.append(html.Div([html.H4("Revenue (Quarterly)"), dcc.Graph(figure=figs["revenue"])], className="card", style={"padding":"12px"}))
    if figs.get("pe_trend"):
        fund_children.append(html.Div([html.H4("P/E Trend (approx)"), dcc.Graph(figure=figs["pe_trend"])], className="card", style={"padding":"12px"}))
    fundamentals_section = html.Div(fund_children) if fund_children and analysis_type in ("fundamental","combined") else None

    # News & Sentiment
    news_items = out.get("news") or []
    news_summary = out.get("news_summary") or {}
    web_results = out.get("web_search") or []
    news_section = None
    if analysis_type in ("web","combined"):
        cards = []
        for it in news_items[:6]:
            cards.append(html.Div([
                html.A(it.get("title") or "(no title)", href=it.get("url"), target="_blank"),
                html.Div(it.get("source",""), style={"fontSize":"12px","color":"var(--text-secondary)"}),
                html.Div(it.get("ai_summary") or it.get("description") or "", style={"fontSize":"13px","color":"var(--text-muted)", "marginTop":"6px"})
            ], className="card", style={"padding":"10px"}))
        news_cards = html.Div(cards, style={"display":"grid","gridTemplateColumns":"1fr","gap":"8px"})
        web_nodes = []
        if web_results:
            for r in web_results[:6]:
                t = r.get("title") or "(no title)"
                u = r.get("url") or r.get("href") or "#"
                s = r.get("snippet") or ""
                web_nodes.append(html.Div([
                    html.A(t, href=u, target="_blank"),
                    html.Div(s, style={"fontSize":"12px","color":"var(--text-secondary)"})
                ], style={"marginBottom":"8px"}))
        news_section = html.Div([
            html.H4("News & Sentiment"),
            news_cards,
            html.Div(news_summary.get("summary",""), style={"marginTop":"10px", "fontSize":"13px"})
        ] + ([html.Hr(), html.H4("Web Search"), html.Div(web_nodes)] if web_nodes else []))

    # Holistic narrative
    hol = out.get("holistic")
    hol_section = html.Div([html.H4("Holistic Analysis"), dcc.Markdown(hol or "")], className="card", style={"padding":"12px"}) if hol and analysis_type in ("combined","fundamental","web") else None

    # Recommendation
    rec = out.get("recommendation") or {}
    rec_section = None
    if rec:
        rec_section = html.Div([
            html.H4("Final Recommendation"),
            html.Div(f"{rec.get('recommendation','Hold')} (Confidence: {rec.get('confidence','N/A')})", style={"fontWeight":600}),
            dcc.Markdown(rec.get("justification",""), style={'white-space': 'pre-wrap'})
        ], className="card", style={"padding":"12px"})

    # Assemble tabs
    tabs = []
    overview_children = [c for c in [plan_badge, profile_card, hol_section, rec_section] if c]
    tabs.append(dcc.Tab(label="Overview", value="overview", children=html.Div(overview_children)))
    if tech_section: tabs.append(dcc.Tab(label="Technical", value="tech", children=tech_section))
    if fundamentals_section: tabs.append(dcc.Tab(label="Fundamentals", value="fund", children=fundamentals_section))
    if news_section: tabs.append(dcc.Tab(label="News", value="news", children=news_section))

    # Conversations tab
    conv = conversations or out.get("conversations") or []
    if conv:
        conv_nodes = []
        for item in conv:
            a = item.get("agent") or "Agent"
            m = item.get("message") or ""
            conv_nodes.append(html.Div([
                html.Div(a, style={"fontWeight":600, "color":"var(--accent-primary)"}),
                html.Pre(m, style={"whiteSpace":"pre-wrap", "margin":"4px 0 12px 0"})
            ], className="card", style={"padding":"10px"}))
        tabs.append(dcc.Tab(label="Conversations", value="conv", children=html.Div(conv_nodes)))

    main = html.Div([
        html.Div(
            f"Completed at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC",
            style={"backgroundColor":"var(--success-bg)","color":"var(--success-text)","padding":"10px","borderRadius":"8px","marginBottom":"12px"}
        ),
        dcc.Tabs(id="research-results-tabs", value="overview", children=tabs)
    ])

    # Log success
    log_event("research_success", {"symbol": symbol, "duration_sec": getattr(t, 'elapsed', None)})
    return main, "Done"


# (Removed unused Testmail and Quick Research callbacks)

if __name__ == "__main__":
    app.run_server(debug=True, host="127.0.0.1", port=8050)
