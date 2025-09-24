from dash import html, dcc

ASSETS = ["BTC", "ETH", "SOL", "NVDA", "GOOGL", "TSLA", "TSMC", "AAPL", "MSFT", "AMZN", "META", "CRCL"]


def layout():
    return html.Div(style={
        "display": "flex",
        "gap": "24px",
        "minHeight": "750px"
    }, children=[
        html.Div(style={"flex": "2 1 0", "minWidth": "0", "display": "flex", "flexDirection": "column"}, children=[
            html.Div(className="toolbar", style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "alignItems": "flex-end", "marginBottom": "8px"}, children=[
                html.Div(className="control", children=[
                    html.Label("🪙 Asset"),
                    dcc.Dropdown(
                        options=[{"label": f"🟡 {a}", "value": a} for a in ASSETS],
                        value="BTC",
                        id="asset-dd",
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        style={"width": "100%"}
                    )
                ]),
                html.Div(className="control", children=[
                    html.Label("⌨️ Custom Symbol"),
                    dcc.Input(
                        id="asset-custom",
                        type="text",
                        placeholder="e.g., NASDAQ:AAPL or BTC",
                        debounce=True,
                        persistence=True,
                        persistence_type="session",
                        style={"width": "100%"}
                    )
                ]),
                html.Div(className="control", children=[
                    html.Label("🤖 AI Model"),
                    dcc.Dropdown(
                        options=[
                            {"label": "🎯 Auto", "value": "auto"},
                            {"label": "💎 Gemini", "value": "gemini"},
                            {"label": "🏠 Local", "value": "local"},
                            {"label": "🧭 DigitalOcean", "value": "do"},
                            {"label": "🧠 Hugging Face", "value": "hf"},
                        ],
                        value="auto",
                        id="model-pref",
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        style={"width": "100%"}
                    )
                ]),
                html.Div(id="do-controls", className="control", children=[
                    dcc.Dropdown(id="news-do-model-dd", options=[], placeholder="Pick a DO model", style={"width": "100%"})
                ]),
                html.Div(id="hf-controls", className="control", children=[
                    dcc.Dropdown(id="news-hf-model-dd", options=[], placeholder="Pick an HF model", style={"width": "100%"})
                ]),
                html.Div(className="control", children=[
                    html.Label("📈 News items (0 disables fetch)"),
                    dcc.Dropdown(
                        id="news-count",
                        options=[{"label": str(n), "value": n} for n in [0, 1, 5, 10, 15, 20, 30]],
                        value=0,
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        style={"width": "100%"}
                    )
                ]),
                html.Div(className="control", children=[
                    html.Label("🗓️ Lookback (days)"),
                    dcc.Dropdown(
                        id="news-days",
                        options=[{"label": str(n), "value": n} for n in [1, 3, 7, 14, 30]],
                        value=7,
                        clearable=False,
                        persistence=True,
                        persistence_type="session",
                        style={"width": "100%"}
                    )
                ]),
                html.Div(className="control", children=[
                    html.Button("🔄 Refresh News", id={"type": "news-refresh", "scope": "news"}, n_clicks=0, className="btn"),
                    html.Div(id="refresh-status", style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "6px"})
                ]),
            ]),
            html.Div(
                id="tv-container",
                className="tv-wrap",
                style={
                    "flex": "1",
                    "borderRadius": "16px",
                    "overflow": "hidden"
                }
            ),
        ]),
        html.Div(id="news-sidebar", style={
            "flex": "1 1 0",
            "minWidth": "320px",
            "maxWidth": "450px",
            "height": "calc(100vh - 100px)",
            "overflowY": "auto",
            "overflowX": "hidden",
            "paddingRight": "8px"
        }, children=[
            html.H3("📰 Current News & AI Analysis", className="section-title", style={"paddingTop": "10px"}),
            dcc.Loading(
                id="news-loader",
                type="cube",
                color="#3b82f6",
                children=[
                    html.Div(id="overall-card"),
                    html.Div(id="news-feed"),
                    # Spacer to push the initial loader lower in the viewport; CSS hides it once content appears
                    html.Div(style={"height": "400px"}, className="initial-spacer"),
                    html.Div(id="news-loading-probe", style={"display": "none"})
                ]
            ),
        ])
    ])
