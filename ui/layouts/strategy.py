import os
import requests
from dotenv import load_dotenv
from datetime import datetime
from dash import Dash, html, dcc, Input, Output, State
import json

# Load the environment variables from the .env file
# Make sure you have a .env file in the same directory with:
# BLOCKCHAIR_API_KEY="your_key_here"
load_dotenv()
API_KEY = os.getenv("BLOCKCHAIR_API_KEY")


# --- 1. DEFINE THE DASH APP LAYOUT ---
# This is the UI structure from your original strategy.py
def layout():
    return html.Div([
        html.Div(className="card", style={"maxWidth": "800px", "margin": "0 auto"}, children=[
            html.Div(style={"textAlign": "center", "marginBottom": "24px"}, children=[
                html.H3("💼 Wallet-based Strategy Maker", style={"margin": "0 0 8px 0"}),
                html.P("Analyze your wallet holdings and get personalized trading strategies",
                      style={"color": "var(--text-secondary)", "margin": "0"})
            ]),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
                html.Div(children=[
                    html.Label("🔗 Blockchain Network", style={"marginBottom": "8px", "display": "block"}),
                    dcc.Dropdown(
                        id="chain-dd",
                        options=[
                            {"label": "⟠ Ethereum", "value": "ethereum"},
                            {"label": "₿ Bitcoin", "value": "bitcoin"},
                        ],
                        value="ethereum", # Default to Ethereum as it supports ERC-20
                        clearable=False
                    ),
                ]),
                html.Div(children=[
                    html.Label("📍 Wallet Address", style={"marginBottom": "8px", "display": "block"}),
                    dcc.Input(
                        id="wallet-input",
                        type="text",
                        placeholder="Enter your wallet address",
                        style={"width": "100%"}
                    ),
                ])
            ]),
            html.Div(style={"textAlign": "center", "marginBottom": "20px"}, children=[
                html.Button("🔍 Analyze Wallet", id="analyze-btn", n_clicks=0, className="btn")
            ]),
            # This Div is where the results will be displayed
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=html.Div(id="strategy-output", style={"marginTop": "20px"})
            )
        ])
    ])

# --- 2. INTEGRATE THE ANALYSIS LOGIC FROM check.py ---
# This function takes user input and returns Dash components
def get_wallet_analysis(blockchain, address):
    """
    Fetches and processes wallet data from Blockchair API,
    then returns the formatted analysis as a list of Dash HTML components.
    """
    if not API_KEY:
        return html.Div("Error: BLOCKCHAIR_API_KEY is not set in your environment.", style={'color': 'red'})
    if not address:
        return html.P("Please enter a wallet address.")

    url = f"https://api.blockchair.com/{blockchain}/dashboards/address/{address}?erc_20=true&limit=100&key={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        return html.Div(f"Error: Failed to fetch data (Status Code: {response.status_code}). Response: {response.text}", style={'color': 'red'})
    data = response.json()
    address_data = data.get('data', {}).get(address.lower(), {})
    if not address_data:
        # Show raw API response for debugging
        return html.Div([
            html.P(f"No data found for this address on the {blockchain} network. Please check the address and selected network."),
            html.Hr(),
            html.H4("Raw API Response (Debug):"),
            html.Pre(json.dumps(data, indent=2))
        ])

    output_components = [
        html.H4("Raw API Response (Debug):"),
        html.Pre(json.dumps(data, indent=2)),
        html.Hr()
    ]
    address_info = address_data.get('address', {})
    eth_balance_wei = address_info.get('balance', '0')
    eth_balance = int(eth_balance_wei) / (10**18)
    tx_count = address_info.get('transaction_count', 0)
    receiving_call_count = address_info.get('receiving_call_count', 0)
    first_seen_receiving = address_info.get('first_seen_receiving')
    last_seen_receiving = address_info.get('last_seen_receiving')
    first_seen_spending = address_info.get('first_seen_spending')
    last_seen_spending = address_info.get('last_seen_spending')
    spent_wei = address_info.get('spent', '0')
    received_wei = address_info.get('received', '0')
    spent_eth = int(spent_wei) / (10**18) if spent_wei else 0
    received_eth = int(received_wei) / (10**18) if received_wei else 0

    # --- BALANCE INFORMATION ---
    output_components.append(html.H4("� BALANCE INFORMATION"))
    output_components.append(html.Ul([
        html.Li(f"Current ETH Balance: {eth_balance:.8f} ETH"),
        html.Li(f"Total ETH Received: {received_eth:.8f} ETH"),
        html.Li(f"Total ETH Spent: {spent_eth:.8f} ETH"),
        html.Li(f"Net ETH Flow: {received_eth - spent_eth:.8f} ETH"),
    ]))

    # --- TRANSACTION STATISTICS ---
    output_components.append(html.H4("📈 TRANSACTION STATISTICS"))
    output_components.append(html.Ul([
        html.Li(f"Total Transactions: {tx_count:,}"),
        html.Li(f"Receiving Calls: {receiving_call_count:,}"),
    ]))

    # --- ACTIVITY TIMELINE ---
    output_components.append(html.H4("⏰ ACTIVITY TIMELINE"))
    timeline_items = []
    if first_seen_receiving:
        try:
            first_receive_date = datetime.fromisoformat(first_seen_receiving.replace('Z', '+00:00'))
            timeline_items.append(html.Li(f"First Received: {first_receive_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"))
        except:
            pass
    if last_seen_receiving:
        try:
            last_receive_date = datetime.fromisoformat(last_seen_receiving.replace('Z', '+00:00'))
            timeline_items.append(html.Li(f"Last Received: {last_receive_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"))
        except:
            pass
    if first_seen_spending:
        try:
            first_spend_date = datetime.fromisoformat(first_seen_spending.replace('Z', '+00:00'))
            timeline_items.append(html.Li(f"First Spent: {first_spend_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"))
        except:
            pass
    if last_seen_spending:
        try:
            last_spend_date = datetime.fromisoformat(last_seen_spending.replace('Z', '+00:00'))
            timeline_items.append(html.Li(f"Last Spent: {last_spend_date.strftime('%Y-%m-%d %H:%M:%S UTC')}"))
        except:
            pass
    if timeline_items:
        output_components.append(html.Ul(timeline_items))

    # --- TOKEN HOLDINGS (ERC-20) ---
    tokens = address_data.get('layer_2', {}).get('erc_20', [])
    # Deduplicate tokens by contract address
    unique_tokens = {}
    for token in tokens:
        token_address = token.get('token_address', 'N/A')
        if token_address not in unique_tokens:
            unique_tokens[token_address] = token
        else:
            existing_token = unique_tokens[token_address]
            current_balance = int(token.get('balance', '0'))
            existing_balance = int(existing_token.get('balance', '0'))
            if current_balance > existing_balance or len(str(token)) > len(str(existing_token)):
                unique_tokens[token_address] = token
    unique_token_list = list(unique_tokens.values())
    output_components.append(html.H4("🪙 TOKEN HOLDINGS (ERC-20)"))
    if not unique_token_list:
        output_components.append(html.P("No ERC-20 tokens found."))
    else:
        output_components.append(html.P(f"Total Unique Tokens: {len(unique_token_list)}"))
        output_components.append(html.P(f"Raw Token Entries: {len(tokens)} (showing duplicates were filtered)"))
        for i, token in enumerate(unique_token_list, 1):
            balance_raw = token.get('balance', '0')
            decimals = token.get('token_decimals', 18)
            balance_decimal = int(balance_raw) / (10**decimals) if balance_raw else 0
            token_name = token.get('token_name', 'Unknown')
            token_symbol = token.get('token_symbol', 'UNKNOWN')
            token_address = token.get('token_address', 'N/A')
            details = [
                html.Li(f"Balance: {balance_decimal:,.6f} {token_symbol}"),
                html.Li(f"Contract: {token_address}"),
                html.Li(f"Decimals: {decimals}"),
            ]
            if 'token_name' in token and token['token_name']:
                details.append(html.Li(f"Full Name: {token['token_name']}"))
            output_components.append(
                html.Details([
                    html.Summary(f"{i}. {token_name} ({token_symbol})"),
                    html.Ul(details)
                ])
            )

    # --- RECENT TRANSACTIONS ---
    transactions = address_data.get('transactions', [])
    output_components.append(html.H4("📋 RECENT TRANSACTIONS"))
    if transactions:
        output_components.append(html.P(f"Showing last {min(len(transactions), 10)} transactions:"))
        tx_list = []
        for i, tx in enumerate(transactions[:10], 1):
            tx_hash = tx.get('hash', 'N/A')
            block_id = tx.get('block_id', 'N/A')
            value_wei = tx.get('value', '0')
            value_eth = int(value_wei) / (10**18) if value_wei else 0
            fee_wei = tx.get('fee', '0')
            fee_eth = int(fee_wei) / (10**18) if fee_wei else 0
            tx_list.append(html.Li([
                html.B(f"{i}. Hash: {tx_hash[:20]}..."),
                html.Ul([
                    html.Li(f"Block: {block_id}"),
                    html.Li(f"Value: {value_eth:.6f} ETH"),
                    html.Li(f"Fee: {fee_eth:.6f} ETH"),
                ])
            ]))
        output_components.append(html.Ul(tx_list))
    else:
        output_components.append(html.P("No recent transactions found."))

    # --- RECENT SMART CONTRACT CALLS ---
    calls = address_data.get('calls', [])
    if calls:
        output_components.append(html.H4("🔧 RECENT SMART CONTRACT CALLS"))
        output_components.append(html.P(f"Showing last {min(len(calls), 5)} calls:"))
        call_list = []
        for i, call in enumerate(calls[:5], 1):
            tx_hash = call.get('transaction_hash', 'N/A')
            block_id = call.get('block_id', 'N/A')
            recipient = call.get('recipient', 'N/A')
            call_list.append(html.Li([
                html.B(f"{i}. Transaction: {tx_hash[:20]}..."),
                html.Ul([
                    html.Li(f"Block: {block_id}"),
                    html.Li(f"Recipient: {recipient[:20]}..."),
                ])
            ]))
        output_components.append(html.Ul(call_list))

    # --- API CONTEXT ---
    context = data.get('context', {})
    if context:
        output_components.append(html.H4("🌐 API CONTEXT"))
        output_components.append(html.Ul([
            html.Li(f"API Version: {context.get('api_version', 'N/A')}"),
            html.Li(f"Results: {context.get('results', 'N/A')}"),
            html.Li(f"State: {context.get('state', 'N/A')}"),
        ]))

    # --- STRATEGY RECOMMENDATIONS ---
    strategy_recommendations = [html.H4("🚀 Strategy Recommendations")]
    if eth_balance == 0 and not unique_token_list:
        strategy_recommendations.append(html.P("🎯 Suggested posture: NO POSITION"))
        strategy_recommendations.append(html.P("This wallet is currently inactive or empty."))
    else:
        if len(unique_token_list) > 5:
            strategy_recommendations.append(html.P("🎯 Suggested posture: DIVERSIFIED PORTFOLIO"))
        elif len(unique_token_list) > 0:
            strategy_recommendations.append(html.P("🎯 Suggested posture: FOCUSED HOLDINGS"))
        else:
            strategy_recommendations.append(html.P("🎯 Suggested posture: NATIVE ASSET ONLY"))
        strategy_recommendations.append(html.P("⚖️ Rebalance monthly if any single asset allocation drifts by more than 10%."))
        strategy_recommendations.append(html.P("📊 Consider Dollar-Cost Averaging (DCA) for top holdings if your conviction remains high."))
    output_components.append(html.Hr())
    output_components.extend(strategy_recommendations)
    output_components.append(html.Hr())

    return html.Div(output_components)

