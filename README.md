# Dash Financial Intelligence App

This is a standalone Dash app with tabs:
- News & Analysis: Big TradingView chart with news below
- Strategy (Wallet): Enter a BTC/ETH address; uses Blockchair API to suggest a simple posture

## Quick start (Windows PowerShell)
```powershell
cd "c:\Users\Jobbie\c++Workspace\Comp301 project\dash"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:8050

## Environment
- Create a `.env` file in this folder to add optional keys:
```
BLOCKCHAIR_API_KEY=your_key_here
```

## Wire your existing services
- Edit `app.py`, see the `load_news` callback. Replace the placeholder with your news service.

## Maintenance

- Cleanup transient artifacts:
	- Dry-run: `python .\\dash\\scripts\\prune_artifacts.py`
	- Apply: `python .\\dash\\scripts\\prune_artifacts.py --apply`
- Safe archival for unused modules: create `dash/_archive/` and move files there. Do not delete modules imported by `app.py` (`services.py`, `nd_agents.py`, `agents.py`, `agents_team.py`, `lc_agents.py`, `telemetry.py`).
