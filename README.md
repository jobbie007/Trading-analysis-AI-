# Dash Financial Intelligence App

This is a standalone Dash app with tabs:
- News & Analysis: Big TradingView chart with news below
  <img width="1908" height="932" alt="image" src="https://github.com/user-attachments/assets/e6b8fc9d-1c71-4992-8b8f-9222b5582734" />

- Strategy (Wallet): Enter a BTC/ETH address; uses Blockchair API to suggest a simple posture

## Quick start (Windows PowerShell)
```powershell
cd "c:\Users\YourName\Documents\dash"
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

## Project structure

- ui/               UI assets and layouts used by the Dash app
	- ui/assets/      Static assets served by Dash
	- ui/layouts/     Tab/page layout modules
- agents/           Agent implementations (AutoGen/LangChain/etc.)
- utils/            Shared utilities
- tests/            Unit tests
- dev/              Developer docs and helper scripts
	- dev/docs/       Internal documentation and plans
	- dev/scripts/    PowerShell/python helper scripts

## Maintenance

- Cleanup transient artifacts:
	- Dry-run: `python .\\dev\\scripts\\prune_artifacts.py`
	- Apply: `python .\\dev\\scripts\\prune_artifacts.py --apply`
- Safe archival for unused modules: create `dash/_archive/` and move files there. Do not delete modules imported by `app.py` (`services.py`, `nd_agents.py`, `agents.py`, `agents_team.py`, `lc_agents.py`, `telemetry.py`).

## Architecture and design patterns (rubric mapping)

- App pattern: The UI is built with Dash callbacks (View). The business logic lives in `services.py` and `news_core.py` (Model/Service layer). The orchestration in `ResearchOrchestrator` coordinates agents as a Presenter/Controller. This aligns with an MVP/MVC hybrid suitable for Dash.
- Additional patterns used:
	- Strategy pattern for LLM provider selection (`_llm_generate` and NewsBridge provider order).
	- Adapter/Bridge to integrate external LLMs via an OpenAI-compatible interface or Gemini bridge.
	- Facade over news/data services via `NewsBridge` to present a uniform API to the UI.
- Code quality:
	- Centralized logging via `telemetry.py` and module logger in `services.py`.
	- Typed dataclasses for domain objects like `NewsItem` and `Task`.
	- JSON utility `json_utils.extract_json_object` with unit tests (`tests/test_json_utils.py`).
	- Tooling configured in `pyproject.toml`: Black (format), Ruff (lint), Mypy (types), Pytest (tests).

### Quality gates

Optional, but recommended before submission/presentation:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install -r dev/dev-requirements.txt
& ".\.venv\Scripts\python.exe" -m ruff check . --fix
& ".\.venv\Scripts\python.exe" -m black .
& ".\.venv\Scripts\python.exe" -m mypy services.py news_core.py
& ".\.venv\Scripts\python.exe" -m pytest -q
```

Notes:
- The lint rules are strict; fix what’s feasible quickly (imports, unuseds), then defer deeper refactors as needed.
- If you don’t have keys for external services, the app degrades gracefully and shows local placeholders.
 - Caches are centralized under `.cache/`:
	 - Pytest: `.cache/pytest`
	 - Ruff: `.cache/ruff`
	 - Mypy: `.cache/mypy`
	 These are ignored by git.
