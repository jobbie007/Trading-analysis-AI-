# News and Analysis Project Restructure Plan

This document proposes a modular, maintainable folder structure for the News and analysis app (Dash-based), maps current files to new locations, and outlines a safe migration path.

## Proposed folder structure

```
 News and analysis/
 ├─ app/
 │  ├─ __init__.py               # Create Dash app, register pages, share app/server
 │  ├─ server.py                 # gunicorn/WSGI entrypoint (imports app from __init__)
 │  ├─ callbacks/                # Dash callbacks grouped by domain
 │  │   ├─ __init__.py
 │  │   ├─ news_callbacks.py
 │  │   ├─ research_callbacks.py
 │  │   └─ strategy_callbacks.py
 │  ├─ ui/
 │  │   ├─ __init__.py
 │  │   ├─ components/           # Small reusable components (cards, badges, toolbars)
 │  │   │   ├─ __init__.py
 │  │   │   └─ news_card.py
 │  │   ├─ pages/                # Page/tab layouts (Dash pages)
 │  │   │   ├─ __init__.py
 │  │   │   ├─ news_page.py
 │  │   │   ├─ strategy_page.py
 │  │   │   └─ research_page.py
 │  │   └─ assets/ -> ../../assets/   # Use Dash assets folder (symlink or shared)
 │  └─ telemetry.py              # Keep lightweight UI telemetry helpers
 │
 ├─ domain/                      # Core business logic, pure Python (no Dash)
 │  ├─ __init__.py
 │  ├─ news/
 │  │   ├─ __init__.py
 │  │   ├─ news_core.py          # from news_core.py (moved)
 │  │   └─ models.py             # NewsArticle, NewsItem (dataclasses)
 │  ├─ research/
 │  │   ├─ __init__.py
 │  │   ├─ orchestrator.py       # from ResearchOrchestrator in services.py
 │  │   └─ workflows.py          # from workflows.py (pure data/config)
 │  └─ utils/
 │      ├─ __init__.py
 │      └─ logging.py            # central logger factory
 │
 ├─ services/                    # Gateways/adapters to external services & providers
 │  ├─ __init__.py
 │  ├─ news_bridge.py            # NewsBridge (wraps domain.news and LLM clients)
 │  ├─ providers/
 │  │   ├─ __init__.py
 │  │   ├─ llm.py                # OpenAI-compatible client helpers
 │  │   └─ web_search.py         # Serper, DDG wrappers
 │  └─ settings.py               # env loading & application settings
 │
 ├─ agents/                      # Optional agent frameworks (LangChain, AutoGen, ND)
 │  ├─ __init__.py
 │  ├─ autogen_team.py           # from agents_team.py
 │  ├─ langchain_team.py         # from lc_agents.py
 │  ├─ nd_team.py                # from nd_agents.py
 │  └─ compatibility.py          # thin shims used by UI when toolkits missing
 │
 ├─ cli/
 │  ├─ __init__.py
 │  └─ main.py                   # CLI entry points (e.g., fetch news, run research)
 │
 ├─ tests/
 │  ├─ test_news_core.py
 │  ├─ test_news_bridge.py
 │  └─ test_orchestrator.py
 │
 ├─ assets/                      # Static CSS/JS/images (existing)
 ├─ logs/                        # Runtime logs (existing)
 ├─ scripts/                     # Helper scripts (existing)
 ├─ .env.example                 # Document expected env vars
 ├─ requirements.txt             # Keep as-is
 ├─ README.md                    # Update run instructions
 └─ run.py                       # Dev entry (flask dev server) – optional
```

Key principles:
- Separate UI (Dash) from domain logic. UI imports domain/services; domain never imports UI.
- Keep external I/O (HTTP, LLMs, web search) behind `services/` adapters.
- Group agent frameworks under `agents/` with clean shims so UI doesn’t depend on them existing.
- Single source of configuration in `services/settings.py` using env vars.

Note about folder name with spaces:
- The project root can have spaces (e.g., `News and analysis/`). Python imports work as long as you run commands from the project root and your internal packages (e.g., `app/`, `services/`, `domain/`) have `__init__.py`. If you prefer running as a package (`python -m ...`), consider adding a top-level package directory with a valid Python name (e.g., `news_and_analysis/`) and placing code under it.

## File mapping (from -> to)

- `app.py` -> split into:
  - `app/__init__.py` (create Dash app, register pages, high-level layout scaffolding)
  - `app/ui/pages/news_page.py` (from `layouts/news.py`)
  - `app/ui/pages/strategy_page.py` (from `layouts/strategy.py`)
  - `app/ui/pages/research_page.py` (extract research layout from `app.py`)
  - `app/callbacks/*` (move callbacks from `app.py` into domain-specific modules)
  - `app/server.py` (expose `server` for deployment)

- `news_core.py` -> `domain/news/news_core.py`
- `workflows.py` -> `domain/research/workflows.py` (data-only)
- `telemetry.py` -> `app/telemetry.py` (UI event logging wrappers); core logger in `domain/utils/logging.py`
- `services.py` -> split into:
  - `services/news_bridge.py` (NewsBridge, NewsItem dataclass)
  - `domain/research/orchestrator.py` (ResearchOrchestrator)
  - `services/providers/llm.py` (OpenAI-compatible helpers used by NewsBridge and agents)
  - `services/providers/web_search.py` (Serper + DDG wrappers)
  - `services/settings.py` (env loading; unify settings used by services and domain)

- `agents.py` -> `agents/compatibility.py` (shim used by UI)
- `agents_team.py` -> `agents/autogen_team.py`
- `lc_agents.py` -> `agents/langchain_team.py`
- `nd_agents.py` -> `agents/nd_team.py`

- `layouts/` -> `app/ui/pages/` and `app/ui/components/`

## Callback re-organization

- `app/callbacks/news_callbacks.py`
  - fetch news, render list, selection toggle, overall summary, provider control visibility
- `app/callbacks/research_callbacks.py`
  - research config handlers, run action, model lists, llm test
- `app/callbacks/strategy_callbacks.py`
  - wallet analyzer, LLM strategy generation

Each callback module gets its own `register_callbacks(app)` function called from `app/__init__.py`.

## Imports and circular dependencies

- UI imports only: `from services.news_bridge import NewsBridge`, `from domain.research.orchestrator import ResearchOrchestrator`.
- Services import domain models (`domain.news.news_core`) but NOT UI.
- Agents import services and domain.

## Migration steps (safe, incremental)

1. Create new package skeleton with `__init__.py` files (no code moves yet).
2. Extract research page layout from `app.py` into `app/ui/pages/research_page.py` and update imports.
3. Move `layouts/news.py` -> `app/ui/pages/news_page.py` and update `app.py` imports to the new function name (`news_page.layout`).
4. Create `app/callbacks/news_callbacks.py` and move the following callbacks from `app.py`:
   - `choose_asset`, `update_tv`, `fetch_news`, `render_news`, `render_overall`, `toggle_selected`, `_toggle_provider_controls`.
   Register them via `register_callbacks(app)`.
5. Split `services.py`:
   - Move `NewsItem` + `NewsBridge` into `services/news_bridge.py` (adjust imports to use `domain.news.news_core`).
   - Move `ResearchOrchestrator` to `domain/research/orchestrator.py` and update agent modules to import from there.
   - Extract `_SerperClient` and DDG logic to `services/providers/web_search.py` and use them from orchestrator when needed.
6. Move `news_core.py` -> `domain/news/news_core.py` and fix imports in `services/news_bridge.py`.
7. Move agent modules into `agents/` and update import paths in the app.
8. Add `app/server.py` that exposes `server = app.server` for deployment.
9. Run app, fix import paths; add `__init__.py` across new packages.
10. Add minimal unit tests for `domain/news/news_core.py` and `domain/research/orchestrator.py`.

You can do steps 2–4 first to reduce `app.py` size significantly without touching core logic.

## Conventions and guidelines

- Module boundaries:
  - `domain/*`: Pure logic, no Dash, minimal I/O.
  - `services/*`: External I/O and provider integrations.
  - `app/*`: Dash UI: pages, components, callbacks.
  - `agents/*`: Optional frameworks; UI should work without them.
- Naming:
  - Files: `snake_case.py`. Classes: `PascalCase`. Functions: `snake_case`.
  - Avoid long files (>500 lines). Split by concern.
- Config:
  - Centralize env reading in `services/settings.py`.
  - UI reads defaults from settings when needed.
- Logging:
  - Use `domain.utils.logging.get_logger(name)` to create namespaced loggers.
  - Keep UI telemetry lightweight and non-fatal.

## Quick wins (do now)

- Add `__init__.py` to key packages: `app/`, `app/ui/`, `app/ui/pages/`, `app/callbacks/`, `services/`, `services/providers/`, `domain/`, `domain/news/`, `domain/research/`, `agents/`.
- Extract the Research tab layout from `app.py` into `app/ui/pages/research_page.py`.
- Move news callbacks to `app/callbacks/news_callbacks.py` and register via `register_callbacks(app)`.
- Update README with run instructions appropriate to this project folder name. On Windows PowerShell, quote paths with spaces when needed.

Example (optional) run commands:

```powershell
# from the project root (News and analysis\)
python .\app.py

# or if you create app/__init__.py and a tiny run.py entrypoint
python .\run.py
```

## Rollback plan

- Keep original files until each step is verified.
- Introduce thin re-export shims (e.g., old module imports new location) during transition.
- Use git moves to preserve history (`git mv`).

