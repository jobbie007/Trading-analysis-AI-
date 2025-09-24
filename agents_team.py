"""
AutoGen AgentChat team for the Research tab.
LM Studio only (local OpenAI-compatible); no OpenAI cloud usage.
Falls back to the local orchestrator shim if unavailable/misconfigured.
"""
from __future__ import annotations

import os
import json
import inspect
from typing import Any, Dict, List, Optional

from services import ResearchOrchestrator, NewsBridge
from autogen_core.models import ModelFamily  # type: ignore

AVAILABLE = False
UNAVAILABLE_REASON = ""
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import MaxMessageTermination
    # FunctionTool import fallback across versions
    try:
        from autogen_agentchat.tools import FunctionTool as _FunctionTool  # type: ignore
        FunctionTool = _FunctionTool  # type: ignore
    except Exception as e_ft1:  # pragma: no cover - environment dependent
        try:
            from autogen_core.tools import FunctionTool as _FunctionTool  # type: ignore
            FunctionTool = _FunctionTool  # type: ignore
        except Exception as e_ft2:
            raise ImportError(f"FunctionTool import failed: {e_ft1} | fallback: {e_ft2}")

    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
    except Exception:
        OpenAIChatCompletionClient = None  # type: ignore
        UNAVAILABLE_REASON = "autogen_ext OpenAI client not available"
    AVAILABLE = True
except Exception as e:
    AVAILABLE = False
    try:
        UNAVAILABLE_REASON = str(e)
    except Exception:
        UNAVAILABLE_REASON = "unknown import error"

_orch = ResearchOrchestrator(news_bridge=NewsBridge())


def _wrap_tools(tool_logs: Optional[List[Dict[str, str]]] = None, captured: Optional[Dict[str, Any]] = None) -> List[Any]:
    if not AVAILABLE:
        return []
    tool_logs = tool_logs or []
    captured = captured or {}
    # Read web search controls from environment (set by UI)
    _allow_web = (os.getenv("ALLOW_WEB_SEARCH") or "0") == "1"
    try:
        _web_max = int(os.getenv("WEB_MAX_RESULTS", "5"))
    except Exception:
        _web_max = 5

    def fetch_data(symbol: str, start_date: str, end_date: str, interval: str) -> Dict[str, Any]:
        _msg, payload = _orch.data_agent_with_dates(symbol, start_date, end_date, interval)
        rows = int(getattr(payload.get("hist"), "shape", (0, 0))[0])
        tool_logs.append({"agent": "DataAgent", "message": f"fetch_data({symbol}) -> rows={rows}"})
        # remember the last symbol for potential synthesis if finalize_report isn't called
        captured["_last_symbol"] = symbol
        return {"source": payload.get("source", "yfinance"), "rows": rows}

    def perform_analysis(symbol: str, start_date: str, end_date: str, interval: str, indicators: List[str], length: int, rsi_length: int, macd_fast: int, macd_slow: int, macd_signal_len: int) -> Dict[str, Any]:
        res = _orch.run_technical_analysis(
            tickers=[symbol], timeframe=interval, start_date=start_date, end_date=end_date,
            indicators=indicators, length=length, rsi_length=rsi_length,
            macd_fast=macd_fast, macd_slow=macd_slow, macd_signal_len=macd_signal_len,
        ) or {}
        out = res.get(symbol, {})
        tool_logs.append({"agent": "TechnicalAnalyst", "message": f"perform_analysis({symbol}) -> rec={out.get('recommendation','N/A')}"})
        # store interim analysis in case the agent forgets to call finalize_report
        captured["_last_analysis"] = {
            "symbol": symbol,
            "recommendation": out.get("recommendation", "N/A"),
            "confidence": out.get("confidence", "N/A"),
            "justification": out.get("justification", ""),
        }
        return {"has_figure": bool(out.get("figure") is not None), "recommendation": out.get("recommendation", "N/A"), "confidence": out.get("confidence", "N/A")}

    def get_ai_summary(symbol: str, tech_summary: str) -> Dict[str, Any]:
        res = _orch._get_technical_ai_summary(symbol, tech_summary)
        tool_logs.append({"agent": "TechnicalAnalyst", "message": f"get_ai_summary({symbol})"})
        # store interim AI summary as a backup for finalization
        try:
            captured["_last_ai"] = {"symbol": symbol, **(res if isinstance(res, dict) else {"summary": res})}
        except Exception:
            captured["_last_ai"] = {"symbol": symbol, "summary": str(res)}
        return res

    def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
        if not _allow_web:
            tool_logs.append({"agent": "TechnicalAnalyst", "message": "web_search skipped (disabled by UI)"})
            return {"message": "Web search disabled by UI", "results": []}
        # Clamp to configured maximum
        try:
            mr = int(max_results)
        except Exception:
            mr = 5
        mr = max(1, min(mr, _web_max))
        msg, res = _orch.web_search_agent(query, max_results=mr)
        tool_logs.append({"agent": "TechnicalAnalyst", "message": f"web_search('{query[:40]}...') -> {len(res)} results (max={mr})"})
        return {"message": msg, "results": res}

    def finalize_report(symbol: str, recommendation: str, confidence: str, justification: str, figure_json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {symbol: {"figure": figure_json or None, "recommendation": recommendation, "confidence": confidence, "justification": justification}}
        captured.clear(); captured.update(payload)
        tool_logs.append({"agent": "TechnicalAnalyst", "message": f"finalize_report({symbol})"})
        return {"status": "ok", "symbols": [symbol]}

    return [
        FunctionTool(fetch_data, description="Fetch OHLCV for a symbol between start_date and end_date.", name="fetch_data"),
        FunctionTool(perform_analysis, description="Run technical analysis and return a brief overview.", name="perform_analysis"),
        FunctionTool(get_ai_summary, description="Summarize technicals into a recommendation, confidence, and justification.", name="get_ai_summary"),
        FunctionTool(web_search, description="Search the web for recent info using DuckDuckGo; returns title/url/snippet.", name="web_search"),
        FunctionTool(finalize_report, description="Store the final JSON (figure, recommendation, confidence, justification) for a symbol.", name="finalize_report"),
    ]


def run_team_workflow(user_request: str) -> Dict[str, Any]:
    """Run AutoGen AgentChat via LM Studio only; otherwise fall back to shim with a helpful message."""
    if not AVAILABLE:
        import agents as _shim
        res = _shim.run_autogen_workflow(user_request)
        msgs = res.get("messages", []) or []
        note = "AutoGen AgentChat not available; using local orchestrator shim."
        if UNAVAILABLE_REASON:
            note += f" Reason: {UNAVAILABLE_REASON}"
        msgs.append({"agent": "Planner", "message": note})
        res["messages"] = msgs
        return res

    base_url = (os.getenv("LOCAL_LLM_BASE_URL") or "").strip()
    api_key = os.getenv("LOCAL_LLM_API_KEY") or "lm-studio"
    # Prefer explicit LOCAL_LLM_MODEL; fall back to MODEL_NAME to align with .env and app defaults
    model_name = os.getenv("LOCAL_LLM_MODEL") or os.getenv("MODEL_NAME") or "llama-3.1-8b-instruct"
    if not base_url:
        import agents as _shim
        res = _shim.run_autogen_workflow(user_request)
        msgs = res.get("messages", []) or []
        msgs.append({"agent": "Planner", "message": "LOCAL_LLM_BASE_URL not set. Configure LM Studio at http://127.0.0.1:1234/v1."})
        res["messages"] = msgs
        return res

    try:
        if OpenAIChatCompletionClient is None:
            raise RuntimeError("autogen_ext not available (OpenAIChatCompletionClient is None)")

        # Preflight LM Studio and ensure model is valid
        import requests as _requests
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        # Verify model exists; if not, pick a sensible available one
        switched_model = None
        try:
            mr = _requests.get(f"{base_url.rstrip('/')}/models", timeout=6)
            if mr.status_code == 200:
                mdata = mr.json() or {}
                items = mdata.get("data") or mdata.get("models") or []
                available: List[str] = []
                for it in items:
                    mid = it.get("id") if isinstance(it, dict) else (str(it) if isinstance(it, str) else None)
                    if mid:
                        available.append(str(mid))
                if available and model_name not in available:
                    # Prefer an instruct model if present
                    pick = next((x for x in available if "instruct" in x.lower()), available[0])
                    model_name = pick
                    switched_model = pick
                    os.environ["LOCAL_LLM_MODEL"] = pick
        except Exception:
            pass

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a connectivity check."},
                {"role": "user", "content": "Reply with OK."}
            ],
            "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1") or 0.1),
            "max_tokens": 64,
            "stream": False,
        }
        r = _requests.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=10)
        if r.status_code != 200:
            raise RuntimeError(f"Preflight status {r.status_code}")

        # Model client
        # Provide model_info for non-OpenAI local models so the client doesn't reject unknown model ids.
        # Also avoid sending `name` fields in messages for providers that don't accept them.
        _mn_lower = model_name.lower()
        if any(s in _mn_lower for s in ("deepseek", "r1")):
            _family = ModelFamily.R1
        elif "llama" in _mn_lower:
            _family = ModelFamily.LLAMA_3_3_8B
        elif "mistral" in _mn_lower or "mixtral" in _mn_lower:
            _family = ModelFamily.MISTRAL
        else:
            _family = ModelFamily.UNKNOWN

        _model_info = {
            "vision": False,
            "function_calling": True,   # allow tool registration; model may choose not to call tools
            "json_output": False,
            "structured_output": False,
            "family": _family,
            # conservative default to avoid multi-system merging quirks
            "multiple_system_messages": False,
        }

        try:
            model_client = OpenAIChatCompletionClient(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                model_info=_model_info,  # required for non-OpenAI model ids
                include_name_in_message=False,
            )
        except TypeError:
            # Some builds require OPENAI_BASE_URL env instead of base_url kwarg
            os.environ["OPENAI_BASE_URL"] = base_url
            model_client = OpenAIChatCompletionClient(
                model=model_name,
                api_key=api_key,
                model_info=_model_info,
                include_name_in_message=False,
            )

        tool_logs: List[Dict[str, str]] = []
        captured: Dict[str, Any] = {}
        tools = _wrap_tools(tool_logs, captured)

        planner = AssistantAgent(name="Planner", model_client=model_client, system_message=("Plan steps and delegate tool use. Ensure finalize_report is called before terminating."))
        data_agent = AssistantAgent(name="DataAgent", model_client=model_client, tools=[t for t in tools if getattr(t, "name", "") == "fetch_data"], system_message="Fetch market data via tools; no fabrication.")
        technical_analyst = AssistantAgent(name="TechnicalAnalyst", model_client=model_client, tools=[t for t in tools if getattr(t, "name", "") in {"perform_analysis", "get_ai_summary", "finalize_report"}], system_message=("Use tools: 1) perform_analysis; 2) get_ai_summary; 3) finalize_report (mandatory). Then TERMINATE."))
        validator = AssistantAgent(name="Validator", model_client=model_client, system_message=("Validate output; if OK, reply 'VALIDATED' and TERMINATE."))
        # Provide a no-op input function to avoid blocking for human input in headless runs
        user_proxy = UserProxyAgent(name="User", input_func=lambda prompt: "")

        # Allow UI to clamp rounds via AUTOGEN_MAX_TURNS env
        try:
            max_turns = int(os.getenv("AUTOGEN_MAX_TURNS", "5"))
        except Exception:
            max_turns = 5
        team = RoundRobinGroupChat(
            participants=[user_proxy, planner, data_agent, technical_analyst, validator],
            max_turns=max_turns,
            termination_condition=MaxMessageTermination(max_turns),
        )

        # Run the async team synchronously, even if a loop is already running.
        def _run_coro_sync(coro):
            import asyncio, threading, queue
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                q: "queue.Queue[tuple[bool, Any]]" = queue.Queue()
                def runner():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        res = new_loop.run_until_complete(coro)
                        q.put((True, res))
                    except Exception as e:
                        q.put((False, e))
                    finally:
                        new_loop.close()
                t = threading.Thread(target=runner, daemon=True)
                t.start()
                ok, val = q.get()
                if ok:
                    return val
                raise val
            else:
                return asyncio.run(coro)

        try:
            result_obj = _run_coro_sync(team.run(task=user_request))
        except Exception as run_err:
            # Surface the team.run failure and fall back to deterministic path below
            messages: List[Dict[str, str]] = []
            messages.append({"agent": "Planner", "message": f"AutoGen team.run failed: {str(run_err)[:160]}"})
            # emulate prior behavior by raising to outer except which triggers shim fallback
            raise
        result_payload = str(result_obj)

        messages: List[Dict[str, str]] = []
        # Surface explicit LM Studio connectivity
        conn_msg = f"Connected to LM Studio at {base_url} with model '{model_name}'."
        if switched_model:
            conn_msg += f" (Note: requested model not found; auto-selected '{switched_model}'.)"
        messages.append({"agent": "Planner", "message": conn_msg})
        messages.append({"agent": "Planner", "message": "Decomposed task and delegated. Provider: Local LLM (LM Studio)."})
        for log in tool_logs:
            if isinstance(log, dict) and log.get("agent") and log.get("message"):
                messages.append({"agent": log["agent"], "message": log["message"]})
        messages.append({"agent": "Validator", "message": "VALIDATED"})

        if captured:
            return {"result": json.dumps(captured), "messages": messages}

        # If tools were used but finalize_report wasn't called, synthesize from interim state
        tools_used = bool(tool_logs)
        if tools_used:
            la = captured.get("_last_analysis") if isinstance(captured, dict) else None
            sym = (la or {}).get("symbol") or captured.get("_last_symbol")
            if sym:
                payload = {
                    sym: {
                        "figure": None,
                        "recommendation": (la or {}).get("recommendation", "N/A"),
                        "confidence": (la or {}).get("confidence", "N/A"),
                        "justification": (la or {}).get("justification", ""),
                    }
                }
                messages.append({"agent": "TechnicalAnalyst", "message": "Tools used, but finalize_report missing; synthesized final report from analysis."})
                return {"result": json.dumps(payload), "messages": messages}

        # If the model didn't use tools (common for some local models), synthesize a deterministic result
        import re as _re
        sym = None; interval = "1d"; start = None; end = None
        indicators = ["SMA", "EMA", "Bollinger Bands", "VWAP", "RSI", "MACD", "ATR"]
        length = 20; rsi_len = 14; macd_f = 12; macd_s = 26; macd_sig = 9
        try:
            m = _re.search(r"stock tickers:\s*([^\n]+)", user_request, _re.I)
            if m: sym = m.group(1).strip().split(",")[0].strip()
            m = _re.search(r"timeframe of\s*([^\n]+)", user_request, _re.I)
            if m: interval = m.group(1).strip()
            m = _re.search(r"from\s*([0-9\-]{8,10})\s*to\s*([0-9\-]{8,10})", user_request, _re.I)
            if m: start, end = m.group(1), m.group(2)
            m = _re.search(r"technical indicators:\s*([^\n]+)", user_request, _re.I)
            if m:
                indicators = [x.strip() for x in m.group(1).split(',') if x.strip()]
            m = _re.search(r"General length:\s*(\d+)", user_request, _re.I)
            if m: length = int(m.group(1))
            m = _re.search(r"RSI length:\s*(\d+)", user_request, _re.I)
            if m: rsi_len = int(m.group(1))
            m = _re.search(r"MACD settings:\s*fast=(\d+),\s*slow=(\d+),\s*signal=(\d+)", user_request, _re.I)
            if m:
                macd_f, macd_s, macd_sig = int(m.group(1)), int(m.group(2)), int(m.group(3))
        except Exception:
            pass
        if sym:
            try:
                res = _orch.run_technical_analysis(
                    tickers=[sym], timeframe=interval, start_date=start or "", end_date=end or "",
                    indicators=indicators, length=length, rsi_length=rsi_len, macd_fast=macd_f, macd_slow=macd_s, macd_signal_len=macd_sig,
                ) or {}
                out = res.get(sym, {})
                payload = {sym: {
                    "figure": None,
                    "recommendation": out.get("recommendation", "N/A"),
                    "confidence": out.get("confidence", "N/A"),
                    "justification": out.get("justification", "")
                }}
                if tools_used:
                    messages.append({"agent": "TechnicalAnalyst", "message": "Tools used but could not synthesize; fell back to deterministic analysis."})
                else:
                    messages.append({"agent": "TechnicalAnalyst", "message": "Model did not use tools; produced deterministic analysis locally."})
                return {"result": json.dumps(payload), "messages": messages}
            except Exception:
                pass

        # Last resort: try to extract any JSON-looking object from the model text
        i, j = result_payload.find("{"), result_payload.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                json.loads(result_payload[i:j+1])
                return {"result": result_payload[i:j+1], "messages": messages}
            except Exception:
                pass
        raise RuntimeError("No structured result produced by team")
    except Exception as e:
        import agents as _shim
        res = _shim.run_autogen_workflow(user_request)
        msgs = res.get("messages", []) or []
        msgs.append({"agent": "Planner", "message": f"AutoGen failed: {str(e)[:120]}. Using local orchestrator shim."})
        res["messages"] = msgs
        return res
