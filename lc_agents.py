"""
LangChain-based lightweight workflow for the Research tab.
Uses a local OpenAI-compatible endpoint via langchain-openai if available.
Falls back to the deterministic orchestrator when imports or endpoint are unavailable.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from services import ResearchOrchestrator, NewsBridge  # Use local services module

AVAILABLE = False
try:
    # langchain-openai provides ChatOpenAI for OpenAI-compatible servers
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    AVAILABLE = True
except Exception:
    AVAILABLE = False

_orch = ResearchOrchestrator(news_bridge=NewsBridge())


def run_langchain_workflow(user_request: str, model_name: str) -> Dict[str, Any]:
    """Runs a minimal LangChain-backed step: we use local deterministic analysis
    for data/indicators and ask the LLM only for a short justification/summary.
    Returns {"result": json_str, "messages": [...]} for UI consumption.
    """
    messages: List[Dict[str, str]] = []
    try:
        from telemetry import log_event
        log_event("agent.lang.request", {"user_request": (user_request or "")[:1200]})
    except Exception:
        pass

    # Try late import (the server env may differ from linter env). If still unavailable, we'll call HTTP directly later.
    _available = AVAILABLE
    if not _available:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
            _available = True
        except Exception:
            _available = False

    # Parse a very small instruction set from the user_request (we expect a single ticker)
    symbol = None
    for line in (user_request or "").splitlines():
        if "stock tickers:" in line.lower():
            part = line.split(":", 1)[-1].strip()
            symbol = (part.split(",")[0] or "").strip()
            break
    if not symbol:
        import agents as _shim
        res = _shim.run_autogen_workflow(user_request)
        msgs = res.get("messages", []) or []
        msgs.append({"agent": "Planner(LC)", "message": "No symbol parsed; using local orchestrator shim."})
        res["messages"] = msgs
        return res

    # Do deterministic technical analysis locally
    messages.append({"agent": "Data(LC)", "message": f"Fetched and analyzed technicals for {symbol}."})
    analysis = _orch.run_technical_analysis(
        tickers=[symbol],
        timeframe="1d",
        start_date=None,
        end_date=None,
        indicators=["SMA", "EMA", "RSI", "MACD"],
        length=20,
        rsi_length=14,
        macd_fast=12,
        macd_slow=26,
        macd_signal_len=9,
    ) or {}
    per = analysis.get(symbol, {})

    # Prepare a concise prompt for justification
    tech_summary = json.dumps({
        "recommendation": per.get("recommendation", "N/A"),
        "confidence": per.get("confidence", "N/A"),
        "notes": per.get("summary", ""),
    })

    base_url = (os.getenv("LOCAL_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()
    model = model_name or "local-model" # Use the parameter, with a fallback
    api_key = os.getenv("LOCAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"
    try:
        max_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "1024") or 1024)
    except Exception:
        max_tokens = 1024
    try:
        temperature = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.1") or 0.1)
    except Exception:
        temperature = 0.1

    # If no local endpoint or LC can't initialize, fallback
    if not base_url:
        import agents as _shim
        res = _shim.run_autogen_workflow(user_request)
        msgs = res.get("messages", []) or []
        msgs.append({"agent": "Planner(LC)", "message": "No LOCAL_LLM_BASE_URL; using local orchestrator shim."})
        res["messages"] = msgs
        try:
            from telemetry import log_event
            log_event("agent.lang.fallback", {"reason": "no_base_url"})
        except Exception:
            pass
        return res

    if _available:
        try:
            llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens)  # type: ignore[name-defined]
            sys_msg = SystemMessage(content=(  # type: ignore[name-defined]
                "You are a financial technical analyst. Given a structured technical summary, write a brief justification"
                " (2-3 sentences) for the recommendation suitable for an executive summary."
            ))
            user_msg = HumanMessage(content=f"Technical summary: {tech_summary}")  # type: ignore[name-defined]
            resp = llm.invoke([sys_msg, user_msg])
            justification = getattr(resp, "content", "") or ""
            messages.append({"agent": "Analyst(LC)", "message": "Generated justification."})
            try:
                from telemetry import log_event
                log_event("agent.lang.llm_response", {"provider": "langchain_openai", "model": model, "snippet": justification[:400]})
            except Exception:
                pass
        except Exception as e:
            # If the LC call fails, try raw HTTP as a fallback
            try:
                import requests as _requests
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a financial technical analyst."},
                        {"role": "user", "content": f"Write 2-3 sentence justification for: {tech_summary}"},
                    ],
                    "temperature": temperature,
                    "max_tokens": min(max_tokens, 1024),
                    "stream": False,
                }
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                r = _requests.post(base_url.rstrip('/') + "/chat/completions", headers=headers, json=payload, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    justification = (
                        data.get("choices", [{}])[0].get("message", {}).get("content")
                        or data.get("choices", [{}])[0].get("text", "")
                        or per.get("summary", "")
                    )
                    messages.append({"agent": "Analyst(LC)", "message": "Generated justification via HTTP."})
                    try:
                        from telemetry import log_event
                        log_event("agent.lang.http_response", {"status": r.status_code, "snippet": (justification or "")[:400]})
                    except Exception:
                        pass
                else:
                    justification = per.get("summary", "")
                    messages.append({"agent": "Analyst(LC)", "message": f"HTTP error {r.status_code}; used deterministic summary."})
                    try:
                        from telemetry import log_event
                        log_event("agent.lang.http_error", {"status": r.status_code, "text": r.text[:300]})
                    except Exception:
                        pass
            except Exception as e2:
                justification = per.get("summary", "")
                messages.append({"agent": "Analyst(LC)", "message": f"LLM error: {str(e)[:90]} / HTTP error: {str(e2)[:90]}. Used deterministic summary."})
                try:
                    from telemetry import log_event
                    log_event("agent.lang.double_error", {"error1": str(e)[:200], "error2": str(e2)[:200]})
                except Exception:
                    pass
    else:
        # No LangChain: use raw HTTP directly
        try:
            import requests as _requests
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a financial technical analyst."},
                    {"role": "user", "content": f"Write 2-3 sentence justification for: {tech_summary}"},
                ],
                "temperature": temperature,
                "max_tokens": min(max_tokens, 1024),
                "stream": False,
            }
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            r = _requests.post(base_url.rstrip('/') + "/chat/completions", headers=headers, json=payload, timeout=20)
            if r.status_code == 200:
                data = r.json()
                justification = (
                    data.get("choices", [{}])[0].get("message", {}).get("content")
                    or data.get("choices", [{}])[0].get("text", "")
                    or per.get("summary", "")
                )
                messages.append({"agent": "Analyst(LC)", "message": "Generated justification via HTTP."})
            else:
                justification = per.get("summary", "")
                messages.append({"agent": "Analyst(LC)", "message": f"HTTP error {r.status_code}; used deterministic summary."})
        except Exception as e3:
            justification = per.get("summary", "")
            messages.append({"agent": "Analyst(LC)", "message": f"HTTP error: {str(e3)[:120]}. Used deterministic summary."})

    # Build a minimal compatible result JSON
    fig = per.get("figure")
    # Ensure figure is properly serialized
    fig_dict = None
    if fig is not None:
        try:
            # Convert Plotly figure to dict format
            if hasattr(fig, "to_dict"):
                fig_dict = fig.to_dict()
            elif hasattr(fig, "to_plotly_json"):
                fig_dict = fig.to_plotly_json()
            elif isinstance(fig, dict):
                fig_dict = fig
        except Exception as e:
            messages.append({"agent": "Analyst(LC)", "message": f"Figure serialization error: {str(e)[:100]}. Chart may not display."})
            fig_dict = None
    
    payload = {
        symbol: {
            "figure": fig_dict,
            "recommendation": per.get("recommendation", "N/A"),
            "confidence": per.get("confidence", "N/A"),
            "justification": justification,
        }
    }
    out = {"result": json.dumps(payload), "messages": messages}
    try:
        from telemetry import log_event
        log_event("agent.lang.response", {"symbol": symbol, "has_figure": bool(payload.get(symbol, {}).get("figure")), "messages": len(messages)})
    except Exception:
        pass
    return out
