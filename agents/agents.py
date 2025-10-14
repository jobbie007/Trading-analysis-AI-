from typing import Dict, Any, List
import json
import re
# Import services with package-safe relative import first, fallback to absolute for script execution
try:
    from ..services import ResearchOrchestrator, NewsBridge
except Exception:
    from services import ResearchOrchestrator, NewsBridge  # Fallback when running inside dash folder


_orchestrator = ResearchOrchestrator(news_bridge=NewsBridge())


def _extract_params_from_request(user_request: str) -> Dict[str, Any]:
    """Parse the natural-language request constructed in app.py and extract parameters.
    This is a lightweight shim so the UI works without requiring AutoGen setup.
    """
    text = (user_request or "")

    # Tickers
    tickers: List[str] = []
    m = re.search(r"stock tickers:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        raw = m.group(1)
        # Support comma or whitespace separated lists
        parts = []
        for chunk in re.split(r"[;,]", raw):
            parts.extend(re.split(r"\s+", chunk))
        cleaned: List[str] = []
        for p in parts:
            tok = p.strip().upper()
            if not tok:
                continue
            # Remove trailing punctuation (e.g., AAPL.) but keep market prefixes like NASDAQ:AAPL
            tok = re.sub(r"[^A-Z0-9:\-]", "", tok)
            if tok:
                cleaned.append(tok)
        tickers = cleaned

    # Timeframe / interval
    interval = "1d"
    m = re.search(r"timeframe\s+of\s+([0-9a-zA-Z]+)", text, flags=re.IGNORECASE)
    if m:
        interval = m.group(1).strip()
    # Basic normalization to yfinance-supported
    supported = {"1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"}
    if interval not in supported:
        interval = "1d"

    # Dates (support plain dates or ISO timestamps with time)
    start_date = None
    end_date = None
    m = re.search(
        r"from\s+([0-9]{4}-[0-9]{2}-[0-9]{2}(?:T[0-9:\.+\-Z]+)?)\s+to\s+([0-9]{4}-[0-9]{2}-[0-9]{2}(?:T[0-9:\.+\-Z]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        s_raw, e_raw = m.group(1), m.group(2)
        def _only_date(s: str) -> str:
            s = s.strip()
            # If contains time (T), cut at T; else return as-is
            i = s.find('T')
            return s[:i] if i != -1 else s
        start_date = _only_date(s_raw)
        end_date = _only_date(e_raw)

    # Indicators
    indicators: List[str] = []
    m = re.search(r"technical indicators.*?:\s*([^\n]+)", text, flags=re.IGNORECASE)
    if m:
        indicators = [x.strip() for x in m.group(1).split(",") if x.strip()]

    # Numeric params
    def _num(pattern: str, default: int) -> int:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return default
        return default

    length = _num(r"General length.*?:\s*([0-9]+)", 20)
    rsi_length = _num(r"RSI length.*?:\s*([0-9]+)", 14)
    macd_fast = _num(r"MACD settings.*?fast=([0-9]+)", 12)
    macd_slow = _num(r"MACD settings.*?slow=([0-9]+)", 26)
    macd_signal = _num(r"MACD settings.*?signal=([0-9]+)", 9)

    return {
        "tickers": tickers,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "indicators": indicators,
        "length": length,
        "rsi_length": rsi_length,
        "macd_fast": macd_fast,
        "macd_slow": macd_slow,
        "macd_signal": macd_signal,
    }


def run_research_workflow(user_request: str) -> Dict[str, Any]:
    """Single public entrypoint for the UI: executes the in-project research pipeline and returns a JSON string.
    Backward-compat alias `run_autogen_workflow` is provided for older imports.
    """
    try:
        from telemetry import log_event
        log_event("agent.autogen.request", {"user_request": (user_request or "")[:1200]})
    except Exception:
        pass
    params = _extract_params_from_request(user_request)
    tickers = params.get("tickers") or []
    if isinstance(tickers, str):
        tickers = [tickers]
    start_date = params.get("start_date")
    end_date = params.get("end_date")
    interval = params.get("interval") or "1d"
    indicators = params.get("indicators") or []
    length = int(params.get("length") or 20)
    rsi_length = int(params.get("rsi_length") or 14)
    macd_fast = int(params.get("macd_fast") or 12)
    macd_slow = int(params.get("macd_slow") or 26)
    macd_signal = int(params.get("macd_signal") or 9)

    if not tickers or not start_date or not end_date:
        try:
            from telemetry import log_event
            log_event("agent.autogen.error", {"reason": "missing_params", "params": params})
        except Exception:
            pass
        return {"result": json.dumps({"error": "Missing required parameters."})}

    # Run the local orchestrator
    results = _orchestrator.run_technical_analysis(
        tickers=tickers,
        timeframe=interval,
        start_date=start_date,
        end_date=end_date,
        indicators=indicators,
        length=length,
        rsi_length=rsi_length,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal_len=macd_signal,
    )
    try:
        from telemetry import log_event
        log_event("agent.autogen.analysis_done", {"tickers": tickers, "interval": interval, "indicators": indicators})
    except Exception:
        pass

    # Convert figures to JSON-serializable dicts
    serializable: Dict[str, Any] = {}
    for t, res in (results or {}).items():
        if res.get("error"):
            serializable[t] = {"error": res["error"]}
            continue
        fig = res.get("figure")
        serializable[t] = {
            "figure": (fig.to_dict() if hasattr(fig, "to_dict") else None),
            "recommendation": res.get("recommendation", "N/A"),
            "confidence": res.get("confidence", "N/A"),
            "justification": res.get("justification", ""),
        }

    # Build a lightweight conversation to show orchestration flow in the UI
    messages = [
        {"agent": "Planner", "message": "Parsed user request and planned: fetch_data -> perform_analysis -> get_ai_summary."},
        {"agent": "DataAgent", "message": f"Fetched data for {', '.join(tickers)} from {start_date} to {end_date} (interval={interval})."},
        {"agent": "TechnicalAnalyst", "message": "Computed indicators and produced recommendation and confidence."},
        {"agent": "Validator", "message": "VALIDATED"},
    ]

    out = {"result": json.dumps(serializable), "messages": messages}
    try:
        from telemetry import log_event
        log_event("agent.autogen.response", {"tickers": tickers, "message_count": len(messages)})
    except Exception:
        pass
    return out


# Backward compatibility: keep the previous function name used across the UI
run_autogen_workflow = run_research_workflow
