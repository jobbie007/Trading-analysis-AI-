"""
Exploratory non-deterministic research pipeline (web-first multi-agent style).
Planner -> Searcher -> Summarizer -> Categorizer -> Analyst -> Synthesis.
Uses the local OpenAI-compatible endpoint for LLM calls and duckduckgo_search for web.
Returns {"result": json_str, "messages": [...]} compatible with the app.
"""
from __future__ import annotations

import os
import json
import time
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # Optional; we'll fallback to raw text
from duckduckgo_search import DDGS
try:
    # Optional import of quick_research from services for a final fallback
    from services import quick_research as _quick_research
    from services import NewsBridge as _NewsBridge
except Exception:
    _quick_research = None
    _NewsBridge = None


def _llm_chat(messages: List[Dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    base_url = (os.getenv("LOCAL_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()
    model = os.getenv("LOCAL_LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("LOCAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"
    if not base_url:
        return ""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(os.getenv("LOCAL_LLM_TEMPERATURE", str(temperature)) or temperature),
        "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", str(max_tokens)) or max_tokens),
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        r = requests.post(base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload, timeout=40)
        if r.status_code != 200:
            return ""
        data = r.json()
        return (
            data.get("choices", [{}])[0].get("message", {}).get("content")
            or data.get("choices", [{}])[0].get("text", "")
            or ""
        )
    except Exception:
        return ""


# ---- Output sanitation & relevance helpers ----
def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract the first JSON object from an LLM response that may include reasoning or extra text.
    Handles patterns like "<think>...</think> {json}" by scanning for balanced braces.
    """
    if not text:
        return None
    # Fast path: direct JSON
    try:
        j = json.loads(text)
        if isinstance(j, dict):
            return j
    except Exception:
        pass
    # Scan for first balanced { ... }
    start = text.find("{")
    while start != -1 and start < len(text):
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        j = json.loads(chunk)
                        if isinstance(j, dict):
                            return j
                    except Exception:
                        break  # try next '{'
        start = text.find("{", start + 1)
    return None


def _safe_domain(url: Optional[str]) -> str:
    try:
        if not url:
            return ""
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


_FINANCE_KEYWORDS = {
    "stock", "stocks", "share", "shares", "ticker", "nasdaq", "nyse", "earnings", "eps", "revenue",
    "valuation", "p/e", "pe ratio", "price-to-earnings", "p/s", "price-to-sales", "guidance", "dividend",
    "buyback", "10-k", "10q", "sec", "investor", "analyst", "downgrade", "upgrade", "market cap",
}
_BLOCK_KEYWORDS = {
    "pizza", "pizzeria", "lieferando", "lieferzeit", "thefork", "uber eats", "ubereats", "foodora",
    "restaurant", "takeaway.com", "menulog", "grubhub", "doordash", "delivery"
}
_WHITELIST_DOMAINS = {
    "sec.gov", "investor.apple.com", "ir.apple.com", "seekingalpha.com", "bloomberg.com", "reuters.com",
    "wsj.com", "ft.com", "cnbc.com", "marketwatch.com", "morningstar.com", "fool.com", "investopedia.com",
    "finance.yahoo.com", "nasdaq.com", "markets.businessinsider.com", "theverge.com", "arstechnica.com",
}


def _is_relevant(symbol: str, title: str, snippet: str, content: str, url: Optional[str]) -> bool:
    """Heuristic relevance filter for equity research. Keeps clear finance/company items and filters out food-delivery spam, etc."""
    sym = (symbol or "").strip().lower()
    t = (title or "").lower()
    s = (snippet or "").lower()
    c = (content or "").lower()
    dom = _safe_domain(url)

    # Whitelist well-known finance/news domains regardless of keywords
    for d in _WHITELIST_DOMAINS:
        if d in dom:
            return True

    # If obviously food/restaurant related and no finance terms present, drop
    has_block = any(k in t or k in s or k in c or k in dom for k in _BLOCK_KEYWORDS)
    has_fin = any(k in t or k in s or k in c for k in _FINANCE_KEYWORDS)
    mentions_symbol = (sym and (sym in t or sym in s or sym in c))

    if has_block and not has_fin and not mentions_symbol:
        return False

    # Require at least the symbol or a finance keyword to pass for unknown domains
    if dom and dom not in _WHITELIST_DOMAINS:
        if not (mentions_symbol or has_fin):
            return False
    return True


def _score_result(symbol: str, title: str, snippet: str, url: Optional[str]) -> int:
    """Assign a rough relevance score so we can pick the highest-signal items first."""
    sym = (symbol or "").strip().lower()
    t = (title or "").lower(); s = (snippet or "").lower()
    d = _safe_domain(url)
    score = 0
    if any(w in d for w in _WHITELIST_DOMAINS):
        score += 3
    if sym and (sym in t or sym in s):
        score += 2
    if any(k in t or k in s for k in _FINANCE_KEYWORDS):
        score += 2
    if any(k in t or k in s or k in d for k in _BLOCK_KEYWORDS):
        score -= 4
    if not d:
        score -= 1
    return score


def _plan_topics(symbol: str, rounds: int = 1) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
    messages_log: List[Dict[str, str]] = []
    sys = "You are a research planner for public company analysis."
    user = textwrap.dedent(f"""
        We need a web-first research plan for {symbol}.
        1) Propose 6-10 research topics (e.g., valuation, earnings trends, product pipeline, competitive landscape, risks, sentiment, supply chain).
        2) Propose 8-12 diverse search queries that will retrieve high-signal sources (analyst notes, SEC filings, reputable news, expert blogs).
        Return JSON with keys: topics [string], queries [string].
    """)
    res = _llm_chat([
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ], temperature=0.4, max_tokens=1024)
    messages_log.append({"agent": "Planner(Web)", "message": "Planned topics and initial queries."})
    topics: List[str] = []
    queries: List[str] = []
    try:
        j = json.loads(res)
        topics = [str(x) for x in (j.get("topics") or [])][:12]
        queries = [str(x) for x in (j.get("queries") or [])][:16]
    except Exception:
        pass
    # Fallback if LLM missing or returned nothing
    if not queries:
        topics = [
            "Valuation & Multiples",
            "Earnings Trends & Guidance",
            "Product & AI Strategy",
            "Competitive Landscape",
            "Risks & Regulation",
            "Supply Chain & Geopolitics",
            "Capital Allocation (Buybacks/Dividends)",
            "Sentiment & Catalysts",
        ]
        queries = [
            f"{symbol} company analysis fundamentals news",
            f"{symbol} earnings call transcript",
            f"{symbol} investor presentation filetype:pdf",
            f"{symbol} SEC 10-K site:sec.gov",
            f"{symbol} AI strategy",
            f"{symbol} competitive landscape",
            f"{symbol} valuation P/E P/S",
            f"{symbol} risks regulatory antitrust",
            f"{symbol} share buybacks dividends",
        ]
        messages_log.append({"agent": "Planner(Web)", "message": "LLM unavailable; using default topics and queries."})
    # optional refinement rounds (lightweight)
    for i in range(max(0, rounds - 1)):
        if not queries:
            break
        res2 = _llm_chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": f"Initial queries: {queries}. Suggest 4-6 refined queries (JSON list)."},
        ], temperature=0.4, max_tokens=512)
        try:
            q2 = json.loads(res2)
            if isinstance(q2, list):
                queries = (queries + [str(x) for x in q2])[:24]
                messages_log.append({"agent": "Planner(Web)", "message": f"Refined queries (round {i+2})."})
        except Exception:
            pass
    return topics, queries, messages_log


def _web_search_many(queries: List[str], max_results: int = 60) -> Tuple[List[Dict[str, str]], List[str]]:
    """Search many queries with robustness. Returns (results, diagnostics)."""
    seen = set()
    results: List[Dict[str, str]] = []
    diags: List[str] = []
    # Seed with a generic query if planner provided none
    if not queries:
        queries = ["company analysis fundamentals news"]

    # pass 1: direct text search
    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    for r in ddgs.text(q, max_results=10):
                        url = r.get("href") or r.get("url") or r.get("link")
                        title = r.get("title") or r.get("heading") or q
                        snippet = r.get("body") or r.get("snippet") or ""
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        results.append({"title": title[:200], "url": url, "snippet": snippet[:300]})
                        if len(results) >= max_results:
                            diags.append("hit:max")
                            return results, diags
                except Exception:
                    continue
            diags.append(f"pass1:{len(results)}")
    except Exception as e:
        diags.append(f"ddg.text.error:{type(e).__name__}")

    # pass 2: query variations if results are thin
    if len(results) < 5:
        variations: List[str] = []
        for q in queries[:8]:
            variations.extend([
                q.replace("analysis", "overview"),
                q + " site:seekingalpha.com",
                q + " site:investopedia.com",
                q + " latest",
            ])
        try:
            with DDGS() as ddgs:
                for vq in variations:
                    try:
                        for r in ddgs.text(vq, max_results=8):
                            url = r.get("href") or r.get("url") or r.get("link")
                            title = r.get("title") or r.get("heading") or vq
                            snippet = r.get("body") or r.get("snippet") or ""
                            if not url or url in seen:
                                continue
                            seen.add(url)
                            results.append({"title": title[:200], "url": url, "snippet": snippet[:300]})
                            if len(results) >= max_results:
                                diags.append("var:hit:max")
                                return results, diags
                    except Exception:
                        continue
            diags.append(f"var:{len(results)}")
        except Exception:
            diags.append("ddg.text.variation.error")

    # pass 3: ddg news endpoint
    if len(results) == 0:
        try:
            with DDGS() as ddgs:
                for q in queries[:10]:
                    try:
                        for r in ddgs.news(q, max_results=8):
                            url = r.get("url") or r.get("href")
                            title = r.get("title") or q
                            snippet = r.get("body") or r.get("snippet") or ""
                            if not url or url in seen:
                                continue
                            seen.add(url)
                            results.append({"title": title[:200], "url": url, "snippet": snippet[:300]})
                            if len(results) >= max_results:
                                diags.append("news:hit:max")
                                return results, diags
                    except Exception:
                        continue
            diags.append(f"news:{len(results)}")
        except Exception:
            diags.append("ddg.news.error")

    # pass 4: final fallback to quick_research helper
    if len(results) == 0 and _quick_research is not None and queries:
        try:
            alt = _quick_research(queries[0], max_results=min(10, max_results))
            for r in alt:
                url = r.get("url")
                if not url or url in seen:
                    continue
                results.append({
                    "title": (r.get("title") or "")[:200],
                    "url": url,
                    "snippet": (r.get("snippet") or r.get("description") or "")[:300],
                })
            diags.append(f"fallback:{len(results)}")
        except Exception:
            diags.append("fallback.error")

    return results, diags


def _fetch_page(url: str, timeout: int = 10, max_chars: int = 12000) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        if BeautifulSoup is None:
            # Fallback: return truncated raw text
            return r.text[:max_chars]
        soup = BeautifulSoup(r.text, "html.parser")
        # crude text extraction
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            tag.extract()
        text = " ".join((soup.get_text(" ") or "").split())
        return text[:max_chars]
    except Exception:
        return ""


def _summarize_page(symbol: str, title: str, snippet: str, content: str) -> Dict[str, Any]:
    user = textwrap.dedent(f"""
        Company: {symbol}
        Title: {title}
        Snippet: {snippet}
        Content: {content[:6000]}

        Return JSON: {{
          "category": string,  # suggested category name
          "summary": string,   # 2-3 sentence summary
          "signal": string     # bullish, bearish, neutral
        }}
    """)
    res = _llm_chat([
        {"role": "system", "content": "You summarize articles for equity research."},
        {"role": "user", "content": user + "\nReturn only valid JSON. Do not include any reasoning or extra text."},
    ], temperature=0.3, max_tokens=512)
    # Try strict parse, then robust extraction
    j = None
    try:
        j = json.loads(res)
    except Exception:
        j = _extract_json_object(res)
    if isinstance(j, dict):
        sig = str(j.get("signal") or "neutral").strip().lower()
        if sig not in {"bullish", "bearish", "neutral"}:
            sig = "neutral"
        return {
            "category": (j.get("category") or "Uncategorized").strip() or "Uncategorized",
            "summary": (j.get("summary") or snippet or "").strip(),
            "signal": sig,
        }
    # Fallback to snippet/content
    return {"category": "Uncategorized", "summary": (snippet or content[:300] or "").strip(), "signal": "neutral"}


def _executive_brief(symbol: str, items: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Produce a concise, actionable executive brief from summarized items.
    Returns (brief_markdown, recommendation_dict).
    """
    lines = []
    for it in items[:12]:
        lines.append(f"- {it.get('title','')}: {it.get('summary','')[:240]}")
    user = textwrap.dedent(f"""
        Create an executive brief for {symbol}. Use the bullet points below as source material.
        Return JSON only with keys:
        {{
          "brief": string,  # 6-10 crisp bullets: key themes, risks, catalysts, near-term watch items
          "recommendation": {{"recommendation": "Buy|Hold|Sell", "confidence": number, "justification": string}}
        }}

        SOURCE BULLETS:\n{chr(10).join(lines)[:9000]}
    """)
    res = _llm_chat([
        {"role": "system", "content": "You are a senior equity analyst. Return JSON only—no extra text."},
        {"role": "user", "content": user},
    ], temperature=0.3, max_tokens=800)
    j = _extract_json_object(res)
    if isinstance(j, dict):
        brief = (j.get("brief") or "").strip()
        rec = j.get("recommendation") or {}
        return brief, rec
    return "", {}


def _categorize(items: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    # items: list of {title,url,summary,category}
    # Let LLM cluster via names; we’ll just respect category fields here for simplicity.
    cats: Dict[str, List[int]] = {}
    for idx, it in enumerate(items):
        c = (it.get("category") or "Uncategorized").strip() or "Uncategorized"
        cats.setdefault(c, []).append(idx)
    return cats


def _analyze(symbol: str, categories: Dict[str, List[int]], items: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    # Build a compact context
    blocks = []
    for cat, idxs in categories.items():
        lines = []
        for i in idxs[:6]:
            it = items[i]
            lines.append(f"- {it.get('title','')} :: {it.get('summary','')[:300]}")
        blocks.append(f"Category: {cat}\n" + "\n".join(lines))
    user = textwrap.dedent(f"""
        We have categorized sources for {symbol}.
        For each category, extract key points, risks, and opportunities.
        Then produce a final overall assessment and actionable recommendation.
        Return JSON with keys: per_category (map), overall (markdown), recommendation {{recommendation, confidence, justification}}.

        CONTEXT:\n{chr(10).join(blocks)[:14000]}
    """)
    res = _llm_chat([
        {"role": "system", "content": "You are a senior equity research analyst."},
        {"role": "user", "content": user},
    ], temperature=0.4, max_tokens=1536)
    try:
        j = json.loads(res)
    except Exception:
        j = {"overall": "(analysis failed)", "recommendation": {"recommendation": "Hold", "confidence": "N/A", "justification": "Insufficient data"}}
    overall = j.get("overall") or ""
    rec = j.get("recommendation") or {}
    return overall, rec


def run_exploratory_research(user_request: str, *, symbol: str, max_results: int = 60, fetch_pages_count: int = 20, rounds: int = 2) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    t0 = time.time()

    # 1) Plan topics and queries
    topics, queries, logs = _plan_topics(symbol, rounds=max(1, rounds))
    messages.extend(logs)

    # 2) Search the web
    results, diags = _web_search_many(queries, max_results=max_results)
    if len(results) == 0:
        messages.append({"agent": "Searcher(Web)", "message": f"Searched queries -> 0 results. Diagnostics: {';'.join(diags)}"})
        messages.append({"agent": "System", "message": "CRITICAL: Web search returned 0 results. Check search tool or network."})
        # Fallback: use NewsBridge (news APIs/scraping) to seed items if available
        try:
            if _NewsBridge is not None:
                nb = _NewsBridge()
                arts = nb.fetch(symbol, days_back=7, max_articles=min(12, max_results), model_preference="auto", analyze=True)
                for a in arts:
                    results.append({
                        "title": getattr(a, 'title', '') or symbol,
                        "url": getattr(a, 'url', '') or "",
                        "snippet": getattr(a, 'description', '') or getattr(a, 'ai_summary', '') or "",
                    })
                if results:
                    messages.append({"agent": "Searcher(Web)", "message": f"Fallback via NewsBridge -> {len(results)} items."})
        except Exception:
            pass
    else:
        messages.append({"agent": "Searcher(Web)", "message": f"Searched queries -> {len(results)} unique results."})
        if diags:
            messages.append({"agent": "Searcher(Web)", "message": f"Diagnostics: {';'.join(diags)}"})

    # 3) Rank, fetch pages and summarize with relevance filtering
    # Rank first so we fetch the most promising items
    ranked = sorted(results, key=lambda r: _score_result(symbol, r.get("title",""), r.get("snippet",""), r.get("url")), reverse=True)
    items: List[Dict[str, Any]] = []
    kept = 0
    examined = 0
    for r in ranked:
        if kept >= max(0, fetch_pages_count):
            break
        examined += 1
        url = r.get("url")
        content = _fetch_page(url) if url else ""
        title = r.get("title") or "(no title)"
        snippet = r.get("snippet") or ""
        if not _is_relevant(symbol, title, snippet, content, url):
            continue
        s = _summarize_page(symbol, title, snippet, content)
        items.append({
            "title": title,
            "url": url or "#",
            "snippet": snippet,
            "summary": s.get("summary") or "",
            "category": s.get("category") or "Uncategorized",
            "signal": s.get("signal") or "neutral",
        })
        kept += 1
    messages.append({"agent": "Summarizer(Web)", "message": f"Summarized {len(items)} high-signal pages (ranked)."})
    if examined:
        messages.append({"agent": "Filter(Web)", "message": f"Filtered {examined - kept} / {examined} results as irrelevant."})

    # 4) Categorize
    cats = _categorize(items)
    messages.append({"agent": "Categorizer(Web)", "message": f"Created {len(cats)} categories."})

    # 5) Analyze & synthesize
    overall_md, rec = _analyze(symbol, cats, items)
    messages.append({"agent": "Analyst(Web)", "message": "Produced per-category insights and overall assessment."})

    # 6) Executive Brief (concise, actionable)
    brief, rec2 = _executive_brief(symbol, items)
    if brief:
        overall_md = brief
        if isinstance(rec2, dict) and rec2.get("recommendation"):
            rec = rec2
        messages.append({"agent": "ExecutiveBrief(Web)", "message": "Generated concise executive brief."})

    # Build result payload compatible with UI
    news_like = []
    for it in items[:30]:
        news_like.append({
            "title": it.get("title"),
            "url": it.get("url"),
            "description": it.get("snippet"),
            "ai_summary": it.get("summary"),
            "source": it.get("category"),
            "sentiment_label": it.get("signal"),
            "sentiment_score": 0.0,
        })

    payload = {
        "plan": {"analysis_type": "web_multi", "steps": [
            "Planner(Web)", "Searcher(Web)", "Summarizer(Web)", "Categorizer(Web)", "Analyst(Web)"
        ]},
        "news": news_like,
        "news_summary": {"summary": overall_md or ""},
        "holistic": overall_md or "",
        "recommendation": rec or {},
        "web_search": results,
        # figures optional; technical remains empty in this engine
        "figures": {},
    }

    return {"result": json.dumps(payload), "messages": messages}
