"""Agents package.

Contains agent workflows:
- agents.py: deterministic orchestrator shim (run_autogen_workflow)
- agents_team.py: AutoGen team (run_team_workflow)
- lc_agents.py: LangChain-based justification step (run_langchain_workflow)
- nd_agents.py: Exploratory web-first pipeline (run_exploratory_research)
"""

__all__ = [
    "agents",
    "agents_team",
    "lc_agents",
    "nd_agents",
]
