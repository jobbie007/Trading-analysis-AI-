# In dash/workflows.py

from services import Task

WORKFLOWS = {
    "technical": [
        Task(agent_name="data_agent", output_key="price_data"),
        Task(agent_name="technical_strategist_agent", output_key="technical"),
        Task(agent_name="recommendation_agent", output_key="recommendation"),
    ],
    "fundamental": [
        Task(agent_name="company_profile_agent", output_key="company_profile"),
        Task(agent_name="financial_data_agent_full", output_key="fundamentals_raw"),
        Task(agent_name="fundamental_ratios_agent", output_key="ratios"),
        Task(agent_name="fundamental_synthesis_agent", output_key="fundamental_summary"),
        Task(agent_name="recommendation_agent", output_key="recommendation"),
    ],
    "combined": [
        Task(agent_name="company_profile_agent", output_key="company_profile"),
        Task(agent_name="financial_data_agent_full", output_key="fundamentals_raw"),
        Task(agent_name="data_agent", output_key="price_data"), # Price data needed by multiple agents
        Task(agent_name="news_agent", output_key="news"),
        Task(agent_name="fundamental_ratios_agent", output_key="ratios"),
        Task(agent_name="technical_strategist_agent", output_key="technical"),
        Task(agent_name="fundamental_synthesis_agent", output_key="fundamental_summary"),
        Task(agent_name="holistic_analysis_agent", output_key="holistic"),
        Task(agent_name="recommendation_agent", output_key="recommendation"),
    ],
    "web": [
        Task(agent_name="news_agent", output_key="news"),
        Task(agent_name="holistic_analysis_agent", output_key="holistic"),
        Task(agent_name="recommendation_agent", output_key="recommendation"),
    ],
}
