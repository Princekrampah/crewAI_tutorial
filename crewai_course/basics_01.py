import os
from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun
from decouple import config

# poetry new crewAI_tutorial --name app
# change the python version to: >=3.10,<3.13
# poetry config virtualenvs.in-project true
# poetry add crewai python-decouple gradio beautifulsoup4
# cd crewAI_tutorial

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")

DDGS_tool = DuckDuckGoSearchRun()


# Define your agents with roles and goals
web_researcher = Agent(
    role='Web Researcher',
    goal='Excellent web researcher, search with for most relevant information',
    backstory="""Your are an expert in web research you have a knack to search the web for most relevant
     and helpfull web content to address many issues. You are very smart at using the web to get 
     most accurate and reliable information.""",
    verbose=True,
    # Agent not allowed to delegate tasks to any other agent
    allow_delegation=False,
    tools=[DDGS_tool]
)


insight_strategist = Agent(
    role='Tech Insight Researcher',
    goal='Craft compelling content on tech advancements in AI',
    backstory="""You are a Insight Researcher. Do step by step. 
        Based on the provided content first identify the list of topics,
        then search internet for each topic one by one
        and finally find insights for each topic one by one.
        Include the insights and sources in the final response""",
    verbose=True,
    allow_delegation=True,
    tools=[DDGS_tool]
)


task1 = Task(
    description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.
  Compile your findings in a detailed report. 
  Make sure to check with the human if the draft is good before returning your Final Answer.
  Your final answer MUST be a full analysis report""",
    agent=web_researcher
)


task2 = Task(
    description="""Using the insights from the web researcher's report, develop an engaging blog
  post that highlights the most significant AI advancements.
  Your post should be informative yet accessible, catering to a tech-savvy audience.
  Aim for a narrative that captures the essence of these breakthroughs and their
  implications for the future. 
  Your final answer MUST be the full blog post of at least 3 paragraphs.""",
    agent=insight_strategist
)


agents = [web_researcher, insight_strategist]

# Instantiate your crew with a sequential process
crew = Crew(
    agents=agents,
    tasks=[task1, task2],
    verbose=2
)

# # Get your crew to work!
result = crew.kickoff()


print(result)
