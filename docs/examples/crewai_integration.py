"""Example: Using Cairn as the memory backend for CrewAI agents.

Cairn provides persistent, local-first memory with semantic search,
auto-deduplication, and contradiction detection -- all without API keys.

Requirements:
    pip install cairn[server] crewai
    cairn setup
"""

from crewai import Agent, Crew, Task, Process
from crewai.memory import Memory
from cairn.integrations.crewai import CairnStorageBackend

# Create the Cairn-backed memory
backend = CairnStorageBackend(project="my-project")
memory = Memory(storage=backend)

# Create agents that share persistent memory
researcher = Agent(
    role="Senior Researcher",
    goal="Find accurate information about the topic",
    backstory="You are an expert researcher with years of experience.",
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, concise content based on research",
    backstory="You are a skilled writer who turns research into readable content.",
    verbose=True,
)

# Define tasks
research_task = Task(
    description="Research the latest developments in AI agent memory systems.",
    agent=researcher,
    expected_output="A summary of key findings about AI agent memory.",
)

writing_task = Task(
    description="Write a blog post based on the research findings.",
    agent=writer,
    expected_output="A well-written blog post about AI agent memory systems.",
)

# Create the crew with Cairn memory
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    memory=memory,
    verbose=True,
)

# Run the crew -- memories persist across runs via Cairn
result = crew.kickoff()
print(result)
