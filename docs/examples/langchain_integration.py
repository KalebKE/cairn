"""Example: Using Cairn with LangChain / LangGraph agents.

Cairn provides persistent, local-first memory with semantic search.
This example shows how to inject Cairn memories as context into
a LangChain chain.

Requirements:
    pip install cairn[server] langchain-core langchain-openai
    cairn setup
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from cairn.integrations.langchain import CairnMemory

# Initialize Cairn memory
mem = CairnMemory(project="my-project")

# Store some decisions (normally auto-captured by Cairn hooks)
mem.save("We use PostgreSQL for the orders service because we need ACID transactions", event_type="decision")
mem.save("Always use early returns, never nest more than 2 levels", event_type="user_preference")
mem.save("Docker node_modules volume mount shadows container modules - use anonymous volume", event_type="lesson_learned")

# Later: recall relevant memories as context for a chain
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful coding assistant. Relevant context from memory:\n{memory}"),
    ("human", "{input}"),
])

chain = prompt | llm

# Cairn semantically matches "database" to the PostgreSQL decision
context = mem.recall_as_context("database choice for orders")
response = chain.invoke({"input": "What database should I use for the orders service?", "memory": context})
print(response.content)
