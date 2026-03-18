# Test file for compliance scanner
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
agent = initialize_agent(tools=[], llm=llm, agent="zero-shot-react-description")
result = agent.run("What is the weather?")
print(result)
