"""LangChain integration example for asqav.

Demonstrates how to attach AsqavCallbackHandler to a LangChain agent so every
LLM call, tool invocation, and chain step is signed and recorded in the audit
trail.

Requirements:
    pip install asqav langchain langchain-openai langchain-community python-dotenv

Usage:
    ASQAV_API_KEY=sk_... OPENAI_API_KEY=sk-... python examples/langchain_example.py
"""

from __future__ import annotations

import os

from asqav.extras.langchain import AsqavCallbackHandler

try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError as err:
    raise SystemExit(
        "langchain is required for this example.\n"
        "Install with: pip install langchain langchain-openai langchain-community"
    ) from err

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

api_key = os.environ.get("ASQAV_API_KEY", "")
if not api_key:
    raise SystemExit(
        "Set the ASQAV_API_KEY environment variable before running this example.\n"
        "Get your key at https://asqav.com"
    )

handler = AsqavCallbackHandler(agent_name="langchain-demo")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))]

prompt = PromptTemplate.from_template(
    """Answer the question using available tools.

Tools: {tools}
Tool names: {tool_names}

Question: {input}
{agent_scratchpad}"""
)

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = executor.invoke(
    {"input": "What is Python and who created it?"},
    config={"callbacks": [handler]},
)

print("\nResult:")
print(result["output"])
print("\nView the signed audit trail at https://asqav.com/dashboard")
