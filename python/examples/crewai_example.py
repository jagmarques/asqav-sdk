"""CrewAI integration example for asqav.

Demonstrates how to wire AsqavCrewHook into a CrewAI ``Crew`` so every step
and task is signed and recorded in the audit trail. Context is hashed
client-side when ``base_url`` points at *.asqav.com.

Requirements:
    pip install asqav crewai python-dotenv

Usage:
    ASQAV_API_KEY=sk_... python examples/crewai_example.py
"""

from __future__ import annotations

import os

from asqav.extras.crewai import AsqavCrewHook

try:
    from crewai import Agent, Crew, Process, Task
except ImportError as err:
    raise SystemExit(
        "crewai is required for this example.\n"
        "Install with: pip install crewai"
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

hook = AsqavCrewHook(agent_name="crewai-demo")

researcher = Agent(
    role="Researcher",
    goal="Find accurate, recent information on the given topic",
    backstory="You are a thorough researcher who finds reliable sources.",
    verbose=True,
)

writer = Agent(
    role="Writer",
    goal="Turn research into clear, concise summaries",
    backstory="You are a technical writer who explains complex topics simply.",
    verbose=True,
)

research_task = Task(
    description=(
        "Research the current state of AI governance frameworks worldwide. "
        "Focus on the EU AI Act and recent US executive orders."
    ),
    expected_output="A list of key frameworks with brief descriptions.",
    agent=researcher,
)

summary_task = Task(
    description=(
        "Take the research and write a 3-paragraph summary suitable for a "
        "developer audience."
    ),
    expected_output="A clear, jargon-free summary of AI governance frameworks.",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, summary_task],
    process=Process.sequential,
    verbose=True,
    step_callback=hook.step_callback,
    task_callback=hook.task_callback,
)

result = crew.kickoff()

print("\nResult:")
print(result)
print("\nView the signed audit trail at https://asqav.com/dashboard")
