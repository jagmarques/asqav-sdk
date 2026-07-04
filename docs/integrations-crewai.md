# CrewAI integration (default-on governance)

asqav can sign a governance receipt on every step and task of a
[CrewAI](https://www.crewai.com) crew. One function call turns it on for a
crew, so governance is default-on rather than manual `step_callback` /
`task_callback` wiring. Signing is fail-open: a signing outage never blocks or
changes a crew run.

## Install

crewAI is not part of the `asqav[crewai]` extra. crewAI hard-requires chromadb,
which has an unpatched critical advisory (CVE-2025-47947), so pulling it into
the extra would drag that into every install. Install crewAI yourself instead:

```bash
pip install asqav
pip install crewai
```

The adapter duck-types the crew object and never imports crewai, so
`from asqav.extras.crewai import enable_crew_governance` works even where
crewAI is not installed. The wiring only runs against a crew you pass in.

## How default-on works

`enable_crew_governance(crew)` sets the crew's `step_callback` and
`task_callback` to an `AsqavCrewHook`. crewAI already calls those after each
agent step and each task, so once wired every step signs a `step:execute`
receipt and every task signs a `task:complete` receipt with no further config.
Any callbacks the crew already has are kept and run first, so this is additive.

## Usage

Turn it on in one call:

```python
import asqav
from asqav.extras.crewai import enable_crew_governance
from crewai import Agent, Crew, Task

asqav.init("sk_...")

crew = Crew(agents=[...], tasks=[...])
enable_crew_governance(crew, agent_name="my-crew")

crew.kickoff()  # every step and task now signs a receipt
```

`enable_crew_governance` accepts `api_key`, `agent_name`, `agent_id`, and
`observe` (log intent instead of signing). It returns the hook, which holds the
recorded signatures.

## Manual attach (unchanged)

The explicit hook path still works if you want to wire callbacks yourself:

```python
from asqav.extras.crewai import AsqavCrewHook

hook = AsqavCrewHook(agent_name="my-crew")
crew = Crew(
    agents=[...],
    tasks=[...],
    step_callback=hook.step_callback,
    task_callback=hook.task_callback,
)
```

## Notes

- Additive and non-breaking: importing the module without calling
  `enable_crew_governance` changes nothing, and the manual-callback path is
  unchanged.
- Duck-typed: no hard crewai import, so the module loads without crewAI present.
- Fail-open: governance signing never raises into the crew run path.

Runnable example: `python/examples/crewai_example.py`.
