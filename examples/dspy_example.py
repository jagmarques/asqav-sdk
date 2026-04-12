"""DSPy integration example for asqav.

Demonstrates how to register AsqavDSPyCallback with DSPy so every module
execution, LM call, and tool invocation is automatically signed and recorded
in the audit trail.

Requirements:
    pip install asqav[dspy]

Usage:
    ASQAV_API_KEY=sk_... python examples/dspy_example.py
"""

from __future__ import annotations

import os

import asqav
from asqav.extras.dspy import AsqavDSPyCallback

try:
    import dspy
except ImportError:
    raise SystemExit(
        "dspy is required for this example.\n"
        "Install with: pip install asqav[dspy]"
    )

# ---------------------------------------------------------------------------
# Initialise asqav
# ---------------------------------------------------------------------------

api_key = os.environ.get("ASQAV_API_KEY", "")
if not api_key:
    raise SystemExit(
        "Set the ASQAV_API_KEY environment variable before running this example.\n"
        "Get your key at https://asqav.com"
    )

asqav.init(api_key=api_key)

# Register the callback once — DSPy calls it automatically for every
# module, LM, and tool invocation from this point on.
dspy.configure(
    lm=dspy.LM("openai/gpt-4o"),
    callbacks=[AsqavDSPyCallback(agent_name="dspy-demo")],
)

# ---------------------------------------------------------------------------
# Define a DSPy program
# ---------------------------------------------------------------------------


class RAGPipeline(dspy.Module):
    """Simple retrieve-then-answer pipeline."""

    def __init__(self) -> None:
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question: str) -> dspy.Prediction:
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)


# ---------------------------------------------------------------------------
# Run the program
# ---------------------------------------------------------------------------

pipeline = RAGPipeline()

print("Running DSPy pipeline with asqav governance...")
result = pipeline(question="What is AI governance and why does it matter?")

print("\nAnswer:")
print(result.answer)
print("\nView the signed audit trail at https://asqav.com/dashboard")
