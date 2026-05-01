"""Streamlit dashboard for asqav audit trails.

Visualizes signed agent activity, replay timelines, and chain verification
in one page. Backed entirely by the asqav SDK and CLI; nothing custom on
the server side.

Run:
    pip install asqav streamlit
    export ASQAV_API_KEY=sk_...
    streamlit run examples/streamlit_dashboard.py

Same data is reachable from the CLI:
    asqav agents list
    asqav replay <agent_id> <session_id>
    asqav compliance export --session <session_id> --output bundle.json
    asqav replay --bundle bundle.json

Closes #56.
"""

from __future__ import annotations

import os

import streamlit as st

import asqav
from asqav.client import Agent, get_session_signatures, list_agents
from asqav.compliance import FRAMEWORKS, export_bundle
from asqav.replay import replay


def _ensure_init() -> None:
    """Init asqav once per session (Streamlit re-runs the script on each interaction)."""
    if st.session_state.get("_asqav_initialized"):
        return
    api_key = os.environ.get("ASQAV_API_KEY")
    if not api_key:
        st.error(
            "ASQAV_API_KEY is not set. Get a key at https://cloud.asqav.com "
            "and re-run with `ASQAV_API_KEY=sk_... streamlit run ...`."
        )
        st.stop()
    asqav.init(api_key=api_key)
    st.session_state["_asqav_initialized"] = True


def _safe_list_agents() -> list[Agent]:
    """Public list_agents() wrapped in a soft-fail for the dashboard."""
    try:
        return list_agents()
    except Exception as exc:
        st.warning(f"Could not list agents: {exc}")
        return []


def _agent_label(agent: Agent) -> str:
    return f"{agent.name} ({agent.agent_id})"


def main() -> None:
    st.set_page_config(page_title="asqav dashboard", layout="wide")
    st.title("asqav governance dashboard")
    st.caption(
        "Live view of agents, signed actions, replay timelines, and "
        "compliance bundles. Every panel below maps 1:1 to a public asqav "
        "Python API and `asqav` CLI command."
    )

    _ensure_init()

    agents = _safe_list_agents()
    if not agents:
        st.info("No agents yet. Create one with `asqav agents create my-agent`.")
        st.stop()

    with st.sidebar:
        st.subheader("Filters")
        agent = st.selectbox(
            "Agent",
            agents,
            format_func=_agent_label,
            key="agent_select",
        )
        session_id = st.text_input(
            "Session ID",
            placeholder="sess_...",
            help="The session whose signatures you want to inspect.",
        )
        framework = st.selectbox(
            "Compliance framework",
            sorted(FRAMEWORKS.keys()),
            index=0,
        )

    if not session_id:
        st.info("Enter a session ID in the sidebar to load its audit trail.")
        st.stop()

    sigs = get_session_signatures(session_id)
    if not sigs:
        st.warning(f"No signatures found for session `{session_id}`.")
        st.stop()

    st.subheader("Signed actions")
    rows = [
        {
            "signature_id": s.signature_id,
            "action_type": s.action_type,
            "algorithm": s.algorithm,
            "signed_at": s.signed_at,
            "verify": s.verification_url,
        }
        for s in sigs
    ]
    st.dataframe(rows, use_container_width=True)

    st.subheader("Replay timeline")
    timeline = replay(agent.agent_id, session_id)
    chain_ok = timeline.verify_chain()

    cols = st.columns(3)
    cols[0].metric("Actions", timeline.total_actions or len(timeline.steps))
    cols[1].metric(
        "Chain integrity",
        "OK" if chain_ok else "BROKEN",
        delta=None if chain_ok else "tampered",
    )
    cols[2].metric("Steps", len(timeline.steps))

    for step in timeline.steps:
        mark = "OK" if step.chain_valid else "BROKEN"
        st.write(
            f"**[{step.index}] {step.action_type}** "
            f"({mark}) - {step.explanation}"
        )

    st.subheader("Compliance bundle")
    if st.button("Export bundle"):
        bundle = export_bundle(sigs, framework=framework)
        st.success(
            f"Exported {bundle.receipt_count} receipts under {framework}. "
            f"Merkle root: `{bundle.merkle_root}`"
        )
        st.download_button(
            "Download bundle JSON",
            bundle.to_json(),
            file_name=f"bundle-{session_id}.json",
            mime="application/json",
        )

    st.caption(
        "Same flows from the terminal: "
        "`asqav replay <agent_id> <session_id>`, "
        "`asqav compliance export --session <id> --output bundle.json`."
    )


if __name__ == "__main__":
    main()
