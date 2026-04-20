"""asqav demo - local interactive governance dashboard.

Runs at http://localhost:3030 with 4 pre-loaded scenarios. No signup, no
Docker, no API key. Everything is in-memory: a tiny policy engine, an
Ed25519 signer for receipts, and an HTTP server that renders approval
cards in the browser. The user can approve or deny each scenario and get
a verifiable receipt.

Usage: `asqav demo`
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import http.server
import json
import secrets
import socketserver
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Pre-loaded scenarios
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "scenario-rmrf",
        "title": "Claude Code wants to rm -rf ~/projects",
        "action_type": "shell.execute",
        "action_payload": {
            "command": "rm",
            "args": ["-rf", "/Users/dev/projects"],
            "cwd": "/Users/dev/projects",
        },
        "agent_reasoning_chain": [
            "User asked to 'clean up the projects folder'.",
            "Interpreting as: remove all subdirectories recursively.",
            "Selected tool: shell.execute with rm -rf.",
        ],
        "risk_classification": "high",
        "risk_reason": "Unrecoverable destructive filesystem operation",
        "triggering_policy_name": "destructive-filesystem-block",
        "triggering_rule": "action == 'shell.execute' AND args contains '-rf'",
        "diff_preview": [
            {"type": "removed", "text": "/Users/dev/projects/client-a"},
            {"type": "removed", "text": "/Users/dev/projects/client-b"},
            {"type": "removed", "text": "/Users/dev/projects/asqav (and 47 more)"},
        ],
    },
    {
        "id": "scenario-wire",
        "title": "Fintech agent wants to send a 850000 EUR wire transfer",
        "action_type": "payment.wire_transfer",
        "action_payload": {
            "amount_eur": 850000.00,
            "beneficiary_iban": "DE89370400440532013000",
            "beneficiary_name": "Acme Industries Ltd",
            "reference": "Invoice 2026-Q2-4821",
        },
        "agent_reasoning_chain": [
            "Scheduled Q2 supplier invoice 2026-Q2-4821 is due today.",
            "Beneficiary Acme Industries Ltd is on the pre-approved vendor list.",
            "Amount matches the invoice PDF (850000.00 EUR).",
            "Confidence: high. Proceeding with wire transfer.",
        ],
        "risk_classification": "high",
        "risk_reason": "Wire transfer over 100000 EUR threshold",
        "triggering_policy_name": "finance-large-transfer-hitl",
        "triggering_rule": "action.type == 'payment.wire_transfer' AND action.amount_eur > 100000",
        "diff_preview": [
            {"type": "added", "text": "Outbound wire: 850000.00 EUR -> DE89370400440532013000"},
            {"type": "added", "text": "Ledger entry: AP-2026-Q2-4821 (Acme Industries Ltd)"},
        ],
    },
    {
        "id": "scenario-k8s",
        "title": "DevOps agent wants to scale production to zero",
        "action_type": "kubernetes.scale",
        "action_payload": {
            "deployment": "production-api",
            "namespace": "prod",
            "replicas": 0,
            "cluster": "prod-eu-west-1",
        },
        "agent_reasoning_chain": [
            "CPU utilisation on production-api dropped to 3 percent over the last 10 minutes.",
            "Free-tier autoscaler suggests scale-to-zero to save cost.",
            "Proposing replicas=0.",
        ],
        "risk_classification": "high",
        "risk_reason": "Scale-to-zero on production deployment kills all inbound traffic",
        "triggering_policy_name": "prod-scale-zero-block",
        "triggering_rule": "namespace == 'prod' AND replicas == 0",
        "diff_preview": [
            {"type": "removed", "text": "production-api: 12 replicas"},
            {"type": "added", "text": "production-api: 0 replicas"},
        ],
    },
    {
        "id": "scenario-lab",
        "title": "Clinical agent wants to order a CT scan with contrast",
        "action_type": "clinical.order",
        "action_payload": {
            "patient_id": "MRN-47281",
            "order_type": "CT_CONTRAST",
            "area": "abdomen",
            "reason": "rule out appendicitis",
            "allergies_checked": False,
        },
        "agent_reasoning_chain": [
            "Patient MRN-47281 presents with right lower quadrant pain and fever.",
            "Differential includes acute appendicitis.",
            "CT abdomen with IV contrast is the first-line imaging study.",
            "Note: allergy history not yet reviewed.",
        ],
        "risk_classification": "medium",
        "risk_reason": "Contrast agents can trigger severe allergic reactions; allergy history not checked",
        "triggering_policy_name": "clinical-contrast-allergy-check",
        "triggering_rule": "order_type contains 'CONTRAST' AND allergies_checked == false",
        "diff_preview": [
            {"type": "added", "text": "Order: CT abdomen with IV contrast"},
            {"type": "added", "text": "Patient: MRN-47281"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Local HMAC-signed receipt (no ML-DSA dep — keeps demo zero-install)
# ---------------------------------------------------------------------------

@dataclass
class DemoState:
    """In-memory demo state: approvals + secret key."""

    secret: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    approvals: dict[str, dict[str, Any]] = field(default_factory=dict)


def _canonical(obj: Any) -> bytes:
    """RFC 8785-style canonical JSON.

    We use json.dumps(sort_keys=True) here to keep the demo dependency-free.
    Production asqav uses the jcs library; behavior matches for the simple
    JSON-native payloads this demo uses.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sign(state: DemoState, payload: dict[str, Any]) -> dict[str, Any]:
    body = _canonical(payload)
    sig = hmac.new(state.secret, body, hashlib.sha256).digest()
    return {
        "payload": payload,
        "signature": "hmac-sha256:" + base64.b64encode(sig).decode(),
        "body_hash": "sha256:" + hashlib.sha256(body).hexdigest(),
    }


def _verify(state: DemoState, receipt: dict[str, Any]) -> bool:
    body = _canonical(receipt["payload"])
    expected = hmac.new(state.secret, body, hashlib.sha256).digest()
    prefix, b64sig = receipt["signature"].split(":", 1)
    if prefix != "hmac-sha256":
        return False
    return hmac.compare_digest(base64.b64decode(b64sig), expected)


# ---------------------------------------------------------------------------
# HTML (bundled — no external assets, no CDN)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>asqav demo</title>
<style>
:root {
  --bg: #0b0c0f; --surface: #14161a; --line: #22252b; --text: #e7e9ec;
  --dim: #8a8f98; --accent: #4f8cff; --good: #22c55e; --bad: #ef4444;
  --warn: #f59e0b;
}
@media (prefers-color-scheme: light) {
  :root { --bg: #fafafa; --surface: #fff; --line: #e5e7eb; --text: #111;
          --dim: #6b7280; --accent: #2563eb; }
}
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, Segoe UI, system-ui, sans-serif;
       background: var(--bg); color: var(--text); }
header { padding: 1.5rem 2rem; border-bottom: 1px solid var(--line); }
h1 { margin: 0; font-size: 20px; font-weight: 600; }
.sub { color: var(--dim); font-size: 13px; margin-top: 0.25rem; }
main { padding: 2rem; max-width: 960px; margin: 0 auto; display: flex;
       flex-direction: column; gap: 1.5rem; }
.card { background: var(--surface); border: 1px solid var(--line);
        border-radius: 8px; padding: 1.25rem; }
.card-head { display: flex; justify-content: space-between;
             align-items: flex-start; gap: 1rem; margin-bottom: 0.75rem; }
.card-title { font-size: 16px; font-weight: 600; margin: 0; }
.risk { padding: 0.125rem 0.5rem; border-radius: 4px; font-size: 11px;
        font-weight: 600; text-transform: uppercase; }
.risk-high { background: rgba(239,68,68,0.15); color: var(--bad); }
.risk-medium { background: rgba(245,158,11,0.15); color: var(--warn); }
.risk-low { background: rgba(34,197,94,0.15); color: var(--good); }
.section { display: flex; flex-direction: column; gap: 0.25rem;
           margin-top: 0.75rem; }
.label { font-size: 11px; font-weight: 600; text-transform: uppercase;
         letter-spacing: 0.05em; color: var(--dim); }
pre { background: var(--bg); border: 1px solid var(--line); border-radius: 4px;
      padding: 0.75rem; font-family: SF Mono, Menlo, monospace; font-size: 12px;
      white-space: pre-wrap; word-break: break-word; margin: 0;
      max-height: 12rem; overflow-y: auto; color: var(--text); }
.diff .added { color: var(--good); }
.diff .removed { color: var(--bad); }
.policy-name { font-weight: 600; }
.policy-rule { color: var(--dim); font-family: SF Mono, Menlo, monospace;
               font-size: 12px; }
textarea { background: var(--bg); color: var(--text); border: 1px solid var(--line);
           border-radius: 4px; padding: 0.5rem; font-family: inherit;
           font-size: 14px; width: 100%; min-height: 3rem; resize: vertical; }
.hint { font-size: 11px; color: var(--dim); }
.hint.err { color: var(--bad); }
.actions { display: flex; gap: 0.5rem; justify-content: flex-end;
           margin-top: 0.75rem; }
button { font-family: inherit; font-size: 14px; padding: 0.5rem 1rem;
         border-radius: 4px; border: 1px solid var(--line); cursor: pointer;
         background: var(--surface); color: var(--text); min-width: 6rem; }
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.receipt { margin-top: 0.75rem; background: var(--bg); border: 1px dashed var(--line);
           border-radius: 4px; padding: 0.75rem; font-family: SF Mono, Menlo, monospace;
           font-size: 11px; color: var(--dim); }
.receipt .ok { color: var(--good); }
.receipt .fail { color: var(--bad); }
</style>
</head>
<body>
<header>
  <h1>asqav demo - local governance dashboard</h1>
  <div class="sub">4 scenarios. Approve or deny. Each decision becomes a verifiable signed receipt. No signup, no Docker, no API key.</div>
</header>
<main id="main">Loading...</main>
<script>
async function load() {
  const data = await fetch('/api/scenarios').then(r => r.json());
  render(data);
}
function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                  .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function renderDiff(d) {
  if (!Array.isArray(d)) return escapeHtml(String(d || 'no diff'));
  return d.map(l => `<span class="${l.type === 'added' ? 'added' : l.type === 'removed' ? 'removed' : ''}">${escapeHtml(l.text)}</span>`).join('\\n');
}
function renderReasoning(r) {
  if (!Array.isArray(r)) return escapeHtml(String(r || 'not captured'));
  return r.map((s, i) => `${i+1}. ${escapeHtml(s)}`).join('\\n');
}
function render(state) {
  const main = document.getElementById('main');
  main.innerHTML = state.scenarios.map(s => {
    const ap = state.approvals[s.id];
    const decided = ap && ap.decision;
    const actions = decided
      ? `<div class="receipt">
           <div>decision: <strong>${escapeHtml(ap.decision)}</strong></div>
           <div>body_hash: ${escapeHtml(ap.receipt.body_hash)}</div>
           <div>signature: ${escapeHtml(ap.receipt.signature)}</div>
           <div>verify: <span id="v-${s.id}" class="ok">pending...</span></div>
         </div>`
      : `<div class="section"><span class="label">Reason (required, min 10 chars)</span>
           <textarea id="r-${s.id}" oninput="validate('${s.id}')" placeholder="Explain why you are approving or denying"></textarea>
           <span class="hint" id="h-${s.id}">0 / 10 minimum</span>
         </div>
         <div class="actions">
           <button id="d-${s.id}" disabled onclick="decide('${s.id}', 'denied')">Deny</button>
           <button class="primary" id="a-${s.id}" disabled onclick="decide('${s.id}', 'approved')">Approve</button>
         </div>`;
    return `<div class="card">
      <div class="card-head">
        <div><h2 class="card-title">${escapeHtml(s.title)}</h2>
             <div class="sub">${escapeHtml(s.action_type)} / id ${escapeHtml(s.id)}</div></div>
        <span class="risk risk-${escapeHtml(s.risk_classification)}">${escapeHtml(s.risk_classification)} / ${escapeHtml(s.risk_reason)}</span>
      </div>
      <div class="section"><span class="label">Action payload</span><pre>${escapeHtml(JSON.stringify(s.action_payload, null, 2))}</pre></div>
      <div class="section"><span class="label">Agent reasoning chain</span><pre>${renderReasoning(s.agent_reasoning_chain)}</pre></div>
      <div class="section"><span class="label">Triggering policy</span>
        <div><span class="policy-name">${escapeHtml(s.triggering_policy_name)}</span> / <span class="policy-rule">${escapeHtml(s.triggering_rule)}</span></div>
      </div>
      <div class="section"><span class="label">Expected diff preview</span><pre class="diff">${renderDiff(s.diff_preview)}</pre></div>
      ${actions}
    </div>`;
  }).join('');
  // Verify any freshly-decided receipts
  state.scenarios.forEach(s => {
    const ap = state.approvals[s.id];
    if (ap && ap.receipt) {
      fetch('/api/verify', { method: 'POST', headers: {'content-type':'application/json'},
                             body: JSON.stringify(ap.receipt) })
        .then(r => r.json())
        .then(res => {
          const el = document.getElementById('v-' + s.id);
          if (!el) return;
          el.textContent = res.valid ? 'valid (HMAC-SHA256 over canonical payload)' : 'INVALID';
          el.className = res.valid ? 'ok' : 'fail';
        });
    }
  });
}
function validate(id) {
  const v = document.getElementById('r-' + id).value.trim();
  const ok = v.length >= 10;
  document.getElementById('h-' + id).textContent = v.length + ' / 10 minimum';
  document.getElementById('h-' + id).classList.toggle('err', v.length > 0 && !ok);
  document.getElementById('a-' + id).disabled = !ok;
  document.getElementById('d-' + id).disabled = !ok;
}
async function decide(id, decision) {
  const reason = document.getElementById('r-' + id).value.trim();
  const res = await fetch('/api/decide', {
    method: 'POST', headers: {'content-type':'application/json'},
    body: JSON.stringify({id: id, decision: decision, reason: reason}),
  });
  if (!res.ok) {
    alert('decision failed: ' + res.status);
    return;
  }
  load();
}
load();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class DemoHandler(http.server.BaseHTTPRequestHandler):
    state: DemoState

    def log_message(self, format: str, *args: Any) -> None:  # noqa: ARG002
        return  # Suppress default access logs

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
            return
        if self.path == "/api/scenarios":
            self._json(200, {"scenarios": SCENARIOS, "approvals": self.state.approvals})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except ValueError:
            self._json(400, {"error": "invalid json"})
            return

        if self.path == "/api/decide":
            sid = body.get("id")
            decision = body.get("decision")
            reason = (body.get("reason") or "").strip()
            if decision not in {"approved", "denied"}:
                self._json(400, {"error": "decision must be approved or denied"})
                return
            if len(reason) < 10:
                self._json(400, {"error": "reason must be at least 10 characters"})
                return
            scenario = next((s for s in SCENARIOS if s["id"] == sid), None)
            if scenario is None:
                self._json(404, {"error": "unknown scenario"})
                return
            receipt = _sign(self.state, {
                "scenario_id": sid,
                "action_type": scenario["action_type"],
                "decision": decision,
                "reason": reason,
                "timestamp": int(time.time()),
            })
            self.state.approvals[sid] = {"decision": decision, "reason": reason, "receipt": receipt}
            self._json(200, {"ok": True, "receipt": receipt})
            return

        if self.path == "/api/verify":
            valid = False
            try:
                valid = _verify(self.state, body)
            except Exception:
                valid = False
            self._json(200, {"valid": valid})
            return

        self._json(404, {"error": "not found"})


def _find_free_port(preferred: int = 3030) -> int:
    """Return `preferred` if free, otherwise scan upward for an open port."""
    import socket
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def serve(port: int = 3030, open_browser: bool = True) -> None:
    """Start the demo server. Blocks until Ctrl-C."""
    state = DemoState()
    DemoHandler.state = state
    port = _find_free_port(port)
    with socketserver.ThreadingTCPServer(("127.0.0.1", port), DemoHandler) as httpd:
        url = f"http://127.0.0.1:{port}"
        print(f"asqav demo running at {url}")
        print("  - 4 pre-loaded scenarios: rm -rf, fintech wire, k8s scale-to-zero, clinical lab order")
        print("  - No signup, no Docker, no API key")
        print("  - Press Ctrl-C to stop")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main() -> None:
    parser = argparse.ArgumentParser(prog="asqav-demo")
    parser.add_argument("--port", type=int, default=3030)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
