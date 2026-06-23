# asqav harness hooks for Claude Code

These hooks let **Claude Code itself** fire asqav signing, so enforcement does not depend on the
agent choosing to call a tool. Claude Code runs the hook, not the model.

There are two modes, and the difference between them is the whole point:

- **Mode A (PostToolUse audit)** is the default and what most users want. It is **fail-open**.
- **Mode B (PreToolUse gate)** is opt-in for hard enforcement. It is **fail-closed**.

Read both before you pick one. The honest cost of each is stated below, because a wrong mental
model here is a security problem.

---

## Identity setup (both modes)

The hook signs as a specific agent using two environment variables:

- `ASQAV_API_KEY`: your asqav API key (`sk_live_...` or `sk_test_...`).
- `ASQAV_AGENT_ID`: the agent id the receipts are signed under.

The `asqav hook` command errors with a non-zero exit and a clear message if either is unset, so a
missing key never looks like silent enforcement.

Set them in your shell profile (so Claude Code inherits them), or scope them to the hook process.
For repeated runs prefer a `sk_test_` key, since a `sk_live_` key writes real receipts on your
production org.

```bash
export ASQAV_API_KEY="sk_test_your_key_here"
export ASQAV_AGENT_ID="your-agent-id"
```

> `asqav` is the console entrypoint shipped by the asqav Python package
> (`pip install asqav`). The `hook` subcommand is the harness-hook surface described here.

---

## Disambiguation: `asqav hook` CLI vs `asqav/hooks.py`

These are two different things with similar names. Do not confuse them.

- **`asqav/hooks.py`** is **in-process sign callbacks**, Python `before`/`after` functions that
  run inside your own process around `Agent.sign(...)`. They are a library feature for app code,
  and they fail-open by design. They are **not** harness hooks.
- **`asqav hook` (this CLI)** is the **harness-hook surface**. Claude Code (the harness) executes
  it as a `settings.json` hook on PostToolUse or PreToolUse. This is what this README is about.

If you are wiring Claude Code, you want `asqav hook`. If you are writing Python that calls
`Agent.sign(...)` and want to observe those calls in-process, you want `asqav/hooks.py`.

---

## Mode A: PostToolUse audit (default, recommended): FAIL-OPEN

PostToolUse runs **after the tool already executed**. Claude Code documents PostToolUse as firing
"After a tool call succeeds," and notes that on PostToolUse exit 2 only "Shows stderr to Claude
(tool already ran)". It does **not** block. So this mode can **never** gate an action. The tool
has already run by the time the hook fires.

That makes Mode A **fail-open**, on purpose:

- If `api.asqav.com` is down, slow, or returns an error, **the action already happened and
  proceeds unsigned.** The hook produces no receipt, and Claude Code moves on.
- This is **best-effort evidence, not a gate.** Use it to build an audit trail of what the agent
  did. Do not market it or rely on it as enforcement.

Mode A can run as a local `type: "command"` hook (recommended) or as a `type: "http"` hook that
POSTs the event to your own endpoint. **An `http` hook also fails open**: Claude Code documents
that for HTTP hooks a non-2xx status, a connection failure, or a timeout is a "non-blocking error,
execution continues." So `http` can never be the hard boundary either. It is audit only.

### Mode A snippet: local command (recommended)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "asqav hook posttool"
          }
        ]
      }
    ]
  }
}
```

### Mode A snippet: HTTP (audit only, also fail-open)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "http",
            "url": "https://your-endpoint.example.com/asqav/posttool"
          }
        ]
      }
    ]
  }
}
```

The command reads the PostToolUse event JSON on stdin (`session_id`, `tool_name`, `tool_input`,
`tool_response`) and signs it. Because the action is already done, a failure here costs you a
receipt, not a blocked action.

---

## Mode B: PreToolUse gate (opt-in, hard enforcement): FAIL-CLOSED

PreToolUse runs **before the tool executes** and Claude Code documents that it "Can block it."
The block signal is **exit code 2**: Claude Code documents that exit 2 "means a blocking error"
and for PreToolUse "Blocks the tool call." Claude Code also warns that only exit 2 blocks, exit 1
is treated as a non-blocking error and the action proceeds, so a real gate must `exit 2`.

A PreToolUse deny holds even under `bypassPermissions`. Claude Code evaluates hooks **first** in
the permission flow, before the mode check, and documents that in `bypassPermissions` mode "Hooks
still execute and can block operations if needed." So this gate is not bypassed by loose
permission modes.

That makes Mode B **fail-closed**: `asqav hook pretool` signs a decision receipt and `exit 2` to
**block** when policy denies **or when it cannot reach the signer.** If asqav is unreachable, the
tool call is blocked rather than allowed through unsigned.

The honest cost of a real gate:

- A shell spawn plus network latency sit **on the critical path of every matched tool call**.
- An **unreachable signer blocks work.** That is the price of fail-closed: a down signer stops
  the agent rather than letting unsigned actions through.

### v1 scope of Mode B

v1 ships Mode B as a documented `type: "command"` variant only. The shipped `pretool` script
**signs and fails closed on error**. It does **not** turn on PreToolUse DENY-policy decisions by
default. That firewall behavior (denying specific tool calls by policy) is **founder-gated**, and
exposed only as an opt-in flag, off by default. So out of the box, Mode B enforces "every matched
call must be signable" and not "deny these specific calls."

Use a `type: "command"` hook for Mode B. Do **not** use `type: "http"` as the gate: HTTP hooks
fail open (see Mode A), so they can never be the hard boundary.

### Mode B snippet: local command gate

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "asqav hook pretool"
          }
        ]
      }
    ]
  }
}
```

There is no `tool_response` before execution, so the pretool receipt binds the decision, not a
result.

---

## Why posttool cannot claim a decision (capture_topology honesty)

The receipt each mode writes is labelled honestly, and the asqav cloud enforces that labelling.

- **PostToolUse receipts** are `capture_topology="passive_telemetry"` with
  `receipt_type="protectmcp:observation"` (or `protectmcp:observation:result_bound` when they bind
  the tool result digest). A PostToolUse hook observed the action **after it ran** and never had
  the option to block, so it cannot honestly claim it made a decision. The cloud's
  false-attestation guard rejects any `passive_telemetry` receipt that tries to claim a decision,
  so a posttool hook physically cannot mislabel itself as a gate.
- **PreToolUse receipts** (Mode B) are the only ones allowed to claim a decision. The local gate
  genuinely had the option to block, so it may use `capture_topology="in_process_sdk"` with
  `receipt_type="protectmcp:decision"` and a `policy_decision` of `permit` or `deny`.

The rule of thumb: a hook that ran after the fact records an observation. Only a hook that could
have blocked records a decision.

---

## Cross-harness note (doc only, code deferred)

Claude Code is the supported harness in v1. Other harnesses also expose lifecycle hooks that could
POST the same event to the same asqav sign endpoint:

- **Cursor** supports hooks that fire on tool execution, file edits, and shell commands.
- **OpenAI Codex CLI** exposes `PreToolUse`, `PermissionRequest`, and `PostToolUse` hook events.

Adapter code for Cursor and Codex is **deferred**. v1 ships Claude Code only. The same fail-open
vs fail-closed reasoning applies to those harnesses: a post-execution hook can only audit, and only
a pre-execution hook that blocks on error can gate.
