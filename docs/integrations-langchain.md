# LangChain integration (default-on governance)

asqav can sign a governance receipt on every chain, tool, and LLM run of a
[LangChain](https://www.langchain.com) pipeline. One function call turns it on
for the whole process, so governance is default-on rather than a per-invoke
`config={"callbacks": [...]}` flag. Signing is fail-open: a signing outage
never blocks or changes a run.

## How default-on works

LangChain lets a callback handler auto-attach to every run through a configure
hook plus a context variable (the same mechanism its own tracing uses).
`enable_langchain_governance` registers that hook once and binds an
`AsqavCallbackHandler` to the context variable, so LangChain attaches it to
each run for you. The manual `config={"callbacks": [handler]}` path still
works unchanged.

## Python

Install the extra:

```bash
pip install asqav[langchain]
```

Turn it on in one call:

```python
import asqav
from asqav.extras.langchain import enable_langchain_governance

asqav.init("sk_...")
enable_langchain_governance(agent_name="my-langchain-agent")

# No config={"callbacks": [...]} needed; runs below sign receipts.
result = chain.invoke({"query": "what is asqav?"})
```

Every chain step, tool call, and LLM interaction now signs a receipt
(`chain:start`, `tool:start`, `llm:end`, and so on). A step that raises also
signs the matching `:error` receipt and the error propagates unchanged.

`enable_langchain_governance` accepts `api_key`, `agent_name`, `agent_id`, and
`observe` (log intent instead of signing). It returns the handler, which holds
the recorded signatures.

### Manual attach (unchanged)

```python
from asqav.extras.langchain import AsqavCallbackHandler

handler = AsqavCallbackHandler(agent_name="my-langchain-agent")
chain.invoke(input, config={"callbacks": [handler]})
```

## TypeScript

Install the peer:

```bash
npm install @asqav/sdk @langchain/core
```

```ts
import { init } from "@asqav/sdk";
import { enableLangchainGovernance } from "@asqav/sdk/extras/langchain";

init({ apiKey: process.env.ASQAV_API_KEY! });
await enableLangchainGovernance({ agentName: "my-chain" });

// No { callbacks: [...] } needed; runs below sign receipts.
await chain.invoke(input);
```

The context variable is bound to the current async context, so run your
LangChain calls in the same context where you called
`enableLangchainGovernance`.

## Notes

- Additive and non-breaking: importing the module without calling the enable
  function changes nothing, and the manual-callback path is unchanged.
- LangChain is an optional extra (Python) / optional peer (TypeScript). The
  adapter loads it lazily and raises a clear install error only when you call
  the entrypoint without the package present.
- Fail-open: governance signing never raises into the LangChain run path.

Runnable examples: `python/examples/langchain_example.py`,
`typescript/examples/langchain-js.ts`.
