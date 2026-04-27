# Asqav TypeScript Examples

Runnable cookbook recipes that show how to wrap popular agent frameworks
with `@asqav/sdk` so every tool execution produces an ML-DSA-signed audit
record.

## Common setup

1. Get an API key at [asqav.com](https://asqav.com).
2. Export it once per shell:

   ```bash
   export ASQAV_API_KEY=sk_...
   export OPENAI_API_KEY=sk-...
   ```

3. From the `typescript/` directory, install the deps for the example you
   want to run (each example lists its own install command at the top of
   the file). Then run with `tsx`:

   ```bash
   npx tsx examples/<file>.ts
   ```

## Recipes

### vercel-ai-sdk.ts

Wraps a Vercel AI SDK `tool({ execute })` so the `execute` body calls
`agent.sign()` before any side effect. Demonstrates `init()`,
`Agent.create()`, `startSession()`, signed tool execution via
`generateText({ model, tools, prompt })`, and `endSession()`.

Install:

```bash
npm install @asqav/sdk ai @ai-sdk/openai zod
```

### langchain-js.ts

Same pattern for LangChain.js. Defines a `tool(async (args) => {...}, { name, description, schema })`, binds it to `ChatOpenAI` with
`.bindTools([refund])`, and signs every invocation through Asqav before
the Stripe side effect runs.

Install:

```bash
npm install @asqav/sdk @langchain/core @langchain/openai zod
```

## Pattern

Both recipes follow the same rule: call `agent.sign({ actionType, context })`
**before** the side effect. The signed intent lands in the Asqav audit
trail even if the downstream API call throws, so you always know what the
agent decided to do.
