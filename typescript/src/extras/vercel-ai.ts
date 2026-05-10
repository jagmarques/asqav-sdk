/**
 * Vercel AI SDK adapter.
 *
 * Wires Asqav signing into Vercel's `experimental_telemetry: { tracer }`
 * hook. Every span the AI SDK opens (generateText, streamText, tool call,
 * embed, etc.) becomes an asqav signature on `span.end()`.
 *
 * Usage:
 *   import { generateText } from "ai";
 *   import { openai } from "@ai-sdk/openai";
 *   import { Agent, init } from "@asqav/sdk";
 *   import { createAsqavExporter } from "@asqav/sdk/extras/vercel-ai";
 *
 *   init({ apiKey: process.env.ASQAV_API_KEY! });
 *   const agent = await Agent.create({ name: "writer" });
 *
 *   const tracer = createAsqavExporter({ agent });
 *   await generateText({
 *     model: openai("gpt-4o"),
 *     prompt: "...",
 *     experimental_telemetry: { isEnabled: true, tracer },
 *   });
 *
 * Source URLs verified:
 *   - https://ai-sdk.dev/docs/ai-sdk-core/telemetry
 *     ("You may provide a `tracer` which must return an OpenTelemetry `Tracer`.")
 *   - https://github.com/vercel/ai/blob/main/packages/otel/src/get-tracer.ts
 *     (signature: `{ isEnabled?: boolean; tracer?: Tracer }` from `@opentelemetry/api`)
 *   - https://github.com/vercel/ai/blob/main/packages/otel/src/noop-tracer.ts
 *     (Tracer surface used: startSpan, startActiveSpan;
 *      Span surface: spanContext, setAttribute, setAttributes, addEvent,
 *      addLink, addLinks, setStatus, updateName, end, isRecording,
 *      recordException)
 *   - https://github.com/vercel/ai/blob/main/packages/otel/src/record-span.ts
 *     (calls tracer.startActiveSpan(name, { attributes }, span => fn(span));
 *      records errors via span.setStatus + span.recordException then span.end)
 */

import type { Agent } from "../index.js";

// Local copies of @opentelemetry/api 1.x types so users without OTel can still import this.

type AttributeValue =
  | string
  | number
  | boolean
  | Array<null | undefined | string>
  | Array<null | undefined | number>
  | Array<null | undefined | boolean>;

interface Attributes {
  [key: string]: AttributeValue | undefined;
}

interface SpanContext {
  traceId: string;
  spanId: string;
  traceFlags: number;
}

interface SpanStatus {
  code: number; // 0 unset, 1 ok, 2 error
  message?: string;
}

export interface Span {
  spanContext(): SpanContext;
  setAttribute(key: string, value: AttributeValue): this;
  setAttributes(attrs: Attributes): this;
  addEvent(name: string, attrs?: Attributes): this;
  addLink(link: unknown): this;
  addLinks(links: unknown[]): this;
  setStatus(status: SpanStatus): this;
  updateName(name: string): this;
  end(endTime?: number): void;
  isRecording(): boolean;
  recordException(exception: unknown, time?: number): this;
}

export interface Tracer {
  startSpan(name: string, options?: unknown, context?: unknown): Span;
  startActiveSpan<F extends (span: Span) => unknown>(
    name: string,
    arg1: F | unknown,
    arg2?: F | unknown,
    arg3?: F,
  ): unknown;
}

// === Action-type mapping ===

/**
 * Map a Vercel AI SDK span name to an asqav action_type.
 *
 * Span names emitted by the AI SDK include `ai.generateText`,
 * `ai.streamText`, `ai.embed`, `ai.embedMany`, `ai.toolCall`,
 * `ai.generateObject`, `ai.streamObject`. The full list lives at
 * https://ai-sdk.dev/docs/ai-sdk-core/telemetry#collected-data.
 */
export function mapSpanNameToActionType(name: string): string {
  if (!name) return "ai:span";
  // Tool calls. AI SDK uses `ai.toolCall` as the span name; tool name
  // travels in attributes (`ai.toolCall.name`).
  if (name === "ai.toolCall" || name.startsWith("ai.toolCall.")) {
    return "ai:tool_call";
  }
  if (name === "ai.generateText" || name.startsWith("ai.generateText.")) {
    return "ai:completion";
  }
  if (name === "ai.streamText" || name.startsWith("ai.streamText.")) {
    return "ai:completion_stream";
  }
  if (name === "ai.generateObject" || name.startsWith("ai.generateObject.")) {
    return "ai:object";
  }
  if (name === "ai.streamObject" || name.startsWith("ai.streamObject.")) {
    return "ai:object_stream";
  }
  if (name === "ai.embed" || name === "ai.embedMany" || name.startsWith("ai.embed")) {
    return "ai:embed";
  }
  // Fallback: replace `ai.foo.bar` -> `ai:foo_bar`.
  if (name.startsWith("ai.")) {
    return `ai:${name.slice(3).replace(/\./g, "_")}`;
  }
  return name;
}

// === Options ===

export interface CreateAsqavExporterOptions {
  /**
   * Pre-built Asqav `Agent`. When provided, `agentId`/`apiKey` are ignored
   * and the adapter signs against this agent directly.
   */
  agent?: Agent;
  /**
   * Asqav API key. Required when `agent` is not supplied; used to lazily
   * attach to the named agent on first span.
   */
  apiKey?: string;
  /** Asqav agent id to attach to when `agent` is not supplied. */
  agentId?: string;
  /** Optional override for the Asqav base URL. */
  baseUrl?: string;
  /**
   * Optional callback invoked when signing fails. Defaults to
   * `console.warn`. The adapter never throws.
   */
  onError?: (err: unknown, span: { name: string; attributes: Attributes }) => void;
}

// === Implementation ===

const noop = (): void => undefined;

interface SpanState {
  name: string;
  attributes: Attributes;
  startTime: number;
  status: SpanStatus;
  exception: unknown;
  ended: boolean;
}

function makeSpan(
  state: SpanState,
  onEnd: (state: SpanState) => void,
): Span {
  const span: Span = {
    spanContext(): SpanContext {
      return { traceId: "", spanId: "", traceFlags: 0 };
    },
    setAttribute(key: string, value: AttributeValue): Span {
      state.attributes[key] = value;
      return span;
    },
    setAttributes(attrs: Attributes): Span {
      Object.assign(state.attributes, attrs);
      return span;
    },
    addEvent(): Span {
      return span;
    },
    addLink(): Span {
      return span;
    },
    addLinks(): Span {
      return span;
    },
    setStatus(status: SpanStatus): Span {
      state.status = status;
      return span;
    },
    updateName(name: string): Span {
      state.name = name;
      return span;
    },
    end(): void {
      if (state.ended) return;
      state.ended = true;
      onEnd(state);
    },
    isRecording(): boolean {
      return !state.ended;
    },
    recordException(exception: unknown): Span {
      state.exception = exception;
      return span;
    },
  };
  return span;
}

/**
 * Build a `Tracer` you can hand to `generateText({ experimental_telemetry: { tracer } })`.
 *
 * Each span the AI SDK opens is buffered locally; on `span.end()` the
 * adapter fires a fire-and-forget `agent.sign(actionType, attributes)`.
 * Network errors are swallowed so signing never breaks generation.
 */
export function createAsqavExporter(opts: CreateAsqavExporterOptions): Tracer {
  if (!opts.agent && !opts.agentId) {
    throw new Error(
      "createAsqavExporter requires either { agent } or { agentId } (and apiKey/init).",
    );
  }

  const onError = opts.onError ?? ((err, span) => {
    // Fail-open: never throw, just warn.
    // eslint-disable-next-line no-console
    console.warn(`[asqav/vercel-ai] sign failed for span ${span.name}:`, err);
  });

  const handleEnd = (state: SpanState): void => {
    if (!opts.agent) {
      // Lazy attach not supported in v1: require pre-built agent.
      onError(
        new Error("createAsqavExporter currently requires a pre-built Agent"),
        { name: state.name, attributes: state.attributes },
      );
      return;
    }

    const isError = state.exception !== undefined || state.status.code === 2;
    const baseAction = mapSpanNameToActionType(state.name);
    const actionType = isError ? "ai:error" : baseAction;
    const durationMs = Math.max(0, Date.now() - state.startTime);

    const context: Record<string, unknown> = {
      ...state.attributes,
      span_name: state.name,
      duration_ms: durationMs,
    };
    if (isError && state.exception !== undefined) {
      context.error =
        state.exception instanceof Error
          ? { name: state.exception.name, message: state.exception.message }
          : String(state.exception);
      context.original_action_type = baseAction;
    }

    // Fire-and-forget. No await, no throw.
    try {
      const promise = opts.agent.sign({ actionType, context });
      Promise.resolve(promise).catch((err) => {
        onError(err, { name: state.name, attributes: state.attributes });
      });
    } catch (err) {
      onError(err, { name: state.name, attributes: state.attributes });
    }
  };

  const tracer: Tracer = {
    startSpan(name: string, options?: unknown): Span {
      const state: SpanState = {
        name,
        attributes:
          options && typeof options === "object" && "attributes" in options
            ? { ...((options as { attributes?: Attributes }).attributes ?? {}) }
            : {},
        startTime: Date.now(),
        status: { code: 0 },
        exception: undefined,
        ended: false,
      };
      return makeSpan(state, handleEnd);
    },
    startActiveSpan<F extends (span: Span) => unknown>(
      name: string,
      arg1: F | unknown,
      arg2?: F | unknown,
      arg3?: F,
    ): unknown {
      // Resolve the (options, context, fn) overload set used by OTel.
      let options: unknown;
      let fn: F | undefined;
      if (typeof arg1 === "function") {
        fn = arg1 as F;
      } else if (typeof arg2 === "function") {
        options = arg1;
        fn = arg2 as F;
      } else if (typeof arg3 === "function") {
        options = arg1;
        fn = arg3;
      }
      const span = tracer.startSpan(name, options);
      if (!fn) return undefined;
      try {
        const result = fn(span);
        if (result && typeof (result as Promise<unknown>).then === "function") {
          return (result as Promise<unknown>).catch((err) => {
            span.recordException(err);
            span.setStatus({ code: 2, message: String(err) });
            span.end();
            throw err;
          }).then((value) => {
            // Only end if the user's fn didn't already call span.end().
            span.end();
            return value;
          });
        }
        span.end();
        return result;
      } catch (err) {
        span.recordException(err);
        span.setStatus({ code: 2, message: String(err) });
        span.end();
        throw err;
      }
    },
  };

  return tracer;
}

// Exported for tests.
export const _internal = { mapSpanNameToActionType, noop };
