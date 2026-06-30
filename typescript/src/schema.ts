/** Opt-in context-schema validator for agent.sign() (criterion 328). */

/** Valid type names for built-in field descriptors. */
export type ContextSchemaFieldType =
  | "string"
  | "number"
  | "boolean"
  | "object"
  | "array";

/** Per-field descriptor in a ContextSchema. */
export interface ContextSchemaField {
  /** Expected JavaScript type of the value. */
  type: ContextSchemaFieldType;
  /** When true the field must be present. Defaults to false. */
  required?: boolean;
}

/** Built-in schema: a record of field names to ContextSchemaField descriptors. */
export type ContextSchema = Record<string, ContextSchemaField>;

/** Docs URL stamped on ValidationError instances. */
const SCHEMA_DOCS_URL = "https://asqav.com/docs/structured-receipts";

/** Validate context against the schema; throws ValidationError on first bad field.
 * null/undefined context treated as empty ({}). */
export function validateContextSchema(
  context: Record<string, unknown> | null | undefined,
  schema: ContextSchema,
): void {
  const ctx = context ?? {};

  for (const [field, descriptor] of Object.entries(schema)) {
    const { type: typeName, required = false } = descriptor;

    const validTypes: readonly string[] = [
      "string",
      "number",
      "boolean",
      "object",
      "array",
    ];
    if (!validTypes.includes(typeName)) {
      throw new Error(
        `context_schema_error: unknown type '${typeName}' for field '${field}'; ` +
          `must be one of ${validTypes.join(", ")}`,
      );
    }

    if (!(field in ctx)) {
      if (required) {
        throw new ValidationError(
          `context_schema_error: field '${field}' is required but missing`,
          SCHEMA_DOCS_URL,
        );
      }
      continue;
    }

    const value = ctx[field];

    if (typeName === "number") {
      // bool is rejected: typeof true === "boolean", not "number", but guard explicitly.
      if (typeof value === "boolean") {
        throw new ValidationError(
          `context_schema_error: field '${field}' expected type number, got boolean`,
          SCHEMA_DOCS_URL,
        );
      }
      if (typeof value !== "number") {
        throw new ValidationError(
          `context_schema_error: field '${field}' expected type number, got ${typeof value}`,
          SCHEMA_DOCS_URL,
        );
      }
      continue;
    }

    if (typeName === "array") {
      if (!Array.isArray(value)) {
        throw new ValidationError(
          `context_schema_error: field '${field}' expected type array, got ${typeof value}`,
          SCHEMA_DOCS_URL,
        );
      }
      continue;
    }

    if (typeName === "object") {
      if (typeof value !== "object" || value === null || Array.isArray(value)) {
        const got = Array.isArray(value) ? "array" : value === null ? "null" : typeof value;
        throw new ValidationError(
          `context_schema_error: field '${field}' expected type object, got ${got}`,
          SCHEMA_DOCS_URL,
        );
      }
      continue;
    }

    // "string" | "boolean"
    if (typeof value !== typeName) {
      throw new ValidationError(
        `context_schema_error: field '${field}' expected type ${typeName}, got ${typeof value}`,
        SCHEMA_DOCS_URL,
      );
    }
  }
}

/** Return a new object with keys sorted. Stable order makes sign() deterministic. */
export function normalizeContext(
  context: Record<string, unknown> | null | undefined,
): Record<string, unknown> | undefined {
  if (context == null) return undefined;
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(context).sort()) {
    sorted[key] = context[key];
  }
  return sorted;
}

/** Thrown when context fails the caller-supplied schema. */
export class ValidationError extends Error {
  /** Docs page for the structured-receipts feature. */
  docsUrl: string;

  constructor(message: string, docsUrl = SCHEMA_DOCS_URL) {
    super(message);
    this.name = "ValidationError";
    this.docsUrl = docsUrl;
  }
}
