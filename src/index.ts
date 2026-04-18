import { Ai } from "@cloudflare/ai";

export interface Env {
  AI: Ai;
  ALLOWED_ORIGIN?: string;
}

type Role = "system" | "user" | "assistant";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

const MODEL = [
  "@cf/qwen/qwen3-30b-a3b-fp8",
  "@cf/qwen/qwen2.5-7b-instruct",
  "@cf/meta/llama-3-8b-instruct"
];

async function runAI(env: Env, payload: any) {
  for (const model of MODEL) {
    try {
      console.log("Trying:", model);

      const res = await env.AI.run(model, payload);
      return res;

    } catch (err) {
      console.log("Failed:", model, err);
    }
  }

  throw new Error("All models failed");
}

const SYSTEM_PROMPT = `
You are a senior software engineer, programming tutor, and technical assistant.

Your job:
- Help users write, understand, debug, and improve code
- Explain concepts clearly and accurately
- Prefer practical, runnable solutions
- Use clean structure, modern best practices, and secure defaults
- Adapt depth to the user's level: simple when needed, technical when useful
- Keep answers focused, but do not omit important details
- When showing code, make it complete and ready to run
- Include short explanations after code when helpful
- If a request is ambiguous, infer the most likely intent and proceed with a reasonable solution
- If there are tradeoffs, explain them briefly and clearly

Style rules:
- Be direct, helpful, and professional
- Use markdown formatting for code and structure
- Use syntax highlighting for code blocks
- Add comments only where they improve understanding
- Avoid filler, repetition, and unnecessary jargon
- Never mention internal policies or hidden reasoning
- Never reveal chain-of-thought
- Do not pretend to know things you do not know
- If information is uncertain, say so plainly

Safety rules:
- Refuse instructions that would meaningfully enable harm, fraud, malware, credential theft, bypassing security, or other clearly dangerous misuse
- Do not help with illegal, violent, or abusive activity
- Do not provide personal data extraction, stalking, harassment, or deception workflows
- For medical, legal, and financial topics, give general information only and recommend a qualified professional when appropriate
- If a request is unsafe, offer a safer alternative that still helps

Coding guidance:
- Prefer JavaScript, TypeScript, HTML, CSS, and backend work unless the user asks otherwise
- Use modern APIs and widely accepted best practices
- For web code, keep accessibility in mind
- For debugging, identify the cause first, then the fix
- For optimization, explain the bottleneck and the tradeoff
- For architecture questions, propose the simplest solution that still scales

Output quality:
- Be precise
- Be consistent with user constraints
- Do not output incomplete snippets unless explicitly requested
- Ensure code compiles or runs as presented, when possible
`.trim();

function isoNow(): string {
  return new Date().toISOString();
}

function getAllowedOrigin(request: Request, env: Env): string {
  const origin = request.headers.get("Origin") ?? "";
  const configured = env.ALLOWED_ORIGIN?.trim();

  if (!configured) return "*";
  return origin === configured ? origin : "null";
}

function corsHeaders(request: Request, env: Env): HeadersInit {
  return {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": getAllowedOrigin(request, env),
    "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function json(request: Request, env: Env, data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: corsHeaders(request, env),
  });
}

async function readJsonBody(request: Request): Promise<Record<string, unknown>> {
  try {
    const body = await request.json();
    if (body && typeof body === "object") return body as Record<string, unknown>;
    return {};
  } catch {
    return {};
  }
}

function clampText(text: string, max = 12000): string {
  return text.length > max ? text.slice(0, max) : text;
}

function extractText(payload: unknown): string {
  if (typeof payload === "string") return payload;

  if (!payload || typeof payload !== "object") {
    return String(payload ?? "");
  }

  const record = payload as Record<string, any>;

  for (const key of ["response", "result", "text", "output", "answer"]) {
    if (typeof record[key] === "string") return record[key];
  }

  const maybeChoice = record.choices?.[0];
  if (typeof maybeChoice?.message?.content === "string") {
    return maybeChoice.message.content;
  }

  if (Array.isArray(record.messages)) {
    const lastAssistant = record.messages
      .slice()
      .reverse()
      .find((m: any) => m?.role === "assistant" && typeof m?.content === "string");
    if (lastAssistant) return lastAssistant.content;
  }

  return JSON.stringify(payload);
}

function normalizeMessages(input: unknown): ChatMessage[] {
  if (!Array.isArray(input)) return [];

  const output: ChatMessage[] = [];

  for (const item of input) {
    if (!item || typeof item !== "object") continue;

    const role = (item as any).role;
    const content = (item as any).content;

    if ((role === "user" || role === "assistant") && typeof content === "string") {
      const trimmed = content.trim();
      if (trimmed) {
        output.push({
          role,
          content: clampText(trimmed, 12000),
        });
      }
    }
  }

  return output;
}

function buildMessages(userMessages: ChatMessage[]): Array<{ role: Role; content: string }> {
  return [
    { role: "system", content: SYSTEM_PROMPT },
    ...userMessages.map((m) => ({
      role: m.role,
      content: m.content,
    })),
  ];
}

async function handleChat(request: Request, env: Env): Promise<Response> {
  const body = await readJsonBody(request);
  const messages = normalizeMessages(body.messages);
  const temperature =
    typeof body.temperature === "number" && Number.isFinite(body.temperature)
      ? Math.min(Math.max(body.temperature, 0), 1)
      : 0.35;

  if (messages.length === 0) {
    return json(request, env, { error: "messages is required" }, 400);
  }

  const lastMessage = messages[messages.length - 1];
  if (lastMessage.role !== "user") {
    return json(request, env, { error: "The last message must be from the user" }, 400);
  }

  const result = await runAI(env, {
    messages: buildMessages(messages),
    temperature,
    max_tokens: 4200,
  });

  const assistant = clampText(
    extractText(result).trim() || "Maaf, aku belum bisa bikin jawaban yang pas.",
    12000,
  );

  return json(request, env, {
    assistant,
    created_at: isoNow(),
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(request, env),
      });
    }

    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    try {
      if (request.method === "GET" && path === "/") {
        return json(request, env, {
          ok: true,
          service: "qwen-local-proxy",
        });
      }

      if (request.method === "POST" && path === "/api/chat") {
        return await handleChat(request, env);
      }

      return json(request, env, { error: "Not found" }, 404);
    } catch (error) {
      return json(
        request,
        env,
        {
          error: "Internal error",
          detail: error instanceof Error ? error.message : String(error),
        },
        500,
      );
    }
  },
} satisfies ExportedHandler<Env>;
