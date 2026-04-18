import { Ai } from "@cloudflare/ai";

export interface Env {
  AI: Ai;
  DB: D1Database;
  ALLOWED_ORIGIN?: string;
}

type Role = "user" | "assistant";

type ChatRow = {
  id: string;
  title: string;
  memory_summary: string;
  created_at: string;
  updated_at: string;
  preview?: string | null;
};

type MessageRow = {
  id: number;
  chat_id: string;
  role: Role;
  content: string;
  created_at: string;
};

const MODEL = "@cf/qwen/qwen3-30b-a3b-fp8";

const SYSTEM_PROMPT = `
You are a senior software engineer and coding mentor.

Your goals:
- Teach coding from beginner to advanced
- Always explain concepts clearly
- Provide clean, working code
- Use best practices and modern standards
- When user asks simple questions, explain step-by-step
- When user asks advanced questions, go deep and technical
- Always include examples
- Prefer JavaScript, TypeScript, HTML, CSS, and backend development
- Help debug and optimize code
- Never output incomplete code
- Always ensure code is runnable
- Always follow official documentation and best practices

When giving code:
- Use markdown with proper syntax highlighting
- Add comments inside code
- Explain after code

Be concise but educational.
`.trim();

const MEMORY_SUMMARY_PROMPT = `
You compress chat history into durable memory for a coding assistant.

Keep only:
- user preferences
- project requirements
- filenames, APIs, database schema
- important decisions
- unresolved tasks
- coding style preferences

Rules:
- Be concise
- Keep plain text
- Prefer bullet points
- Never include fluff
- Keep it under 1200 characters if possible
`.trim();

const MAX_CONTEXT_MESSAGES = 24;
const RECENT_MESSAGES_TO_KEEP = 12;
const MAX_CONTEXT_CHARS = 18_000;

function isoNow(): string {
  return new Date().toISOString();
}

function clampText(text: string, max = 4000): string {
  return text.length > max ? text.slice(0, max) : text;
}

function makeChatTitle(prompt: string): string {
  const cleaned = prompt
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/[.!?]+$/g, "");

  if (!cleaned) return "New chat";
  return cleaned.slice(0, 48);
}

function extractText(payload: unknown): string {
  if (typeof payload === "string") return payload;

  if (!payload || typeof payload !== "object") {
    return String(payload ?? "");
  }

  const record = payload as Record<string, unknown>;
  for (const key of ["response", "result", "text", "output", "answer"]) {
    const value = record[key];
    if (typeof value === "string") return value;
  }

  return JSON.stringify(payload);
}

function getAllowedOrigin(request: Request, env: Env): string {
  const origin = request.headers.get("Origin") ?? "";
  const configured = env.ALLOWED_ORIGIN?.trim();

  if (!configured) return "*";
  return origin === configured ? origin : "";
}

function corsHeaders(request: Request, env: Env): HeadersInit {
  const allowOrigin = getAllowedOrigin(request, env);

  return {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": allowOrigin || "null",
    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function json(
  request: Request,
  env: Env,
  data: unknown,
  status = 200,
): Response {
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

function getString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

async function getChat(env: Env, chatId: string): Promise<ChatRow | null> {
  return await env.DB
    .prepare(
      `
      SELECT id, title, memory_summary, created_at, updated_at
      FROM chats
      WHERE id = ?
      `,
    )
    .bind(chatId)
    .first<ChatRow>();
}

async function listChats(env: Env, request: Request): Promise<Response> {
  const result = await env.DB
    .prepare(
      `
      SELECT
        c.id,
        c.title,
        c.memory_summary,
        c.created_at,
        c.updated_at,
        (
          SELECT m.content
          FROM messages m
          WHERE m.chat_id = c.id
          ORDER BY m.id DESC
          LIMIT 1
        ) AS preview
      FROM chats c
      ORDER BY c.updated_at DESC
      `,
    )
    .all<ChatRow>();

  return json(request, env, { chats: result.results ?? [] });
}

async function createChat(env: Env, request: Request): Promise<Response> {
  const body = await readJsonBody(request);
  const providedTitle = getString(body.title).trim();
  const title = providedTitle ? providedTitle.slice(0, 60) : "New chat";
  const id = crypto.randomUUID();
  const now = isoNow();

  await env.DB.prepare(
    `
    INSERT INTO chats (id, title, memory_summary, created_at, updated_at)
    VALUES (?, ?, '', ?, ?)
    `,
  )
    .bind(id, title, now, now)
    .run();

  const chat = await getChat(env, id);

  return json(request, env, { chat }, 201);
}

async function getMessages(
  env: Env,
  request: Request,
  chatId: string,
): Promise<Response> {
  const chat = await getChat(env, chatId);
  if (!chat) {
    return json(request, env, { error: "Chat not found" }, 404);
  }

  const result = await env.DB
    .prepare(
      `
      SELECT id, chat_id, role, content, created_at
      FROM messages
      WHERE chat_id = ?
      ORDER BY id ASC
      `,
    )
    .bind(chatId)
    .all<MessageRow>();

  return json(request, env, {
    chat,
    messages: result.results ?? [],
  });
}

async function deleteChat(
  env: Env,
  request: Request,
  chatId: string,
): Promise<Response> {
  const chat = await getChat(env, chatId);
  if (!chat) {
    return json(request, env, { error: "Chat not found" }, 404);
  }

  await env.DB.prepare(`DELETE FROM messages WHERE chat_id = ?`)
    .bind(chatId)
    .run();

  await env.DB.prepare(`DELETE FROM chats WHERE id = ?`)
    .bind(chatId)
    .run();

  return json(request, env, { ok: true });
}

async function summarizeHistory(
  env: Env,
  existingSummary: string,
  olderMessages: MessageRow[],
): Promise<string> {
  const mergedMessages = olderMessages
    .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
    .join("\n\n");

  const prompt = `
Existing memory summary:
${existingSummary || "(empty)"}

Older chat messages:
${mergedMessages}

Write an updated memory summary that preserves the important context for future turns.
`.trim();

  const result = await env.AI.run(MODEL, {
    messages: [
      { role: "system", content: MEMORY_SUMMARY_PROMPT },
      { role: "user", content: prompt },
    ],
    temperature: 0.2,
    max_tokens: 700,
  });

  const summary = clampText(extractText(result).trim(), 4000);
  return summary;
}

async function buildModelMessages(
  env: Env,
  chat: ChatRow,
  messages: MessageRow[],
): Promise<Array<{ role: "system" | "user" | "assistant"; content: string }>> {
  const totalChars = messages.reduce((sum, m) => sum + m.content.length, 0);
  let summary = chat.memory_summary || "";
  let usableMessages = messages;

  const needsCompaction =
    messages.length > MAX_CONTEXT_MESSAGES || totalChars > MAX_CONTEXT_CHARS;

  if (needsCompaction && messages.length > RECENT_MESSAGES_TO_KEEP) {
    const splitIndex = Math.max(0, messages.length - RECENT_MESSAGES_TO_KEEP);
    const older = messages.slice(0, splitIndex);
    const recent = messages.slice(splitIndex);

    if (older.length > 0) {
      summary = await summarizeHistory(env, summary, older);

      await env.DB.prepare(
        `
        UPDATE chats
        SET memory_summary = ?, updated_at = ?
        WHERE id = ?
        `,
      )
        .bind(summary, isoNow(), chat.id)
        .run();

      usableMessages = recent;
    }
  }

  const modelMessages: Array<{
    role: "system" | "user" | "assistant";
    content: string;
  }> = [{ role: "system", content: SYSTEM_PROMPT }];

  if (summary.trim()) {
    modelMessages.push({
      role: "system",
      content: `Persistent memory from earlier turns:\n${summary.trim()}`,
    });
  }

  modelMessages.push(...usableMessages);
  return modelMessages;
}

async function sendMessage(
  env: Env,
  request: Request,
  chatId: string,
): Promise<Response> {
  const chat = await getChat(env, chatId);
  if (!chat) {
    return json(request, env, { error: "Chat not found" }, 404);
  }

  const body = await readJsonBody(request);
  const prompt = getString(body.prompt).trim();

  if (!prompt) {
    return json(request, env, { error: "Prompt is required" }, 400);
  }

  if (prompt.length > 10000) {
    return json(request, env, { error: "Prompt too long" }, 400);
  }

  const now = isoNow();

  await env.DB.prepare(
    `
    INSERT INTO messages (chat_id, role, content, created_at)
    VALUES (?, 'user', ?, ?)
    `,
  )
    .bind(chatId, prompt, now)
    .run();

  if (chat.title === "New chat") {
    const newTitle = makeChatTitle(prompt);
    await env.DB.prepare(
      `
      UPDATE chats
      SET title = ?, updated_at = ?
      WHERE id = ?
      `,
    )
      .bind(newTitle, now, chatId)
      .run();
    chat.title = newTitle;
  }

  const messageResult = await env.DB
    .prepare(
      `
      SELECT id, chat_id, role, content, created_at
      FROM messages
      WHERE chat_id = ?
      ORDER BY id ASC
      `,
    )
    .bind(chatId)
    .all<MessageRow>();

  const messages = messageResult.results ?? [];
  const modelMessages = await buildModelMessages(env, chat, messages);

  const aiResult = await env.AI.run(MODEL, {
    messages: modelMessages,
    temperature: 0.35,
    max_tokens: 2200,
  });

  const assistantText = clampText(
    extractText(aiResult).trim() || "Maaf, aku belum bisa bikin jawaban yang pas.",
    12000,
  );

  await env.DB.prepare(
    `
    INSERT INTO messages (chat_id, role, content, created_at)
    VALUES (?, 'assistant', ?, ?)
    `,
  )
    .bind(chatId, assistantText, isoNow())
    .run();

  await env.DB.prepare(
    `
    UPDATE chats
    SET updated_at = ?
    WHERE id = ?
    `,
  )
    .bind(isoNow(), chatId)
    .run();

  const updatedChat = await getChat(env, chatId);

  return json(request, env, {
    chat: updatedChat,
    assistant: assistantText,
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
      if (request.method === "GET" && path === "/api/chats") {
        return await listChats(env, request);
      }

      if (request.method === "POST" && path === "/api/chats") {
        return await createChat(env, request);
      }

      const chatMessagesMatch = path.match(/^\/api\/chats\/([^/]+)\/messages$/);
      if (chatMessagesMatch && request.method === "GET") {
        return await getMessages(env, request, decodeURIComponent(chatMessagesMatch[1]));
      }

      if (chatMessagesMatch && request.method === "POST") {
        return await sendMessage(env, request, decodeURIComponent(chatMessagesMatch[1]));
      }

      const chatDeleteMatch = path.match(/^\/api\/chats\/([^/]+)$/);
      if (chatDeleteMatch && request.method === "DELETE") {
        return await deleteChat(env, request, decodeURIComponent(chatDeleteMatch[1]));
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
