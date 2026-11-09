"""A Toponymy LLM wrapper that calls Claude through the Anthropic SDK (adapted from sep-map's namer.py).

Toponymy's own AnthropicNamer goes through LiteLLM and always sends `temperature`
with a small `max_tokens` (128 for a topic name). Claude Opus 5.5 rejects sampling
parameters (Sonnet 5.5 rejects non-default ones), and their thinking needs headroom
beyond the visible answer, so this wrapper drops temperature and sets its own
ceiling. Effort is always set explicitly (defaults differ: medium on Opus 5.5, high
on Sonnet 5.5). Refusals fall back server-side (`fallbacks: "default"`, the only
form Sonnet 5.5 accepts, Claude API only), as in stage 06.

Toponymy drives async wrappers with a fresh `asyncio.run()` per stage, so the
semaphore is rebuilt whenever the running event loop changes, and each request
opens and closes its own client inside the loop that runs it. (A client kept
across loops was closed by the garbage collector on an already-closed loop:
harmless "Event loop is closed" tracebacks in the 2026-11-09 Sonnet test.)
Successful responses are cached in
SQLite by model, effort and prompt, so re-running a fit never pays twice for the
same prompt. Failed calls don't stop a Toponymy fit -- it warns "Failed to
generate topic name" and leaves the name empty -- so callers should check for
empty names.
"""

import asyncio
import hashlib
import json
import sqlite3

import anthropic
from toponymy.llm_wrappers import AsyncLLMWrapper

import config

PRICE_PER_MTOK = {  # (input, output) USD per million tokens, claude-api skill 2026-10-06
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
}


class AsyncClaudeNamer(AsyncLLMWrapper):
    # Errors an identical retry can't fix; Toponymy's base class fails fast on these.
    FAIL_FAST_EXCEPTIONS = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
        anthropic.NotFoundError,
    )

    def __init__(
        self,
        model: str = config.NAME_MODEL,
        effort: str = config.NAME_EFFORT,
        max_concurrent_requests: int = config.NAME_WORKERS,
        max_tokens: int = 16000,
        cache_path=config.NAME_CACHE,
        instructions: str = config.NAME_INSTRUCTIONS,
    ):
        self.model = model
        self.instructions = instructions  # appended to every prompt Toponymy sends
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.extra_prompting = ""
        self.callback = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cached": 0}
        self._loop = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, text TEXT)")

    def _semaphore_for_loop(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._semaphore

    def _key(self, system: str | None, user: str, schema: dict | None = None) -> str:
        parts = [self.model, self.effort, system or "", user]
        if schema is not None:  # only when set, so stage 10's cached keys stay valid
            parts.append(json.dumps(schema, sort_keys=True))
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()

    def cost_usd(self) -> float:
        price_in, price_out = PRICE_PER_MTOK[self.model]
        return (self.usage["input_tokens"] * price_in + self.usage["output_tokens"] * price_out) / 1e6

    async def _create(self, system: str | None, user: str, schema: dict | None = None) -> str:
        """One reply's text (cached). With a JSON schema, the reply is structured output: valid JSON for it."""
        key = self._key(system, user, schema)
        row = self.db.execute("SELECT text FROM llm WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.usage["cached"] += 1
            return row[0]

        kwargs = {"system": system} if system else {}
        async with self._semaphore_for_loop(), anthropic.AsyncAnthropic(max_retries=4) as client:
            msg = await client.beta.messages.create(
                model=self.model,
                # Toponymy's per-call max_tokens sizes only the visible answer; it would
                # truncate thinking, so the wrapper sets its own ceiling.
                max_tokens=self.max_tokens,
                output_config={"effort": self.effort}
                | ({"format": {"type": "json_schema", "schema": schema}} if schema else {}),
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": user}],
                **kwargs,
            )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += msg.usage.input_tokens
        self.usage["output_tokens"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {msg.stop_details}")
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no text block (stop_reason={msg.stop_reason})")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, text))
        return text

    async def _call_single_llm(self, prompt, temperature, max_tokens) -> str:
        return await self._create(None, prompt["combined"] + self.instructions)

    async def _call_single_llm_with_system(self, prompt, temperature, max_tokens) -> str:
        return await self._create(prompt["system"], prompt["user"] + self.instructions)
