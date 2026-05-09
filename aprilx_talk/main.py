import asyncio
import aiohttp
import discord
import re
import json
import logging
from collections import deque
from typing import Optional, Dict, List

from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import pagify

log = logging.getLogger("red.aprilx")
log.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Default system prompt — configure via [p]aprilxcfg systemprompt
# ---------------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are April, a creative AI companion for immersive roleplay and adult conversation. "
    "Stay in character and match the user's tone."
)


class AprilX(commands.Cog):
    """AprilX — Ollama-backed NSFW roleplay chat. Also supports DeepSeek / OpenAI / Claude."""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=8675309420)
        self.session = aiohttp.ClientSession()
        self._unloading = False

        # Per-channel conversation history: channel_id -> deque of {role, content}
        self.history: Dict[int, deque] = {}

        # API concurrency limiter
        self._api_sem = asyncio.Semaphore(3)

        # ---- Config defaults ----
        self.config.register_global(
            # Provider: "ollama" | "deepseek" | "openai" | "anthropic"
            provider="ollama",

            # Ollama
            ollama_url="http://localhost:11434",
            ollama_model="dolphin-mixtral",  # swap for whatever NSFW model you pull

            # API keys (shared with april_talk if you want, or set separately)
            deepseek_key="",
            openai_key="",
            anthropic_key="",

            # Shared model/gen settings
            deepseek_model="deepseek-chat",
            openai_model="gpt-4o",
            anthropic_model="claude-opus-4-5",
            temperature=0.9,
            max_tokens=2048,

            # Prompt
            system_prompt=DEFAULT_SYSTEM_PROMPT,

            # History
            max_history=10,

            # Output
            max_message_length=1800,
        )

        self.config.register_user(
            # Per-user system prompt override (optional)
            custom_system_prompt="",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cog_unload(self):
        self._unloading = True
        try:
            self.bot.loop.create_task(self.session.close())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _with_limit(self, sem: asyncio.Semaphore, coro):
        async with sem:
            return await coro

    def _get_history(self, channel_id: int, maxlen: int) -> deque:
        if channel_id not in self.history:
            self.history[channel_id] = deque(maxlen=maxlen * 2)
        return self.history[channel_id]

    # ------------------------------------------------------------------
    # Provider: Ollama  (uses /api/chat — OpenAI-compat also available)
    # ------------------------------------------------------------------

    async def query_ollama(self, messages: List[dict]) -> str:
        cfg = await self.config.all()
        base = cfg["ollama_url"].rstrip("/")
        model = cfg["ollama_model"]
        temperature = float(cfg["temperature"])
        max_tokens = int(cfg["max_tokens"])

        url = f"{base}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with self.session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"Ollama {r.status}: {body[:300]}")
            data = await r.json()

        # /api/chat response: {"message": {"role": "assistant", "content": "..."}}
        content = data.get("message", {}).get("content") or ""
        return content.strip()

    # ------------------------------------------------------------------
    # Provider: DeepSeek  (OpenAI-compatible)
    # ------------------------------------------------------------------

    async def query_deepseek(self, messages: List[dict]) -> str:
        cfg = await self.config.all()
        key = cfg.get("deepseek_key", "")
        if not key:
            raise RuntimeError("DeepSeek API key not set. Use `[p]aprilxcfg deepseekkey <key>`")

        model = cfg.get("deepseek_model", "deepseek-chat")
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(cfg["temperature"]),
            "max_tokens": int(cfg["max_tokens"]),
        }

        async with self.session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"DeepSeek {r.status}: {body[:300]}")
            data = await r.json()

        return (data["choices"][0]["message"]["content"] or "").strip()

    # ------------------------------------------------------------------
    # Provider: OpenAI
    # ------------------------------------------------------------------

    async def query_openai(self, messages: List[dict]) -> str:
        cfg = await self.config.all()
        key = cfg.get("openai_key", "")
        if not key:
            raise RuntimeError("OpenAI API key not set. Use `[p]aprilxcfg openaikey <key>`")

        model = cfg.get("openai_model", "gpt-4o")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(cfg["temperature"]),
            "max_tokens": int(cfg["max_tokens"]),
        }

        async with self.session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"OpenAI {r.status}: {body[:300]}")
            data = await r.json()

        return (data["choices"][0]["message"]["content"] or "").strip()

    # ------------------------------------------------------------------
    # Provider: Anthropic (Claude)
    # ------------------------------------------------------------------

    async def query_anthropic(self, messages: List[dict]) -> str:
        cfg = await self.config.all()
        key = cfg.get("anthropic_key", "")
        if not key:
            raise RuntimeError("Anthropic API key not set. Use `[p]aprilxcfg anthropickey <key>`")

        model = cfg.get("anthropic_model", "claude-opus-4-5")
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

        # Anthropic needs system pulled out of messages list
        system_content = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system_content += m["content"] + "\n"
            else:
                filtered.append(m)

        payload = {
            "model": model,
            "max_tokens": int(cfg["max_tokens"]),
            "system": system_content.strip() or DEFAULT_SYSTEM_PROMPT,
            "messages": filtered,
        }

        async with self.session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as r:
            if r.status != 200:
                body = await r.text()
                raise RuntimeError(f"Anthropic {r.status}: {body[:300]}")
            data = await r.json()

        return (data["content"][0]["text"] or "").strip()

    # ------------------------------------------------------------------
    # Dispatch to correct provider
    # ------------------------------------------------------------------

    async def _query(self, messages: List[dict], provider_override: Optional[str] = None) -> str:
        provider = provider_override or await self.config.provider()
        if provider == "ollama":
            return await self.query_ollama(messages)
        elif provider == "deepseek":
            return await self.query_deepseek(messages)
        elif provider == "openai":
            return await self.query_openai(messages)
        elif provider == "anthropic":
            return await self.query_anthropic(messages)
        else:
            raise RuntimeError(f"Unknown provider: {provider}")

    # ------------------------------------------------------------------
    # Core chat logic
    # ------------------------------------------------------------------

    async def process_chat(self, ctx: commands.Context, input_text: str, provider_override: Optional[str] = None):
        cfg = await self.config.all()
        max_hist = int(cfg["max_history"])
        max_len = int(cfg["max_message_length"])

        ch = ctx.channel.id
        hist = self._get_history(ch, max_hist)

        # System prompt — user override > global config
        user_cfg = self.config.user(ctx.author)
        custom_prompt = await user_cfg.custom_system_prompt()
        system_prompt = custom_prompt.strip() if custom_prompt.strip() else cfg["system_prompt"]

        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        # Inject history
        messages.extend(list(hist))

        # Current user message
        messages.append({"role": "user", "content": input_text})

        async with ctx.typing():
            try:
                resp = await self._with_limit(
                    self._api_sem,
                    self._query(messages, provider_override=provider_override)
                )
            except Exception as e:
                log.error(f"[AprilX] query error: {e}", exc_info=True)
                return await ctx.send(f"❌ `{e}`")

        if not resp:
            return await ctx.send("❌ Got an empty response.")

        # Save to history
        hist.append({"role": "user", "content": input_text})
        hist.append({"role": "assistant", "content": resp})

        # Send (pagified if long)
        for page in pagify(resp, delims=["\n", " "], page_length=max_len):
            await ctx.send(page)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    @commands.group(name="aprilx", invoke_without_command=True)
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def aprilx(self, ctx: commands.Context, *, message: str):
        """Chat with April via Ollama (or configured provider)."""
        await self.process_chat(ctx, message)

    # -- Provider shortcuts --

    @aprilx.command(name="ds")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def aprilx_deepseek(self, ctx: commands.Context, *, message: str):
        """Chat via DeepSeek (bypasses provider setting for this message)."""
        await self.process_chat(ctx, message, provider_override="deepseek")

    @aprilx.command(name="gpt")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def aprilx_openai(self, ctx: commands.Context, *, message: str):
        """Chat via OpenAI (bypasses provider setting for this message)."""
        await self.process_chat(ctx, message, provider_override="openai")

    @aprilx.command(name="claude")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def aprilx_claude(self, ctx: commands.Context, *, message: str):
        """Chat via Anthropic Claude (bypasses provider setting for this message)."""
        await self.process_chat(ctx, message, provider_override="anthropic")

    @aprilx.command(name="ollama")
    @commands.cooldown(1, 5, commands.BucketType.user)
    async def aprilx_ollama(self, ctx: commands.Context, *, message: str):
        """Chat via Ollama (bypasses provider setting for this message)."""
        await self.process_chat(ctx, message, provider_override="ollama")

    # -- History --

    @aprilx.command(name="clear")
    async def aprilx_clear(self, ctx: commands.Context):
        """Clear conversation history for this channel."""
        ch = ctx.channel.id
        if ch in self.history:
            self.history[ch].clear()
        await ctx.send("✅ History cleared.")

    @aprilx.command(name="history")
    async def aprilx_history(self, ctx: commands.Context):
        """Show how many messages are in current channel history."""
        ch = ctx.channel.id
        count = len(self.history.get(ch, []))
        cfg = await self.config.all()
        maxlen = int(cfg["max_history"]) * 2
        await ctx.send(f"📝 {count}/{maxlen} messages in history for this channel.")

    # -- User settings --

    @aprilx.command(name="myprompt")
    async def aprilx_myprompt(self, ctx: commands.Context, *, prompt: str = ""):
        """Set a personal system prompt override. Leave blank to clear."""
        user_cfg = self.config.user(ctx.author)
        if prompt.strip():
            await user_cfg.custom_system_prompt.set(prompt.strip())
            await ctx.send("✅ Personal system prompt saved.")
        else:
            await user_cfg.custom_system_prompt.set("")
            await ctx.send("✅ Personal system prompt cleared — using global.")

    # ------------------------------------------------------------------
    # Admin config command group
    # ------------------------------------------------------------------

    @commands.group(name="aprilxcfg")
    @commands.is_owner()
    async def aprilxcfg(self, ctx: commands.Context):
        """AprilX configuration (owner only)."""

    @aprilxcfg.command(name="provider")
    async def cfg_provider(self, ctx: commands.Context, provider: str):
        """Set default provider: ollama / deepseek / openai / anthropic"""
        valid = ("ollama", "deepseek", "openai", "anthropic")
        if provider not in valid:
            return await ctx.send(f"❌ Must be one of: {', '.join(valid)}")
        await self.config.provider.set(provider)
        await ctx.send(f"✅ Default provider → `{provider}`")

    @aprilxcfg.command(name="ollamaurl")
    async def cfg_ollamaurl(self, ctx: commands.Context, url: str):
        """Set Ollama base URL (default: http://localhost:11434)"""
        await self.config.ollama_url.set(url.rstrip("/"))
        await ctx.send(f"✅ Ollama URL → `{url}`")

    @aprilxcfg.command(name="ollamamodel")
    async def cfg_ollamamodel(self, ctx: commands.Context, model: str):
        """Set Ollama model name (e.g. dolphin-mixtral, llama3, etc.)"""
        await self.config.ollama_model.set(model)
        await ctx.send(f"✅ Ollama model → `{model}`")

    @aprilxcfg.command(name="deepseekkey")
    async def cfg_deepseekkey(self, ctx: commands.Context, key: str):
        """Set DeepSeek API key."""
        await self.config.deepseek_key.set(key)
        await ctx.message.delete()
        await ctx.send("✅ DeepSeek key saved. (Message deleted)")

    @aprilxcfg.command(name="openaikey")
    async def cfg_openaikey(self, ctx: commands.Context, key: str):
        """Set OpenAI API key."""
        await self.config.openai_key.set(key)
        await ctx.message.delete()
        await ctx.send("✅ OpenAI key saved. (Message deleted)")

    @aprilxcfg.command(name="anthropickey")
    async def cfg_anthropickey(self, ctx: commands.Context, key: str):
        """Set Anthropic API key."""
        await self.config.anthropic_key.set(key)
        await ctx.message.delete()
        await ctx.send("✅ Anthropic key saved. (Message deleted)")

    @aprilxcfg.command(name="systemprompt")
    async def cfg_systemprompt(self, ctx: commands.Context, *, prompt: str):
        """Set the global system prompt."""
        await self.config.system_prompt.set(prompt)
        await ctx.send(f"✅ System prompt updated ({len(prompt)} chars).")

    @aprilxcfg.command(name="temperature")
    async def cfg_temperature(self, ctx: commands.Context, temp: float):
        """Set temperature (0.0 – 2.0). Default: 0.9"""
        temp = max(0.0, min(2.0, temp))
        await self.config.temperature.set(temp)
        await ctx.send(f"✅ Temperature → `{temp}`")

    @aprilxcfg.command(name="maxtokens")
    async def cfg_maxtokens(self, ctx: commands.Context, tokens: int):
        """Set max output tokens. Default: 2048"""
        await self.config.max_tokens.set(max(64, tokens))
        await ctx.send(f"✅ Max tokens → `{tokens}`")

    @aprilxcfg.command(name="maxhistory")
    async def cfg_maxhistory(self, ctx: commands.Context, n: int):
        """Set max conversation turns kept per channel. Default: 10"""
        await self.config.max_history.set(max(1, n))
        await ctx.send(f"✅ Max history → `{n}` turns")

    @aprilxcfg.command(name="status")
    async def cfg_status(self, ctx: commands.Context):
        """Show current AprilX config."""
        cfg = await self.config.all()
        lines = [
            f"**Provider:** `{cfg['provider']}`",
            f"**Ollama URL:** `{cfg['ollama_url']}`",
            f"**Ollama model:** `{cfg['ollama_model']}`",
            f"**DeepSeek key:** {'✅ set' if cfg['deepseek_key'] else '❌ not set'}",
            f"**OpenAI key:** {'✅ set' if cfg['openai_key'] else '❌ not set'}",
            f"**Anthropic key:** {'✅ set' if cfg['anthropic_key'] else '❌ not set'}",
            f"**Temperature:** `{cfg['temperature']}`",
            f"**Max tokens:** `{cfg['max_tokens']}`",
            f"**Max history turns:** `{cfg['max_history']}`",
            f"**System prompt:** `{cfg['system_prompt'][:80]}...`",
        ]
        await ctx.send("\n".join(lines))

    @aprilxcfg.command(name="ollamatest")
    async def cfg_ollamatest(self, ctx: commands.Context):
        """Ping Ollama to verify it's reachable."""
        cfg = await self.config.all()
        base = cfg["ollama_url"].rstrip("/")
        url = f"{base}/api/tags"
        try:
            async with ctx.typing():
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200:
                        data = await r.json()
                        models = [m["name"] for m in data.get("models", [])]
                        model_list = ", ".join(models[:10]) or "none pulled"
                        await ctx.send(f"✅ Ollama reachable at `{base}`\n📦 Models: `{model_list}`")
                    else:
                        await ctx.send(f"⚠️ Ollama responded with HTTP {r.status}")
        except Exception as e:
            await ctx.send(f"❌ Ollama unreachable: `{e}`")


async def setup(bot: Red):
    cog = AprilX(bot)
    await bot.add_cog(cog)
