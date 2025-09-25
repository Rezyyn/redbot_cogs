# april_talk.py
# Full Red cog for April (talk-only)
# - Keeps Config ID = 1398462
# - STRICT localtracks playback to avoid YouTube fallback
# - Fixes _collect_image_attachments missing method
# - Artist-style chatter while generating images
# - Loki-powered memory bootstrap and on-demand recall
# - Sleep mode (reply only to owner when enabled)
# - TTS archive (or delete) after playback
# - Gentle concurrency controls for TTS/image

import asyncio
import aiohttp
import base64
import contextlib
import discord
import json
import logging
import os
import random
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lavalink
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path

log = logging.getLogger("red.april_talk")
log.setLevel(logging.DEBUG)

STYLE_SUFFIX = ", in a futuristic neo-cyberpunk aesthetic"

TENOR_GIFS = {
    "happy": [
        "https://media.giphy.com/media/XbxZ41fWLeRECPsGIJ/giphy.gif",
        "https://media.giphy.com/media/l0HlMG1EX2H38cZeE/giphy.gif",
        "https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif",
    ],
    "thinking": [
        "https://media.giphy.com/media/d3mlE7uhX8KFgEmY/giphy.gif",
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",
        "https://media.giphy.com/media/l0HlUNj5BRuYDLxFm/giphy.gif",
    ],
    "confused": [
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",
        "https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif",
        "https://media.giphy.com/media/3oEjI5VtIhHvK37WYo/giphy.gif",
    ],
    "excited": [
        "https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif",
        "https://media.giphy.com/media/l0HlMURBbyUqF0XQI/giphy.gif",
        "https://media.giphy.com/media/3rgXBOmTlzyFCURutG/giphy.gif",
    ],
    "sad": [
        "https://media.giphy.com/media/OPU6wzx8JrHna/giphy.gif",
        "https://media.giphy.com/media/l1AsyjZ8XLd1V7pUk/giphy.gif",
        "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif",
    ],
}


class AprilTalk(commands.Cog):
    """April Talk — chat, images, TTS (strict localtracks), and Loki-powered memory"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.session = aiohttp.ClientSession()
        self.config = Config.get_conf(self, identifier=1398462)

        # Global config
        self.config.register_global(
            # Models & prompts
            deepseek_key="",
            anthropic_key="",
            openai_key="",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=1024,
            system_prompt=(
                "You are April, a helpful, playful but concise AI assistant for Discord. "
                "Default to useful answers, be kind. If user wants an image, you may craft a vivid prompt."
            ),
            smert_prompt=(
                "You are April in 'smert' mode — witty, sharp, and insightful, with broad knowledge."
            ),
            # TTS & messaging
            tts_enabled=True,
            text_response_when_voice=True,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            max_history=5,
            max_message_length=1800,
            use_gifs=True,
            # Localtracks & archive
            localtracks_root="",           # must match `.audioset localpath` if set; else uses Red default
            localtracks_subdir="april",    # saved under localtracks/<subdir>
            tts_archive_after=90,
            tts_archive_keep=True,
            # Loki
            loki_query_url="",             # e.g. http://host:3100/loki/api/v1/query_range
            loki_token="",
            loki_default_range="48h",
            loki_bootstrap_on_start=True,
            loki_label_app="discord-bot",
            loki_label_event="message",
            # Sleep mode
            sleep_enabled=False,
            sleep_owner_id=165548483128983552,
        )

        # Per-user config
        self.config.register_user(
            smert_mode=False,
            custom_anthropic_key="",
            custom_smert_prompt="",
        )

        # Runtime state
        self._unloading = False
        self._tts_sem = asyncio.Semaphore(2)
        self._img_sem = asyncio.Semaphore(2)

        # Per-channel rolling history (chat memory)
        self._history: Dict[int, deque] = {}
        # Track if loki bootstrap done per channel
        self._loki_bootstrapped: Dict[int, bool] = {}

    # ---------------------- Lavalink helpers ----------------------

    def get_player(self, guild_id: int):
        try:
            if hasattr(lavalink, "get_player"):
                return lavalink.get_player(guild_id)
        except Exception as e:
            log.error("get_player failed: %s", e)
        return None

    def is_player_connected(self, player) -> bool:
        if not player:
            return False
        try:
            if hasattr(player, "is_connected") and callable(player.is_connected):
                return player.is_connected()
            if hasattr(player, "is_connected"):
                return bool(player.is_connected)
            if hasattr(player, "channel_id"):
                return bool(player.channel_id)
            if hasattr(player, "channel"):
                return bool(player.channel)
        except Exception as e:
            log.error("is_player_connected error: %s", e)
        return False

    def get_player_channel_id(self, player):
        if not player:
            return None
        try:
            if hasattr(player, "channel_id"):
                return player.channel_id
            if hasattr(player, "channel") and player.channel:
                return player.channel.id
        except Exception as e:
            log.error("get_player_channel_id error: %s", e)
        return None

    # ---------------------- Utils: text/images/gifs ----------------------

    def style_prompt(self, prompt: str) -> str:
        p = prompt.strip()
        if any(k in p.lower() for k in ["cyberpunk", "synthwave", "futuristic", "sci-fi", "science fiction", "blade runner"]):
            return p
        return p + STYLE_SUFFIX

    def maybe_extract_draw_prompt(self, text: str) -> Optional[str]:
        m = re.search(r"<draw>(.*?)</draw>", text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        raw = re.sub(r"\s+", " ", m.group(1).strip())
        return raw or None

    async def detect_emotion(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ["happy", "great", "awesome", "excellent", "wonderful", "amazing"]):
            return "happy"
        if any(w in t for w in ["think", "consider", "ponder", "wonder", "hmm"]):
            return "thinking"
        if any(w in t for w in ["confused", "don't understand", "what", "huh", "unclear"]):
            return "confused"
        if any(w in t for w in ["excited", "can't wait", "awesome!", "wow", "amazing!"]):
            return "excited"
        if any(w in t for w in ["sad", "sorry", "unfortunately", "regret"]):
            return "sad"
        return None

    def _collect_image_attachments(self, message: discord.Message) -> List[str]:
        """Collect direct image URLs from attachments/embeds to pass as context."""
        urls: List[str] = []
        try:
            for a in getattr(message, "attachments", []) or []:
                ct = (a.content_type or "").lower() if hasattr(a, "content_type") and a.content_type else ""
                if any(x in ct for x in ["image/", "jpg", "jpeg", "png", "webp", "gif"]):
                    urls.append(a.url)
                else:
                    if str(a.filename).lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                        urls.append(a.url)
            for e in getattr(message, "embeds", []) or []:
                if getattr(e, "image", None) and getattr(e.image, "url", None):
                    urls.append(e.image.url)
                if getattr(e, "thumbnail", None) and getattr(e.thumbnail, "url", None):
                    urls.append(e.thumbnail.url)
        except Exception as e:
            log.debug("collect_image_attachments error: %s", e)
        return urls[:8]

    # ---------------------- Commands ----------------------

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april(self, ctx: commands.Context, *, input: str):
        """Main April command"""
        if await self._sleep_gate(ctx):
            return

        cmd = input.strip().lower()
        if cmd == "join":
            return await self.join_voice(ctx)
        if cmd == "leave":
            return await self.leave_voice(ctx)
        if cmd == "clearhistory":
            return await self.clear_history(ctx)
        if cmd == "smert":
            return await self.toggle_smert_mode(ctx)

        image_urls = self._collect_image_attachments(ctx.message)
        await self.process_query(ctx, input, context_images=image_urls)

    @april.command(name="draw")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april_draw(self, ctx: commands.Context, *, prompt: str):
        """Draw with OpenAI Images (keeps context and chats while generating)."""
        if await self._sleep_gate(ctx):
            return

        p = prompt.strip()
        styled_prompt = self.style_prompt(p)

        # Artist chatter via DeepSeek while generating
        try:
            artist_text = await self._artist_chatter(styled_prompt)
        except Exception:
            artist_text = f"Ok—I'll paint this: **{styled_prompt}** (warming up my neon palette…)"

        await ctx.send(artist_text)

        try:
            async with self._img_sem:
                png = await self.generate_openai_image_png(styled_prompt, size="1024x1024")
            file = discord.File(BytesIO(png), filename="april_draw.png")
            await ctx.send(content=f"**Image prompt used:** {styled_prompt}", file=file)
        except Exception as e:
            log.exception("Draw failed")
            await ctx.send(f"⚠️ I couldn't draw that: `{e}`\nUse `[p]aprilcfg openaikey <key>` if needed.")

    @april.command(name="sleep")
    @commands.is_owner()
    async def april_sleep(self, ctx: commands.Context, enabled: bool):
        """Enable/disable sleep mode (only replies to configured owner while sleeping)."""
        await self.config.sleep_enabled.set(bool(enabled))
        await ctx.send("😴 Sleep mode **ON**. I’ll only reply to my owner." if enabled else "🌞 Sleep mode **OFF**. I’m awake!")

    @april.command(name="sleepowner")
    @commands.is_owner()
    async def april_sleep_owner(self, ctx: commands.Context, owner_id: int):
        """Change the user ID who can talk to April while sleeping."""
        await self.config.sleep_owner_id.set(int(owner_id))
        await ctx.send(f"✅ Sleep owner set to `{owner_id}`")

    @april.command(name="join")
    async def join_voice(self, ctx: commands.Context):
        """Join your current voice channel."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a voice channel.")
        channel = ctx.author.voice.channel
        perms = channel.permissions_for(ctx.me)
        if not perms.connect or not perms.speak:
            return await ctx.send("❌ I need permissions to connect and speak!")
        try:
            if not hasattr(lavalink, "get_player"):
                return await ctx.send("❌ Lavalink not initialized. Load Audio with `[p]load audio`")
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                cur = self.get_player_channel_id(player)
                if cur == channel.id:
                    return await ctx.send(f"✅ Already connected to {channel.name}")
                await player.disconnect()
            await lavalink.connect(channel)
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await ctx.send(f"🔊 Joined {channel.name}")
            else:
                await ctx.send("❌ Failed to connect.")
        except Exception as e:
            log.exception("join_voice failed")
            await ctx.send(f"❌ Join failed: {e}")

    @april.command(name="leave")
    async def leave_voice(self, ctx: commands.Context):
        """Leave voice channel."""
        try:
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await player.stop()
                await player.disconnect()
                await ctx.send("👋 Disconnected.")
            else:
                await ctx.send("❌ Not connected.")
        except Exception as e:
            await ctx.send(f"❌ Disconnect failed: {e}")

    @april.command(name="clearhistory")
    async def clear_history(self, ctx: commands.Context):
        """Clear channel memory (not Loki)."""
        cid = ctx.channel.id
        if cid in self._history:
            self._history[cid].clear()
        await ctx.send("🧼 Cleared conversation memory for this channel.")

    @april.command(name="smert")
    async def toggle_smert_mode(self, ctx: commands.Context):
        """Toggle 'smert' mode (uses Anthropic if configured)."""
        ucfg = self.config.user(ctx.author)
        mode = await ucfg.smert_mode()
        if not mode:
            custom_key = await ucfg.custom_anthropic_key()
            global_key = await self.config.anthropic_key()
            if not custom_key and not global_key:
                return await ctx.send(
                    "❌ Smert mode needs an Anthropic API key. "
                    "Set with `[p]aprilcfg anthropickey <key>` or `[p]apriluser setkey <key>`."
                )
            await ucfg.smert_mode.set(True)
            await ctx.send("🧠 Smert mode ON (Claude engaged).")
        else:
            await ucfg.smert_mode.set(False)
            await ctx.send("💡 Smert mode OFF.")

    # ---------------------- Config group ----------------------

    @commands.group(name="aprilcfg")
    @commands.is_owner()
    async def aprilcfg(self, ctx: commands.Context):
        """Configure AprilTalk."""
        if ctx.invoked_subcommand is None:
            await self._show_settings(ctx)

    @aprilcfg.command(name="deepseekkey")
    async def cfg_deepseek(self, ctx: commands.Context, key: str):
        await self.config.deepseek_key.set(key)
        await ctx.tick()
        with contextlib.suppress(Exception):
            await ctx.message.delete()

    @aprilcfg.command(name="anthropickey")
    async def cfg_anthropic(self, ctx: commands.Context, key: str):
        await self.config.anthropic_key.set(key)
        await ctx.tick()
        with contextlib.suppress(Exception):
            await ctx.message.delete()

    @aprilcfg.command(name="openaikey")
    async def cfg_openai(self, ctx: commands.Context, key: str):
        await self.config.openai_key.set(key)
        await ctx.tick()
        with contextlib.suppress(Exception):
            await ctx.message.delete()

    @aprilcfg.command(name="tts")
    async def cfg_tts(self, ctx: commands.Context, enabled: bool):
        await self.config.tts_enabled.set(bool(enabled))
        await ctx.send(f"✅ TTS {'enabled' if enabled else 'disabled'}")

    @aprilcfg.command(name="voice")
    async def cfg_voice(self, ctx: commands.Context, voice_id: str):
        await self.config.voice_id.set(voice_id)
        await ctx.send(f"✅ ElevenLabs voice set to `{voice_id}`")

    @aprilcfg.command(name="model")
    async def cfg_model(self, ctx: commands.Context, model: str):
        await self.config.model.set(model)
        await ctx.send(f"✅ Chat model set to `{model}`")

    @aprilcfg.command(name="localtracks")
    async def cfg_localtracks(self, ctx: commands.Context, root: str, subdir: Optional[str] = "april"):
        await self.config.localtracks_root.set(root)
        await self.config.localtracks_subdir.set(subdir or "april")
        await ctx.send(f"✅ localtracks root=`{root}` subdir=`{subdir or 'april'}`")

    @aprilcfg.command(name="loki")
    async def cfg_loki(self, ctx: commands.Context, url: str, token: str = ""):
        await self.config.loki_query_url.set(url)
        await self.config.loki_token.set(token)
        await ctx.send(f"✅ Loki query configured to `{url}` (token={'set' if token else 'unset'})")

    @aprilcfg.command(name="ttsarchive")
    async def cfg_ttsarchive(self, ctx: commands.Context, keep: bool = True, after_seconds: int = 90):
        await self.config.tts_archive_keep.set(bool(keep))
        await self.config.tts_archive_after.set(int(max(5, after_seconds)))
        await ctx.send(f"✅ After playback I will {'archive' if keep else 'delete'} TTS files (delay {after_seconds}s).")

    @aprilcfg.command(name="settings")
    async def _show_settings(self, ctx: commands.Context):
        cfg = await self.config.all()
        e = discord.Embed(title="AprilTalk Settings", color=await ctx.embed_color())
        for k in [
            "model", "temperature", "max_tokens", "max_history", "max_message_length",
            "tts_enabled", "voice_id", "use_gifs", "localtracks_root", "localtracks_subdir",
            "tts_archive_after", "tts_archive_keep", "loki_query_url", "loki_default_range",
            "loki_bootstrap_on_start", "sleep_enabled", "sleep_owner_id",
        ]:
            v = cfg.get(k)
            e.add_field(name=k, value=f"`{v}`", inline=True)
        await ctx.send(embed=e)

    # ---------------------- Core flow ----------------------

    async def process_query(self, ctx: commands.Context, input_text: str, *, context_images: Optional[List[str]] = None):
        # TTS?
        use_voice = False
        if await self.config.tts_enabled():
            try:
                player = self.get_player(ctx.guild.id)
                use_voice = player and self.is_player_connected(player)
            except Exception:
                use_voice = False

        cid = ctx.channel.id
        if cid not in self._history:
            max_hist = await self.config.max_history()
            self._history[cid] = deque(maxlen=max_hist * 2)

        # Bootstrap from Loki only once per channel
        if not self._loki_bootstrapped.get(cid) and (await self.config.loki_bootstrap_on_start()):
            self._loki_bootstrapped[cid] = True
            asyncio.create_task(self._bootstrap_loki_history(ctx))

        # Build messages
        ucfg = self.config.user(ctx.author)
        smert = await ucfg.smert_mode()
        if smert:
            sys_prompt = (await ucfg.custom_smert_prompt()) or (await self.config.smert_prompt())
        else:
            sys_prompt = await self.config.system_prompt()

        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(self._history[cid])

        user_content = input_text
        if context_images:
            imgs_text = "\n".join(f"[image]: {u}" for u in context_images)
            user_content = f"{input_text}\n\nAttached images:\n{imgs_text}"

        messages.append({"role": "user", "content": user_content})

        # LLM call
        async with ctx.typing():
            try:
                if smert:
                    resp = await self.query_anthropic(ctx.author.id, messages)
                else:
                    resp = await self.query_deepseek(ctx.author.id, messages)
            except Exception as e:
                log.exception("LLM error")
                return await ctx.send(f"❌ Error: {e}")

        draw_prompt = self.maybe_extract_draw_prompt(resp)
        clean_resp = re.sub(r"<draw>.*?</draw>", "", resp, flags=re.IGNORECASE | re.DOTALL).strip()

        # Update memory
        self._history[cid].append({"role": "user", "content": input_text})
        self._history[cid].append({"role": "assistant", "content": clean_resp})

        tasks = []
        if not (use_voice and not await self.config.text_response_when_voice()):
            tasks.append(asyncio.create_task(self.send_streamed_response(ctx, clean_resp)))
        if use_voice:
            tasks.append(asyncio.create_task(self.speak_response(ctx, clean_resp)))

        if draw_prompt:
            styled = self.style_prompt(draw_prompt)

            async def _draw_flow():
                try:
                    chatter = await self._artist_chatter(styled)
                except Exception:
                    chatter = f"Let me sketch this idea: **{styled}** …"
                await ctx.send(chatter)
                try:
                    async with self._img_sem:
                        png = await self.generate_openai_image_png(styled, size="1024x1024")
                    file = discord.File(BytesIO(png), filename="april_draw.png")
                    await ctx.send(content=f"**Image prompt used:** {styled}", file=file)
                except Exception as e:
                    await ctx.send(f"⚠️ Couldn't render the suggested image: `{e}`")

            tasks.append(asyncio.create_task(_draw_flow()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_streamed_response(self, ctx: commands.Context, resp: str):
        max_len = await self.config.max_message_length()
        use_gifs = await self.config.use_gifs()
        emotion = await self.detect_emotion(resp) if use_gifs else None
        gif_url = random.choice(TENOR_GIFS.get(emotion, [])) if (emotion and random.random() < 0.3) else None

        if len(resp) <= max_len:
            embed = discord.Embed(description="💭 *Thinking...*", color=await ctx.embed_color())
            msg = await ctx.send(embed=embed)
            chunk_size = 50
            for i in range(0, len(resp), chunk_size):
                chunk = resp[: i + chunk_size]
                if i + chunk_size >= len(resp) and gif_url:
                    await msg.edit(content=chunk + f"\n{gif_url}", embed=None)
                else:
                    await msg.edit(content=chunk + "▌", embed=None)
                await asyncio.sleep(0.05)
            if not gif_url:
                await msg.edit(content=resp, embed=None)
            return

        embed = discord.Embed(description="💭 *Thinking...*", color=await ctx.embed_color())
        thinking = await ctx.send(embed=embed)
        words = resp.split(" ")
        chunks: List[str] = []
        cur, cur_len = [], 0
        for w in words:
            if cur_len + len(w) + 1 > max_len:
                chunks.append(" ".join(cur))
                cur, cur_len = [w], len(w)
            else:
                cur.append(w)
                cur_len += len(w) + 1
        if cur:
            chunks.append(" ".join(cur))
        await thinking.edit(content=chunks[0], embed=None)
        for i, ch in enumerate(chunks[1:], 1):
            await asyncio.sleep(0.4)
            if gif_url and i == len(chunks) - 1:
                await ctx.send(ch + f"\n{gif_url}")
            else:
                await ctx.send(ch)

    # ---------------------- TTS (STRICT localtracks) ----------------------

    def clean_text_for_tts(self, text: str) -> str:
        t = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        t = re.sub(r"\*(.*?)\*", r"\1", t)
        t = re.sub(r"`(.*?)`", r"\1", t)
        t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)
        t = re.sub(r"http[s]?://\S+", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) > 1000:
            t = t[:1000].rsplit(" ", 1)[0] + "..."
        return t

    async def generate_tts_audio(self, text: str, api_key: str) -> Optional[bytes]:
        try:
            voice_id = await self.config.voice_id()
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
            headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
            async with self.session.post(url, json=payload, headers=headers, timeout=45) as r:
                if r.status == 200:
                    return await r.read()
                err = await r.text()
                log.error("TTS API error %s: %s", r.status, err)
        except Exception as e:
            log.error("TTS generation failed: %s", e)
        return None

    def _resolve_localtracks_paths(self) -> Tuple[Path, str]:
        """
        Returns (outdir_path, rel_prefix) where rel_prefix is what commands expect.
        If localtracks_root not configured, use Red's default: data/Audio/localtracks
        """
        root = (self.bot.loop.run_until_complete(self.config.localtracks_root())  # careful in sync context
                if asyncio.get_event_loop().is_running() is False else None)
        # The above trick is risky; instead, return placeholders; speak_response will call an async version.
        # We'll never call this sync helper in practice; kept for completeness.
        base = cog_data_path(self).parent / "Audio" / "localtracks"
        outdir = base / "april"
        rel = "localtracks/april"
        return outdir, rel

    async def _get_localtracks_dirs(self) -> Tuple[Path, str]:
        root = (await self.config.localtracks_root()).strip()
        subdir = (await self.config.localtracks_subdir()).strip() or "april"
        if not root:
            base = cog_data_path(self).parent / "Audio" / "localtracks"
            outdir = base / subdir
            rel = f"localtracks/{subdir}"
        else:
            outdir = Path(root) / subdir
            rel = f"localtracks/{subdir}"
        return outdir, rel

    async def _ensure_localtracks_registered(self, ctx: commands.Context) -> bool:
        audio_cog = self.bot.get_cog("Audio")
        if not audio_cog:
            return False
        # Prefer callable APIs
        try:
            if hasattr(audio_cog, "localtracks_scan"):
                await audio_cog.localtracks_scan(ctx)
                return True
            if hasattr(audio_cog, "command_localtracks_scan"):
                await audio_cog.command_localtracks_scan(ctx)
                return True
        except Exception:
            pass
        # Fallback to command
        try:
            cmd = self.bot.get_command("localtracks scan")
            if cmd:
                await ctx.invoke(cmd)
                return True
        except Exception:
            pass
        return False

    async def _play_localtracks_strict(self, ctx: commands.Context, rel_path: str) -> bool:
        """
        Try ONLY localtracks play variants; NEVER generic play (which may hit YT).
        Returns True if invoked, False if no suitable command found.
        """
        # 1) localtracks play <rel_path>
        cmd = self.bot.get_command("localtracks play")
        if cmd:
            await ctx.invoke(cmd, path=rel_path)
            return True
        # 2) local play <rel_path> (some Audio forks)
        cmd = self.bot.get_command("local play")
        if cmd:
            await ctx.invoke(cmd, path=rel_path)
            return True
        # 3) audio localtracks play (namespaced)
        cmd = self.bot.get_command("audio localtracks play")
        if cmd:
            await ctx.invoke(cmd, path=rel_path)
            return True
        # If none of the above exist, do NOT call generic "play"
        return False

    async def _archive_or_delete(self, fpath: Path):
        try:
            cfg = await self.config.all()
            delay = int(cfg.get("tts_archive_after", 90))
            keep = bool(cfg.get("tts_archive_keep", True))
            await asyncio.sleep(max(5, delay))
            if not fpath.exists():
                return
            if keep:
                archive_dir = fpath.parent / "archive"
                archive_dir.mkdir(parents=True, exist_ok=True)
                target = archive_dir / fpath.name
                try:
                    fpath.replace(target)
                except Exception:
                    import shutil
                    shutil.copy2(fpath, target)
                    with contextlib.suppress(Exception):
                        fpath.unlink()
            else:
                with contextlib.suppress(Exception):
                    fpath.unlink()
        except Exception:
            pass

    async def speak_response(self, ctx: commands.Context, text: str):
        # must be connected
        player = self.get_player(ctx.guild.id)
        if not self.is_player_connected(player):
            return

        tts_key = await self.config.tts_key()
        if not tts_key:
            return

        clean = self.clean_text_for_tts(text)
        if not clean:
            return

        async with self._tts_sem:
            audio = await self.generate_tts_audio(clean, tts_key)
        if not audio:
            with contextlib.suppress(Exception):
                await ctx.send("⚠️ TTS failed to generate audio.")
            return

        outdir, rel_prefix = await self._get_localtracks_dirs()
        outdir.mkdir(parents=True, exist_ok=True)
        fname = f"tts_{int(time.time())}_{random.randint(1000,9999)}.mp3"
        fpath = outdir / fname
        try:
            with open(fpath, "wb") as f:
                f.write(audio)
        except Exception as e:
            with contextlib.suppress(Exception):
                await ctx.send(f"❌ Failed to write TTS file: `{e}`")
            return

        # Ensure localtracks DB is aware of the new file
        await self._ensure_localtracks_registered(ctx)

        rel_path = f"{rel_prefix}/{fname}"

        # STRICT: Attempt only localtracks play forms. Never generic "play".
        invoked = await self._play_localtracks_strict(ctx, rel_path)
        if not invoked:
            # As a last resort, show the exact command so the user can run it without YT risk.
            with contextlib.suppress(Exception):
                await ctx.send(f"▶️ Your Audio cog doesn’t expose `localtracks play` — run this manually:\n"
                               f"`{ctx.prefix}localtracks play {rel_path}`")
        # Archive/delete later regardless (gives time for playback to finish)
        asyncio.create_task(self._archive_or_delete(fpath))

    # ---------------------- Image Generation ----------------------

    async def generate_openai_image_png(self, prompt: str, size: str = "1024x1024") -> bytes:
        key = await self.config.openai_key()
        if not key:
            raise RuntimeError("OpenAI key not set. Use `[p]aprilcfg openaikey <key>`")
        url = "https://api.openai.com/v1/images/generations"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": "gpt-image-1", "prompt": prompt, "size": size}
        async with self.session.post(url, json=payload, headers=headers, timeout=120) as r:
            if r.status != 200:
                try:
                    err = await r.json()
                except Exception:
                    err = {"error": {"message": await r.text()}}
                msg = err.get("error", {}).get("message", f"HTTP {r.status}")
                raise RuntimeError(f"OpenAI Images API error: {msg}")
            data = await r.json()
            b64 = data["data"][0]["b64_json"]
            return base64.b64decode(b64)

    async def _artist_chatter(self, styled_prompt: str) -> str:
        """Ask chat model to roleplay describing the drawing process while image renders."""
        sys = {
            "role": "system",
            "content": (
                "You are April describing your process as an artist while you work on an image. "
                "Speak in present tense, 1–3 short sentences. Friendly, vivid, no code fences."
            ),
        }
        user = {"role": "user", "content": f"Describe starting a piece casually: {styled_prompt}"}
        try:
            text = await self.query_deepseek(0, [sys, user])
            return text.strip()[:400]
        except Exception:
            return f"Ok—I'll paint this: **{styled_prompt}**"

    # ---------------------- DeepSeek / Anthropic ----------------------

    async def query_deepseek(self, user_id: int, messages: list) -> str:
        key = await self.config.deepseek_key()
        if not key:
            raise RuntimeError("DeepSeek key not set. `[p]aprilcfg deepseekkey <key>`")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": messages,
            "temperature": await self.config.temperature(),
            "max_tokens": await self.config.max_tokens(),
            "user": str(user_id),
        }
        async with self.session.post(
            "https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=60
        ) as r:
            if r.status != 200:
                with contextlib.suppress(Exception):
                    data = await r.json()
                    raise RuntimeError(data.get("error", {}).get("message", f"HTTP {r.status}"))
                raise RuntimeError(f"DeepSeek HTTP {r.status}")
            data = await r.json()
            return data["choices"][0]["message"]["content"].strip()

    async def query_anthropic(self, user_id: int, messages: list) -> str:
        # prefer user key
        user = self.bot.get_user(user_id)
        if user:
            ucfg = self.config.user(user)
            custom = await ucfg.custom_anthropic_key()
            key = custom or (await self.config.anthropic_key())
        else:
            key = await self.config.anthropic_key()
        if not key:
            raise RuntimeError("Anthropic key not set.")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        system_content = None
        conv = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                conv.append(m)
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": conv,
            "max_tokens": await self.config.max_tokens(),
            "temperature": await self.config.temperature(),
        }
        if system_content:
            payload["system"] = system_content
        async with self.session.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=60) as r:
            if r.status != 200:
                raise RuntimeError(f"Anthropic error {r.status}: {await r.text()}")
            data = await r.json()
            return data["content"][0]["text"].strip()

    # ---------------------- Loki Memory (query_range) ----------------------

    async def _bootstrap_loki_history(self, ctx: commands.Context):
        """Fetch last few messages (by users in this channel) from Loki and seed memory."""
        try:
            url = await self.config.loki_query_url()
            if not url:
                return
            channel_id = str(ctx.channel.id)
            guild_id = str(ctx.guild.id) if ctx.guild else "DM"
            lookback = await self.config.loki_default_range()
            logs = await self._loki_fetch_messages(url, channel_id=channel_id, guild_id=guild_id, query_range=lookback, limit=5)
            if not logs:
                return
            cid = ctx.channel.id
            dq = self._history.get(cid)
            if not dq:
                dq = deque(maxlen=(await self.config.max_history()) * 2)
                self._history[cid] = dq
            for entry in logs[-5:]:
                content = entry.get("content", "")
                author = entry.get("author", {})
                name = author.get("name", "someone")
                dq.append({"role": "user", "content": f"{name} (earlier): {content}"})
        except Exception as e:
            log.debug("Loki bootstrap failed: %s", e)

    async def _loki_fetch_messages(
        self,
        query_url: str,
        *,
        channel_id: Optional[str] = None,
        guild_id: Optional[str] = None,
        query_range: str = "24h",
        limit: int = 20,
        author_id: Optional[str] = None,
    ) -> List[dict]:
        token = await self.config.loki_token()
        app = await self.config.loki_label_app()
        event = await self.config.loki_label_event()
        now = int(datetime.now(timezone.utc).timestamp() * 1e9)

        def parse_range(r: str) -> int:
            m = re.fullmatch(r"(\d+)([smhdw])", r.strip(), re.I)
            if not m:
                return int((datetime.now(timezone.utc) - timedelta(hours=24)).timestamp() * 1e9)
            n = int(m.group(1))
            unit = m.group(2).lower()
            secs = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
            return now - n * secs * (10**9)

        start = parse_range(query_range)
        params = {
            "direction": "BACKWARD",
            "limit": str(max(1, min(200, limit))),
            "start": str(start),
            "end": str(now),
        }

        labels = [f'app="{app}"', f'event_type="{event}"']
        if channel_id:
            labels.append(f'channel_id="{channel_id}"')
        if guild_id:
            labels.append(f'guild_id="{guild_id}"')
        selector = "{" + ",".join(labels) + "}"
        query = f'{selector} |= ""'

        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with self.session.get(query_url, params={"query": query, **params}, headers=headers, timeout=20) as r:
            if r.status != 200:
                txt = await r.text()
                log.debug("Loki query error %s: %s", r.status, txt)
                return []
            data = await r.json()
            result = data.get("data", {}).get("result", [])
            out: List[dict] = []
            for stream in result:
                for ts, line in stream.get("values", []):
                    try:
                        obj = json.loads(line)
                        if author_id and str(obj.get("author", {}).get("id")) != str(author_id):
                            continue
                        out.append(obj)
                    except Exception:
                        continue
            out.sort(key=lambda x: x.get("created_at", ""))
            return out

    # ---------------------- Voice events ----------------------

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        if member.bot:
            return
        try:
            player = self.get_player(member.guild.id)
            if player and self.is_player_connected(player):
                channel_id = self.get_player_channel_id(player)
                if not channel_id:
                    return
                vc = self.bot.get_channel(channel_id)
                if not vc:
                    return
                humans = [m for m in vc.members if not m.bot]
                if len(humans) == 0:
                    await player.disconnect()
                    log.debug("Left voice in %s (empty channel)", member.guild)
        except Exception as e:
            log.error("voice_state error: %s", e)

    # ---------------------- Sleep gate ----------------------

    async def _sleep_gate(self, ctx: commands.Context) -> bool:
        """If sleep mode is on, only reply to configured owner."""
        if not (await self.config.sleep_enabled()):
            return False
        owner_id = await self.config.sleep_owner_id()
        if int(getattr(ctx.author, "id", 0)) != int(owner_id):
            with contextlib.suppress(Exception):
                await ctx.message.add_reaction("😴")
            return True
        return False

    # ---------------------- Unload / cleanup ----------------------

    def cog_unload(self):
        self._unloading = True
        with contextlib.suppress(Exception):
            self.bot.loop.create_task(self.session.close())


async def setup(bot: Red):
    await bot.add_cog(AprilTalk(bot))
