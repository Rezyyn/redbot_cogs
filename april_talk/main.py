# april_talk.py
# Full Red cog for April with working Lavalink TTS (integrated from your old code),
# chat (DeepSeek/Claude), images (OpenAI), GIFs, and handy config commands.
# Keeps Config ID = 1398462

import asyncio
import aiohttp
import base64
import contextlib
import discord
import json
import lavalink
import logging
import os
import random
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path

log = logging.getLogger("red.april_talk")
log.setLevel(logging.DEBUG)

# ---------------------------
# Constants / Small resources
# ---------------------------

STYLE_SUFFIX = ", in a futuristic neo-cyberpunk aesthetic"

EMOTION_GIFS = {
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


# ---------------
# The Cog
# ---------------

class AprilTalk(commands.Cog):
    """April — chat, images, and Lavalink TTS (localtracks)."""

    def __init__(self, bot: Red):
        self.bot = bot
        self.session = aiohttp.ClientSession()
        self.config = Config.get_conf(self, identifier=1398462)

        # Runtime state
        self._unloading = False
        self._history: Dict[int, deque] = {}     # channel_id -> deque of chat messages
        self._img_sem = asyncio.Semaphore(2)

        # Storage dir (for misc)
        self.tts_dir = Path(cog_data_path(self)) / "tts"
        self.tts_dir.mkdir(exist_ok=True, parents=True)
        self.tts_files = set()

        # Config (global)
        self.config.register_global(
            # Keys
            deepseek_key="",
            anthropic_key="",
            openai_key="",
            tts_key="",  # ElevenLabs
            # LLMs
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            system_prompt=(
                "You are April, a helpful AI assistant for Discord. "
                "Default to useful answers. Be concise and kind. "
                "You can generate images when asked; do not say you cannot draw."
            ),
            smert_prompt=(
                "You are April in 'smert' mode — an incredibly intelligent, witty, and creative "
                "AI assistant with deep knowledge across all domains."
            ),
            # Messaging / UX
            tts_enabled=True,
            text_response_when_voice=True,
            max_history=5,
            use_gifs=True,
            max_message_length=1800,
            voice_id="21m00Tcm4TlvDq8ikWAM",
        )

        # Per-user config
        self.config.register_user(
            smert_mode=False,
            custom_anthropic_key="",
            custom_smert_prompt="",
        )

    # --------------
    # Cog lifecycle
    # --------------

    def cog_unload(self):
        self._unloading = True
        with contextlib.suppress(Exception):
            self.bot.loop.create_task(self.session.close())
        # Clean remaining TTS files (best-effort)
        for path in list(self.tts_files):
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
            finally:
                self.tts_files.discard(path)

    # -------------------
    # Utility Helpers
    # -------------------

    def _ensure_channel_history(self, channel_id: int):
        if channel_id not in self._history:
            max_history = self.bot.loop.create_task(self.config.max_history())
            # in case we can't await here (constructor-like), set a sane default
            maxlen = 10
            if hasattr(max_history, "result"):
                try:
                    maxlen = int(max_history.result()) * 2
                except Exception:
                    pass
            self._history[channel_id] = deque(maxlen=maxlen)

    def style_prompt(self, prompt: str) -> str:
        p = prompt.strip()
        if any(k in p.lower() for k in ["cyberpunk", "synthwave", "futuristic", "sci-fi", "science fiction", "blade runner"]):
            return p
        return p + STYLE_SUFFIX

    def _collect_image_attachments(self, message: discord.Message) -> List[str]:
        """Collect image URLs from attachments/embeds to use as chat context."""
        urls: List[str] = []
        try:
            for a in getattr(message, "attachments", []) or []:
                ct = (a.content_type or "").lower() if getattr(a, "content_type", None) else ""
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
        except Exception as ex:
            log.debug("collect_image_attachments err: %s", ex)
        return urls[:8]

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

    # -------------------
    # Lavalink Helpers
    # -------------------

    def get_player(self, guild_id: int):
        """Safe player getter — no exceptions if missing."""
        try:
            pm = getattr(lavalink, "player_manager", None)
            if pm:
                players = getattr(pm, "players", None)
                if isinstance(players, dict):
                    return players.get(guild_id)
                get = getattr(pm, "get", None)
                if callable(get):
                    return get(guild_id)
            if hasattr(lavalink, "get_player"):
                try:
                    return lavalink.get_player(guild_id)
                except Exception:
                    return None
        except Exception:
            return None
        return None

    def is_player_connected(self, player) -> bool:
        if not player:
            return False
        try:
            if hasattr(player, "is_connected") and callable(player.is_connected):
                return bool(player.is_connected())
            if hasattr(player, "is_connected"):
                return bool(player.is_connected)
            if hasattr(player, "channel_id"):
                return bool(player.channel_id)
            if hasattr(player, "channel"):
                return bool(player.channel)
        except Exception:
            return False
        return False

    def get_player_channel_id(self, player):
        if not player:
            return None
        try:
            if hasattr(player, "channel_id"):
                return player.channel_id
            if hasattr(player, "channel") and player.channel:
                return player.channel.id
        except Exception:
            return None
        return None

    # -------------------
    # Commands
    # -------------------

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april(self, ctx: commands.Context, *, input: str):
        """Main April command."""
        # short-hands
        low = input.strip().lower()
        if low == "join":
            return await self.join_voice(ctx)
        if low == "leave":
            return await self.leave_voice(ctx)
        if low == "clearhistory":
            return await self.clear_history(ctx)
        if low == "smert":
            return await self.toggle_smert_mode(ctx)

        image_urls = self._collect_image_attachments(ctx.message)
        await self.process_query(ctx, input, context_images=image_urls)

    @april.command(name="draw")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april_draw(self, ctx: commands.Context, *, prompt: str):
        """Generate an image via OpenAI Images."""
        styled = self.style_prompt(prompt.strip())
        # while generating, speak casually like an artist
        chatter = await self._artist_chatter(styled)
        await ctx.send(chatter)

        try:
            async with self._img_sem:
                png = await self.generate_openai_image_png(styled, size="1024x1024")
            file = discord.File(BytesIO(png), filename="april_draw.png")
            await ctx.send(content=f"**Image prompt used:** {styled}", file=file)
            # remember context
            self._remember_draw_context(ctx.channel.id, styled)
        except Exception as e:
            log.exception("Draw failed")
            await ctx.send(f"⚠️ I couldn't draw that: `{e}`")

    @april.command(name="join")
    async def join_voice(self, ctx: commands.Context):
        """Join your current voice channel."""
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a voice channel.")
        ch = ctx.author.voice.channel
        perms = ch.permissions_for(ctx.me)
        if not perms.connect or not perms.speak:
            return await ctx.send("❌ I need permissions to connect and speak!")
        try:
            # disconnect from other channel if already connected
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                cur = self.get_player_channel_id(player)
                if cur == ch.id:
                    return await ctx.send(f"✅ Already connected to {ch.name}")
                await player.disconnect()
            await lavalink.connect(ch)
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await ctx.send(f"🔊 Joined {ch.name}")
            else:
                await ctx.send("❌ Failed to connect.")
        except Exception as e:
            log.exception("join failed")
            await ctx.send(f"❌ Join failed: {e}")

    @april.command(name="leave")
    async def leave_voice(self, ctx: commands.Context):
        """Leave voice."""
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
        """Toggle 'smert' mode (Claude if key present)."""
        ucfg = self.config.user(ctx.author)
        mode = await ucfg.smert_mode()
        if not mode:
            custom = await ucfg.custom_anthropic_key()
            global_key = await self.config.anthropic_key()
            if not custom and not global_key:
                return await ctx.send(
                    "❌ Smert mode requires an Anthropic API key. "
                    "Use `[p]aprilcfg anthropickey <key>` or `[p]apriluser setkey <key>`."
                )
            await ucfg.smert_mode.set(True)
            await ctx.send("🧠 Smert mode ON.")
        else:
            await ucfg.smert_mode.set(False)
            await ctx.send("💡 Smert mode OFF.")

    # -------------------
    # Config commands
    # -------------------

    @commands.group(name="aprilcfg", aliases=["aprilconfig"])
    @commands.is_owner()
    async def aprilcfg(self, ctx: commands.Context):
        """Configure April."""
        if ctx.invoked_subcommand is None:
            await self.show_settings(ctx)

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

    @aprilcfg.command(name="elevenlabs")
    async def cfg_elevenlabs(self, ctx: commands.Context, key: str):
        await self.config.tts_key.set(key)
        await ctx.tick()
        with contextlib.suppress(Exception):
            await ctx.message.delete()

    @aprilcfg.command(name="voice")
    async def cfg_voice(self, ctx: commands.Context, voice_id: str):
        await self.config.voice_id.set(voice_id)
        await ctx.send(f"✅ Voice ID set to `{voice_id}`")

    @aprilcfg.command(name="model")
    async def cfg_model(self, ctx: commands.Context, model_name: str):
        await self.config.model.set(model_name.lower())
        await ctx.send(f"✅ Model set to `{model_name}`")

    @aprilcfg.command(name="prompt")
    async def cfg_prompt(self, ctx: commands.Context, *, system_prompt: str):
        await self.config.system_prompt.set(system_prompt)
        await ctx.send("✅ System prompt updated.")

    @aprilcfg.command(name="smertprompt")
    async def cfg_smertprompt(self, ctx: commands.Context, *, prompt: str):
        await self.config.smert_prompt.set(prompt)
        await ctx.send("✅ Smert prompt updated.")

    @aprilcfg.command(name="temperature")
    async def cfg_temperature(self, ctx: commands.Context, value: float):
        if 0.0 <= value <= 1.0:
            await self.config.temperature.set(value)
            await ctx.send(f"✅ Temperature set to `{value}`")
        else:
            await ctx.send("❌ Value must be between 0.0 and 1.0")

    @aprilcfg.command(name="tokens")
    async def cfg_tokens(self, ctx: commands.Context, num: int):
        if 100 <= num <= 4096:
            await self.config.max_tokens.set(num)
            await ctx.send(f"✅ Max tokens set to `{num}`")
        else:
            await ctx.send("❌ Value must be between 100 and 4096")

    @aprilcfg.command(name="tts")
    async def cfg_tts(self, ctx: commands.Context, enabled: bool):
        await self.config.tts_enabled.set(enabled)
        await ctx.send(f"✅ TTS {'enabled' if enabled else 'disabled'}")

    @aprilcfg.command(name="textresponse")
    async def cfg_textresponse(self, ctx: commands.Context, enabled: bool):
        await self.config.text_response_when_voice.set(enabled)
        await ctx.send(f"✅ Text responses will be {'shown' if enabled else 'hidden'} when using voice")

    @aprilcfg.command(name="maxhistory")
    async def cfg_maxhistory(self, ctx: commands.Context, num: int):
        if 1 <= num <= 20:
            await self.config.max_history.set(num)
            for channel_id in self._history:
                self._history[channel_id] = deque(self._history[channel_id], maxlen=num * 2)
            await ctx.send(f"✅ Max history set to `{num}` exchanges")
        else:
            await ctx.send("❌ Value must be between 1 and 20")

    @aprilcfg.command(name="gifs")
    async def cfg_gifs(self, ctx: commands.Context, enabled: bool):
        await self.config.use_gifs.set(enabled)
        await ctx.send(f"✅ Emotion GIFs {'enabled' if enabled else 'disabled'}")

    @aprilcfg.command(name="messagelength")
    async def cfg_msglen(self, ctx: commands.Context, length: int):
        if 500 <= length <= 2000:
            await self.config.max_message_length.set(length)
            await ctx.send(f"✅ Max message length set to `{length}`")
        else:
            await ctx.send("❌ Value must be between 500 and 2000")

    @aprilcfg.command(name="settings")
    async def show_settings(self, ctx: commands.Context):
        cfg = await self.config.all()
        e = discord.Embed(title="April Configuration", color=await ctx.embed_color())
        e.add_field(name="DeepSeek", value="✅" if cfg["deepseek_key"] else "❌", inline=True)
        e.add_field(name="Anthropic", value="✅" if cfg["anthropic_key"] else "❌", inline=True)
        e.add_field(name="OpenAI (images)", value="✅" if cfg["openai_key"] else "❌", inline=True)
        e.add_field(name="ElevenLabs", value="✅" if cfg["tts_key"] else "❌", inline=True)
        e.add_field(name="Voice ID", value=f"`{cfg['voice_id']}`", inline=True)
        e.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        e.add_field(name="Temperature", value=f"`{cfg['temperature']}`", inline=True)
        e.add_field(name="Max Tokens", value=f"`{cfg['max_tokens']}`", inline=True)
        e.add_field(name="Max History", value=f"`{cfg['max_history']}`", inline=True)
        e.add_field(name="Max Msg Length", value=f"`{cfg['max_message_length']}`", inline=True)
        e.add_field(name="TTS Enabled", value="✅" if cfg["tts_enabled"] else "❌", inline=True)
        e.add_field(name="Text w/ Voice", value="✅" if cfg["text_response_when_voice"] else "❌", inline=True)
        e.add_field(name="GIFs", value="✅" if cfg["use_gifs"] else "❌", inline=True)
        sp = cfg["system_prompt"][:200] + ("..." if len(cfg["system_prompt"]) > 200 else "")
        e.add_field(name="System Prompt", value=f"```{sp}```", inline=False)
        sm = cfg["smert_prompt"][:200] + ("..." if len(cfg["smert_prompt"]) > 200 else "")
        e.add_field(name="Smert Prompt", value=f"```{sm}```", inline=False)
        await ctx.send(embed=e)

    # ---------------
    # Core chat flow
    # ---------------

    async def process_query(self, ctx: commands.Context, input_text: str, *, context_images: Optional[List[str]] = None):
        # Should we voice?
        use_voice = False
        if await self.config.tts_enabled():
            try:
                player = self.get_player(ctx.guild.id)
                use_voice = bool(player) and self.is_player_connected(player)
                log.debug("TTS check - player: %s, connected: %s", player, use_voice)
            except Exception:
                use_voice = False

        # Mode & prompts
        ucfg = self.config.user(ctx.author)
        smert = await ucfg.smert_mode()
        if smert:
            sys_prompt = (await ucfg.custom_smert_prompt()) or (await self.config.smert_prompt())
        else:
            sys_prompt = await self.config.system_prompt()

        cid = ctx.channel.id
        self._ensure_channel_history(cid)

        # Build messages
        messages = [{"role": "system", "content": sys_prompt}]
        messages.extend(self._history[cid])

        user_content = input_text
        if context_images:
            imgs_text = "\n".join(f"[image]: {u}" for u in context_images)
            user_content = f"{input_text}\n\nAttached images:\n{imgs_text}"

        messages.append({"role": "user", "content": user_content})

        # Call model
        async with ctx.typing():
            try:
                if smert:
                    resp = await self.query_anthropic(ctx.author.id, messages)
                else:
                    resp = await self.query_deepseek(ctx.author.id, messages)
            except Exception as e:
                log.exception("LLM error")
                return await ctx.send(f"❌ Error: {e}")

        # Extract <draw> blocks, remove from visible text
        draw_prompt = self._maybe_extract_draw_prompt(resp)
        clean_resp = re.sub(r"<draw>.*?</draw>", "", resp, flags=re.IGNORECASE | re.DOTALL).strip()

        # Update rolling memory
        self._history[cid].append({"role": "user", "content": input_text})
        self._history[cid].append({"role": "assistant", "content": clean_resp})

        tasks = []
        if not (use_voice and not await self.config.text_response_when_voice()):
            tasks.append(asyncio.create_task(self.send_streamed_response(ctx, clean_resp)))
        if use_voice:
            tasks.append(asyncio.create_task(self.speak_response(ctx, clean_resp)))

        if draw_prompt:
            styled = self.style_prompt(draw_prompt)

            async def do_draw():
                try:
                    chatter = await self._artist_chatter(styled)
                except Exception:
                    chatter = f"Let me sketch this: **{styled}** …"
                await ctx.send(chatter)
                try:
                    async with self._img_sem:
                        png = await self.generate_openai_image_png(styled, size="1024x1024")
                    file = discord.File(BytesIO(png), filename="april_draw.png")
                    await ctx.send(content=f"**Image prompt used:** {styled}", file=file)
                    self._remember_draw_context(cid, styled)
                except Exception as e:
                    await ctx.send(f"⚠️ Couldn't render the suggested image: `{e}`")

            tasks.append(asyncio.create_task(do_draw()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # streaming text + optional GIF
    async def send_streamed_response(self, ctx: commands.Context, resp: str):
        max_length = await self.config.max_message_length()
        use_gifs = await self.config.use_gifs()
        emotion = await self.detect_emotion(resp) if use_gifs else None
        gif_url = None
        if emotion and random.random() < 0.3:
            gif_url = random.choice(EMOTION_GIFS[emotion])

        if len(resp) <= max_length:
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

        # split long
        embed = discord.Embed(description="💭 *Thinking...*", color=await ctx.embed_color())
        thinking_msg = await ctx.send(embed=embed)

        chunks: List[str] = []
        words = resp.split(" ")
        cur, cur_len = [], 0
        for w in words:
            if cur_len + len(w) + 1 > max_length:
                chunks.append(" ".join(cur))
                cur, cur_len = [w], len(w)
            else:
                cur.append(w)
                cur_len += len(w) + 1
        if cur:
            chunks.append(" ".join(cur))

        await thinking_msg.edit(content=chunks[0], embed=None)
        for i, ch in enumerate(chunks[1:], 1):
            await asyncio.sleep(0.5)
            if gif_url and i == len(chunks) - 1:
                await ctx.send(ch + f"\n{gif_url}")
            else:
                await ctx.send(ch)

    # -------------
    # TTS (OLD FLOW integrated): ElevenLabs -> file in Audio/localtracks/april_tts -> play via Audio.command_play
    # -------------

    def clean_text_for_tts(self, text: str) -> str:
        # Remove markdown and links
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        text = re.sub(r"`(.*?)`", r"\1", text)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 1000:
            text = text[:1000].rsplit(" ", 1)[0] + "..."
        return text

    async def generate_tts_audio(self, text: str, api_key: str) -> Optional[bytes]:
        """ElevenLabs API call (as in your working version)."""
        try:
            voice_id = await self.config.voice_id()
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            payload = {
                "text": text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                },
            }
            headers = {"xi-api-key": api_key, "Content-Type": "application/json"}

            log.debug("Requesting TTS for text: %s...", text[:100])
            async with self.session.post(url, json=payload, headers=headers, timeout=60) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    log.debug("TTS bytes: %s", len(data))
                    return data
                else:
                    err = await resp.text()
                    log.error("TTS API error %s: %s", resp.status, err)
                    return None
        except Exception as e:
            log.error("TTS generation failed: %s", e)
            return None

    async def speak_response(self, ctx: commands.Context, text: str):
        """Generate TTS, write to Audio/localtracks/april_tts, and play via Audio.command_play."""
        tts_key = await self.config.tts_key()
        if not tts_key:
            log.warning("Skipping TTS: missing ElevenLabs key.")
            return

        # Must be connected
        player = self.get_player(ctx.guild.id)
        if not self.is_player_connected(player):
            log.warning("Skipping TTS: player not connected.")
            return

        clean_text = self.clean_text_for_tts(text)
        if not clean_text:
            return

        audio = await self.generate_tts_audio(clean_text, tts_key)
        if not audio:
            with contextlib.suppress(Exception):
                await ctx.send("TTS failed to generate audio.")
            return

        # Path: <cog_data>/../Audio/localtracks/april_tts/<file>.mp3
        localtrack_dir = cog_data_path(self).parent / "Audio" / "localtracks" / "april_tts"
        localtrack_dir.mkdir(parents=True, exist_ok=True)

        filename = f"tts_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
        filepath = localtrack_dir / filename

        try:
            with open(filepath, "wb") as f:
                f.write(audio)
            self.tts_files.add(str(filepath))
            log.debug("TTS saved: %s", filepath)
        except Exception as e:
            with contextlib.suppress(Exception):
                await ctx.send(f"❌ Failed to write TTS file: `{e}`")
            return

        # Play using the Audio cog directly (your working approach)
        audio_cog = self.bot.get_cog("Audio")
        if not audio_cog:
            log.error("Audio cog not found")
            return

        # IMPORTANT: prefix with localtracks/ so provider doesn't hit YouTube
        query = f"localtracks/april_tts/{filename}"
        try:
            play_cmd = getattr(audio_cog, "command_play", None)
            if callable(play_cmd):
                await play_cmd(ctx, query=query)
            else:
                # Fallback to invoking the `play` command if attribute differs by version
                play_cmd2 = self.bot.get_command("play")
                if play_cmd2:
                    await ctx.invoke(play_cmd2, query=query)
                else:
                    await ctx.send(f"▶️ Please run: `{ctx.prefix}play {query}`")
        except Exception as e:
            log.exception("Failed to invoke Audio play for TTS: %s", e)
            with contextlib.suppress(Exception):
                await ctx.send(f"❌ Failed to play TTS: `{e}`")

        # Cleanup after delay (archive handling could be added later if you want)
        async def delayed_delete():
            await asyncio.sleep(30)
            try:
                if filepath.exists():
                    filepath.unlink()
                    log.debug("TTS file deleted: %s", filepath.name)
            except Exception as e:
                log.error("Failed to delete TTS file %s — %s", filepath.name, e)
            finally:
                self.tts_files.discard(str(filepath))

        asyncio.create_task(delayed_delete())

    # -------------
    # Images
    # -------------

    async def generate_openai_image_png(self, prompt: str, size: str = "1024x1024") -> bytes:
        key = await self.config.openai_key()
        if not key:
            raise RuntimeError("OpenAI API key not set. Use `[p]aprilcfg openaikey <key>`.")
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
        """Quick 'artist at work' chatter to cover image latency."""
        sys = {
            "role": "system",
            "content": (
                "You are April describing your process as an artist while you work on an image. "
                "1–3 short sentences, casual, present tense."
            ),
        }
        usr = {"role": "user", "content": f"Describe starting a piece casually: {styled_prompt}"}
        try:
            return (await self.query_deepseek(0, [sys, usr]))[:400]
        except Exception:
            return f"Ok—I'll paint this: **{styled_prompt}**"

    def _maybe_extract_draw_prompt(self, text: str) -> Optional[str]:
        m = re.search(r"<draw>(.*?)</draw>", text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return None
        raw = re.sub(r"\s+", " ", m.group(1).strip())
        return raw or None

    def _remember_draw_context(self, channel_id: int, styled_prompt: str):
        self._ensure_channel_history(channel_id)
        self._history[channel_id].append(
            {"role": "assistant", "content": f"[image-used-prompt]: {styled_prompt}"}
        )

    # -------------
    # LLM calls
    # -------------

    async def query_deepseek(self, user_id: int, messages: list) -> str:
        key = await self.config.deepseek_key()
        if not key:
            raise RuntimeError("DeepSeek API key not set. Use `[p]aprilcfg deepseekkey <key>`.")
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
        user = self.bot.get_user(user_id)
        if user:
            custom = await self.config.user(user).custom_anthropic_key()
            key = custom or (await self.config.anthropic_key())
        else:
            key = await self.config.anthropic_key()
        if not key:
            raise RuntimeError("Anthropic API key not set.")

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

    # -------------
    # Voice events
    # -------------

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
                voice_ch = self.bot.get_channel(channel_id)
                if not voice_ch:
                    return
                humans = [m for m in voice_ch.members if not m.bot]
                if len(humans) == 0:
                    await player.disconnect()
                    log.debug("Left voice in %s (empty channel)", member.guild)
        except Exception as e:
            log.error("voice_state error: %s", e)


async def setup(bot: Red):
    await bot.add_cog(AprilTalk(bot))
