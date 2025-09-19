import asyncio
import aiohttp
import discord
import lavalink
import os
import logging
import random
import time
import base64
import re
import json
from collections import deque
from pathlib import Path
from io import BytesIO
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import pagify
from typing import Optional

# Logger
tllogger = logging.getLogger("red.aprilai")
tllogger.setLevel(logging.DEBUG)

STYLE_SUFFIX = ", in a futuristic neo-cyberpunk aesthetic"

EMOTION_GIFS = {
    "happy": [
        "https://media.giphy.com/media/XbxZ41fWLeRECPsGIJ/giphy.gif",
        "https://media.giphy.com/media/l0HlMG1EX2H38cZeE/giphy.gif",
        "https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif"
    ],
    "thinking": [
        "https://media.giphy.com/media/d3mlE7uhX8KFgEmY/giphy.gif",
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",
        "https://media.giphy.com/media/l0HlUNj5BRuYDLxFm/giphy.gif"
    ],
    "confused": [
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",
        "https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif",
        "https://media.giphy.com/media/3oEjI5VtIhHvK37WYo/giphy.gif"
    ],
    "excited": [
        "https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif",
        "https://media.giphy.com/media/l0HlMURBbyUqF0XQI/giphy.gif",
        "https://media.giphy.com/media/3rgXBOmTlzyFCURutG/giphy.gif"
    ],
    "sad": [
        "https://media.giphy.com/media/OPU6wzx8JrHna/giphy.gif",
        "https://media.giphy.com/media/l1AsyjZ8XLd1V7pUk/giphy.gif",
        "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif"
    ]
}


class AprilAI(commands.Cog):
    """AI assistant with text, voice via Lavalink, and Loki memory recall"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1398462)
        self.session = aiohttp.ClientSession()
        self.history = {}
        # Per-channel lightweight recall cache {channel_id: {"username": [lines...]}}
        self.recall = {}
        self.tts_files = set()
        self._unloading = False
        self.streaming_messages = {}

        # Performance tunables
        self._api_sem = asyncio.Semaphore(3)   # cap concurrent external calls
        self._tts_sem = asyncio.Semaphore(1)   # serialize TTS
        self._edit_delay = 0.025               # reduce Discord edit spam
        self._chunk_size = 160                 # streamed edit size chars

        # Create TTS directory
        self.tts_dir = Path(cog_data_path(self)) / "tts"
        self.tts_dir.mkdir(exist_ok=True, parents=True)

        self.config.register_global(
            deepseek_key="",
            anthropic_key="",
            tts_key="",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="You are April, a helpful AI assistant for Discord. Default to useful answers. Be concise and kind.",
            smert_prompt="You are April in 'smert' mode - an incredibly intelligent, witty, and creative AI assistant with deep knowledge across all domains.",
            tts_enabled=True,
            text_response_when_voice=True,
            max_history=5,
            use_gifs=True,
            max_message_length=1800,
            openai_key="",
            # Loki config
            loki_url="http://localhost:3100",
            loki_token="",
        )

        self.config.register_user(
            smert_mode=False,
            custom_anthropic_key="",
            custom_smert_prompt=""
        )

    def cog_unload(self):
        self._unloading = True
        self.bot.loop.create_task(self.session.close())

        # Clean up any remaining TTS files
        for path in list(self.tts_files):
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass
            finally:
                self.tts_files.discard(path)

    # -------------- small utility --------------
    async def _with_limit(self, sem: asyncio.Semaphore, coro):
        async with sem:
            return await coro

    # -----------------------
    # Prompt helpers
    # -----------------------
    def style_prompt(self, prompt: str) -> str:
        p = prompt.strip()
        if any(k in p.lower() for k in ["cyberpunk", "synthwave", "futuristic", "sci-fi", "science fiction", "blade runner"]):
            return p
        return p + STYLE_SUFFIX

    def maybe_extract_draw_prompt(self, text: str) -> Optional[str]:
        match = re.search(r"<draw>(.*?)</draw>", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        raw = match.group(1).strip()
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw or None

    # -----------------------
    # Lavalink helpers
    # -----------------------
    async def detect_emotion(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        if any(word in text_lower for word in ["happy", "great", "awesome", "excellent", "wonderful", "amazing"]):
            return "happy"
        elif any(word in text_lower for word in ["think", "consider", "ponder", "wonder", "hmm"]):
            return "thinking"
        elif any(word in text_lower for word in ["confused", "don't understand", "what", "huh", "unclear"]):
            return "confused"
        elif any(word in text_lower for word in ["excited", "can't wait", "awesome!", "wow", "amazing!"]):
            return "excited"
        elif any(word in text_lower for word in ["sad", "sorry", "unfortunately", "regret"]):
            return "sad"
        return None

    def get_player(self, guild_id: int):
        try:
            return lavalink.get_player(guild_id)
        except Exception:
            return None

    def is_player_connected(self, player) -> bool:
        if not player:
            return False
        try:
            if hasattr(player, 'is_connected') and callable(player.is_connected):
                return player.is_connected()
            elif hasattr(player, 'is_connected'):
                return player.is_connected
            elif hasattr(player, 'channel_id'):
                return bool(player.channel_id)
            elif hasattr(player, 'channel'):
                return bool(player.channel)
            else:
                return False
        except Exception:
            return False

    def get_player_channel_id(self, player):
        try:
            if hasattr(player, 'channel_id'):
                return player.channel_id
            elif hasattr(player, 'channel') and player.channel:
                return player.channel.id
        except Exception:
            pass
        return None

    # -----------------------
    # Commands
    # -----------------------
    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, input: str):
        """Main April command"""
        cmd = input.strip()

        # special trigger (case-insensitive, tolerant)
        if cmd.lower().startswith("i know what you're thinking"):
            return await self._draw_what_you_think(ctx)

        # Handle special short text commands
        low = cmd.lower()
        if low == "join":
            return await self.join_voice(ctx)
        if low == "leave":
            return await self.leave_voice(ctx)
        if low == "clearhistory":
            return await self.clear_history(ctx)
        if low == "smert":
            return await self.toggle_smert_mode(ctx)

        await self.process_query(ctx, input)

    @april.command(name="draw")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april_draw(self, ctx: commands.Context, *, prompt: str):
        p = prompt.strip()
        if p.lower().startswith("me "):
            p = p[3:].strip()
        styled_prompt = self.style_prompt(p)
        try:
            async with ctx.typing():
                png_bytes = await self._with_limit(self._api_sem, self.generate_openai_image_png(styled_prompt, size="1024x1024"))
                file = discord.File(BytesIO(png_bytes), filename="april_draw.png")
                await ctx.send(content=f"**Prompt:** {styled_prompt}", file=file)
        except Exception as e:
            await ctx.send(f"⚠️ I couldn't draw that: `{e}`\nSet the key with `[p]aprilconfig openaikey <key>` if needed.")

    @april.command(name="smert")
    async def toggle_smert_mode(self, ctx):
        user_config = self.config.user(ctx.author)
        current_mode = await user_config.smert_mode()
        if not current_mode:
            custom_key = await user_config.custom_anthropic_key()
            global_key = await self.config.anthropic_key()
            if not custom_key and not global_key:
                return await ctx.send(
                    "❌ Smert mode requires an Anthropic API key. "
                    "Set one with `[p]aprilconfig anthropickey <key>` or `[p]apriluser setkey <key>`"
                )
            await user_config.smert_mode.set(True)
            await ctx.send("🧠 Smert mode activated! I'm now using Claude for enhanced intelligence.")
        else:
            await user_config.smert_mode.set(False)
            await ctx.send("💡 Smert mode deactivated. Back to regular mode.")

    @april.command(name="clearhistory")
    async def clear_history(self, ctx):
        channel_id = ctx.channel.id
        if channel_id in self.history:
            self.history[channel_id].clear()
        await ctx.send("✅ Conversation history cleared for this channel.")

    @april.command(name="join")
    async def join_voice(self, ctx):
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a voice channel.")
        channel = ctx.author.voice.channel
        permissions = channel.permissions_for(ctx.me)
        if not permissions.connect or not permissions.speak:
            return await ctx.send("❌ I need permissions to connect and speak!")
        try:
            if not hasattr(lavalink, 'get_player'):
                return await ctx.send("❌ Lavalink is not initialized. `[p]load audio`")
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                current_channel_id = self.get_player_channel_id(player)
                if current_channel_id == channel.id:
                    return await ctx.send(f"✅ Already connected to {channel.name}")
                else:
                    await player.disconnect()
            await lavalink.connect(channel)
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await ctx.send(f"🔊 Joined {channel.name}")
            else:
                await ctx.send("❌ Failed to connect to voice channel")
        except Exception as e:
            await ctx.send(f"❌ Join failed: {e}")

    @april.command(name="leave")
    async def leave_voice(self, ctx):
        try:
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await player.stop()
                await player.disconnect()
                await ctx.send("👋 Disconnected from voice")
            else:
                await ctx.send("❌ Not connected to a voice channel.")
        except Exception as e:
            await ctx.send(f"❌ Disconnect failed: {e}")

    # -----------------------
    # Core flow
    # -----------------------
    async def process_query(self, ctx, input_text):
        use_voice = False
        if await self.config.tts_enabled():
            try:
                player = self.get_player(ctx.guild.id)
                use_voice = player and self.is_player_connected(player)
            except Exception:
                use_voice = False

        user_config = self.config.user(ctx.author)
        smert_mode = await user_config.smert_mode()

        async with ctx.typing():
            try:
                channel_id = ctx.channel.id
                if channel_id not in self.history:
                    max_history = await self.config.max_history()
                    self.history[channel_id] = deque(maxlen=max_history * 2)

                # Build messages
                if smert_mode:
                    custom_prompt = await user_config.custom_smert_prompt()
                    system_prompt = custom_prompt or await self.config.smert_prompt()
                else:
                    system_prompt = await self.config.system_prompt()

                messages = [{"role": "system", "content": system_prompt}]

                # Inject recall (memory)
                ch_recall = self.recall.get(channel_id, {})
                if ch_recall:
                    mem_chunks = []
                    for uname, lines in ch_recall.items():
                        if not lines:
                            continue
                        subset = lines[:5]
                        mem_chunks.append(f"{uname} recent: " + " | ".join(subset))
                    if mem_chunks:
                        messages.append({"role": "system", "content": f"[memory] " + " || ".join(mem_chunks)})

                messages.extend(self.history[channel_id])
                messages.append({"role": "user", "content": input_text})

                # Query model
                if smert_mode:
                    resp = await self._with_limit(self._api_sem, self.query_anthropic(ctx.author.id, messages))
                else:
                    resp = await self._with_limit(self._api_sem, self.query_deepseek(ctx.author.id, messages))

                # Handle inline draw
                draw_prompt = self.maybe_extract_draw_prompt(resp)
                clean_resp = re.sub(r"<draw>.*?</draw>", "", resp, flags=re.IGNORECASE | re.DOTALL).strip()

                # Update history
                self.history[channel_id].append({"role": "user", "content": input_text})
                self.history[channel_id].append({"role": "assistant", "content": clean_resp})

                tasks = []
                if not (use_voice and not await self.config.text_response_when_voice()):
                    tasks.append(asyncio.create_task(self.send_streamed_response(ctx, clean_resp)))
                if use_voice:
                    tasks.append(asyncio.create_task(self.speak_response(ctx, clean_resp)))

                if draw_prompt:
                    styled = self.style_prompt(draw_prompt)

                    async def _draw_and_send():
                        try:
                            png = await self._with_limit(self._api_sem, self.generate_openai_image_png(styled, size="1024x1024"))
                            file = discord.File(BytesIO(png), filename="april_draw.png")
                            await ctx.send(content=f"pic related:", file=file)
                        except Exception as e:
                            await ctx.send(f"⚠️ Couldn't render the suggested image: `{e}`")

                    tasks.append(asyncio.create_task(_draw_and_send()))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                await ctx.send(f"❌ Error: {e}")

    async def send_streamed_response(self, ctx, resp: str):
        max_length = await self.config.max_message_length()
        use_gifs = await self.config.use_gifs()

        emotion = await self.detect_emotion(resp) if use_gifs else None
        gif_url = None
        if emotion and random.random() < 0.3:
            gif_url = random.choice(EMOTION_GIFS[emotion])

        if len(resp) > max_length:
            chunks = []
            words = resp.split(' ')
            current_chunk = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 > max_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            for i, chunk in enumerate(chunks):
                suffix = f"\n{gif_url}" if gif_url and i == len(chunks) - 1 else ""
                await ctx.send(chunk + suffix)
                await asyncio.sleep(0.2)
        else:
            embed = discord.Embed(description="💭 *Thinking...*")
            msg = await ctx.send(embed=embed)

            chunk_size = getattr(self, "_chunk_size", 160)
            for i in range(0, len(resp), chunk_size):
                chunk = resp[:i + chunk_size]
                if i + chunk_size >= len(resp) and gif_url:
                    await msg.edit(content=chunk + f"\n{gif_url}", embed=None)
                else:
                    await msg.edit(content=chunk + "▌", embed=None)
                await asyncio.sleep(getattr(self, "_edit_delay", 0.025))

            if gif_url:
                await msg.edit(content=resp + f"\n{gif_url}", embed=None)
            else:
                await msg.edit(content=resp, embed=None)

    def clean_text_for_tts(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 1000:
            text = text[:1000].rsplit(' ', 1)[0] + "..."
        return text

    async def speak_response(self, ctx, text: str):
        tts_key = await self.config.tts_key()
        if not tts_key:
            return
        player = self.get_player(ctx.guild.id)
        if not self.is_player_connected(player):
            return
        clean_text = self.clean_text_for_tts(text)
        if not clean_text.strip():
            return
        if len(clean_text) > 800:
            clean_text = clean_text[:800].rsplit(' ', 1)[0] + "..."
        audio_data = await self._with_limit(self._tts_sem, self.generate_tts_audio(clean_text, tts_key))
        if not audio_data:
            return
        localtrack_dir = cog_data_path(self).parent / "Audio" / "localtracks" / "april_tts"
        localtrack_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tts_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
        filepath = localtrack_dir / filename
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        audio_cog = self.bot.get_cog("Audio")
        if audio_cog:
            play_command = audio_cog.command_play
            await play_command(ctx, query=f"localtracks/april_tts/{filename}")

    async def generate_tts_audio(self, text: str, api_key: str) -> bytes:
        voice_id = await self.config.voice_id()
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        async with self.session.post(url, json=payload, headers=headers, timeout=30) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                return None

    async def generate_openai_image_png(self, prompt: str, size: str = "1024x1024") -> bytes:
        key = await self.config.openai_key()
        if not key:
            raise RuntimeError("OpenAI API key not set. Use `[p]aprilconfig openaikey <key>`.")
        url = "https://api.openai.com/v1/images/generations"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": "gpt-image-1", "prompt": prompt, "size": size}
        async with self.session.post(url, json=payload, headers=headers, timeout=90) as r:
            if r.status != 200:
                try:
                    err = await r.json()
                except Exception:
                    err = {"error": {"message": await r.text()}}
                message = err.get("error", {}).get("message", f"HTTP {r.status}")
                raise RuntimeError(f"OpenAI Images API error: {message}")
            data = await r.json()
            b64 = data["data"][0]["b64_json"]
            return base64.b64decode(b64)

    # -----------------------
    # DeepSeek / Anthropic
    # -----------------------
    async def query_deepseek(self, user_id: int, messages: list):
        key = await self.config.deepseek_key()
        if not key:
            raise Exception("DeepSeek API key not set. Use `[p]aprilconfig deepseekkey <key>`")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": messages,
            "temperature": await self.config.temperature(),
            "max_tokens": await self.config.max_tokens(),
            "user": str(user_id)
        }
        async with self.session.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=60
        ) as r:
            if r.status != 200:
                error_data = await r.json()
                err_msg = error_data.get("error", {}).get("message", f"HTTP Error {r.status}")
                raise Exception(f"API Error: {err_msg}")
            data = await r.json()
            return data["choices"][0]["message"]["content"].strip()

    async def query_anthropic(self, user_id: int, messages: list):
        # prefer user-specific key if present
        key = None
        user = self.bot.get_user(user_id)
        if user:
            custom_key = await self.config.user(user).custom_anthropic_key()
            key = custom_key or await self.config.anthropic_key()
        else:
            key = await self.config.anthropic_key()
        if not key:
            raise Exception("Anthropic API key not set. Use `[p]aprilconfig anthropickey <key>` or `[p]apriluser setkey <key>`")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        # Extract system content if provided
        system_content = None
        conversation_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                conversation_messages.append(msg)
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": conversation_messages,
            "max_tokens": await self.config.max_tokens(),
            "temperature": await self.config.temperature()
        }
        if system_content:
            payload["system"] = system_content
        async with self.session.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
            timeout=60
        ) as r:
            if r.status != 200:
                raise Exception(f"Anthropic API Error {r.status}: {await r.text()}")
            data = await r.json()
            return data["content"][0]["text"].strip()

    # -----------------------
    # Loki integration (tailored to your labels)
    # -----------------------
    def _mention_to_user(self, mention: str):
        """Return (id_str, name) from @mention or '@name' fallback."""
        m = re.fullmatch(r"<@!?(\d+)>", mention.strip())
        if m:
            uid = int(m.group(1))
            user = self.bot.get_user(uid)
            return (str(uid), (user.name if user else None))
        # fallback: literal username (no snowflake)
        return (None, mention.lstrip("@"))

    def _since_to_ns(self, since: str) -> int:
        # supports 1h, 6h, 24h, 7d (defaults to 24h)
        m = re.fullmatch(r"(\d+)([hd])", (since or "24h"))
        n = int(m.group(1)) if m else 24
        unit = m.group(2) if m else "h"
        sec = n * 3600 if unit == "h" else n * 86400
        return (int(time.time()) - sec) * 1_000_000_000

    async def _loki_query_range(self, query: str, start_ns: int, end_ns: int, limit: int = 50):
        cfg = await self.config.all()
        url = f"{cfg['loki_url'].rstrip('/')}/loki/api/v1/query_range"
        headers = {}
        if cfg.get("loki_token"):
            headers["Authorization"] = f"Bearer {cfg['loki_token']}"
        params = {
            "query": query,
            "start": str(start_ns),
            "end": str(end_ns),
            "limit": str(limit),
            "direction": "backward",
        }
        async with self.session.get(url, params=params, headers=headers, timeout=20) as r:
            if r.status != 200:
                raise RuntimeError(f"Loki {r.status}: {await r.text()}")
            return await r.json()

    def _build_logql_for_user(self, *, guild_id: int, channel_id: int, user_id: Optional[str], user_name: Optional[str]) -> str:
        """
        Build a LogQL that matches your labels:
          {app="discord-bot",event_type="message",guild_id="<id>",channel_id="<id>"} | json
        Then filter on either author.id or author.name.
        """
        base = f'{{app="discord-bot",event_type="message",guild_id="{guild_id}",channel_id="{channel_id}"}} | json'
        if user_id:
            # filter by snowflake id (most reliable)
            return base + f' | author.id="{user_id}"'
        if user_name:
            # fallback exact name match
            return base + f' | author.name="{user_name}"'
        return base

    @commands.guild_only()
    @april.command(name="fetch")
    async def april_fetch(self, ctx, user_mention: str, since: Optional[str] = "24h", limit: Optional[int] = 20):
        """
        Fetch recent messages for @username from Loki and cache as recall memory.
        Usage: .april fetch @username [since] [limit]
          since: 1h, 6h, 24h, 7d   (default 24h)
          limit: 1..200            (default 20)
        Scope: current guild + channel (uses labels guild_id/channel_id).
        """
        if not ctx.guild:
            return await ctx.send("❌ Must be used in a server channel.")

        uid, uname = self._mention_to_user(user_mention)
        if not uid and not uname:
            return await ctx.send("❌ I need an @mention or username.")

        limit = max(1, min(int(limit or 20), 200))
        end_ns = int(time.time() * 1_000_000_000)
        start_ns = self._since_to_ns(since or "24h")

        logql = self._build_logql_for_user(
            guild_id=ctx.guild.id,
            channel_id=ctx.channel.id,
            user_id=uid,
            user_name=uname if not uid else None,
        )

        try:
            async with ctx.typing():
                data = await self._with_limit(self._api_sem, self._loki_query_range(logql, start_ns, end_ns, limit))
                streams = data.get("data", {}).get("result", [])
                if not streams:
                    return await ctx.send(f"🔎 No logs for {user_mention} in last {since}.")

                # Flatten and parse newest-first
                rows = []
                for s in streams:
                    for ts, line in s.get("values", []):
                        rows.append((int(ts), line))
                rows.sort(key=lambda x: x[0], reverse=True)
                rows = rows[:limit]

                # Extract content safely, then print & cache for recall
                nice_lines = []
                recall_lines = []
                display_name = None
                for ts, line in rows:
                    content = None
                    author_name = None
                    try:
                        obj = json.loads(line)
                        content = obj.get("content")
                        author = obj.get("author") or {}
                        author_name = author.get("name")
                    except Exception:
                        # fallback regex
                        m = re.search(r'"content"\s*:\s*"([^"]+)"', line)
                        content = m.group(1) if m else line
                        m2 = re.search(r'"author"\s*:\s*{[^}]*"name"\s*:\s*"([^"]+)"', line)
                        author_name = m2.group(1) if m2 else None
                    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts/1_000_000_000))
                    who = author_name or uname or uid or "user"
                    display_name = display_name or who
                    text = content if content is not None else line
                    nice_lines.append(f"**{who}** at {when}: {text}")
                    recall_lines.append(text)

                # Cache recall (newest first)
                ch = ctx.channel.id
                if ch not in self.recall:
                    self.recall[ch] = {}
                key = display_name or (uname or uid or "user")
                self.recall[ch][key] = recall_lines

                # Show paged output
                for page in pagify("\n".join(nice_lines), delims=["\n"], page_length=1800):
                    await ctx.send(page)

                await ctx.send(f"🧠 Cached {len(recall_lines)} lines as memory recall for **{key}**. I’ll use them in the next replies.")

        except Exception as e:
            await ctx.send(f"⚠️ Loki query failed: `{e}`\nConfigure with `[p]aprilconfig lokiurl <url>` and optional `[p]aprilconfig lokitoken <token>`.")

    @commands.is_owner()
    @april.command(name="lokiurl")
    async def lokiurl_cmd(self, ctx, url: str):
        """Set Loki base URL, e.g. http://loki:3100"""
        await self.config.loki_url.set(url)
        await ctx.tick()

    @commands.is_owner()
    @april.command(name="lokitoken")
    async def lokitoken_cmd(self, ctx, token: str):
        """Set optional Bearer token for Loki"""
        await self.config.loki_token.set(token)
        await ctx.tick()

    # -----------------------
    # Special flow
    # -----------------------
    async def _draw_what_you_think(self, ctx: commands.Context):
        async with ctx.typing():
            try:
                channel_id = ctx.channel.id
                max_history = await self.config.max_history()
                if channel_id not in self.history:
                    self.history[channel_id] = deque(maxlen=max_history * 2)

                system_prompt = await self.config.system_prompt()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.history[channel_id])
                messages.append({"role": "user", "content": "User: I know what you're thinking, draw me it!"})

                chat_resp = await self._with_limit(self._api_sem, self.query_deepseek(ctx.author.id, messages))
                img_prompt = await self._with_limit(self._api_sem, self.query_deepseek(0, [
                    {"role": "system", "content": "You write exactly ONE single-line image prompt; no commentary, no quotes."}
                ] + messages[-10:] + [{"role": "assistant", "content": chat_resp}]))

                styled = self.style_prompt(img_prompt)
                png = await self._with_limit(self._api_sem, self.generate_openai_image_png(styled, size="1024x1024"))

                file = discord.File(BytesIO(png), filename="april_draw.png")
                await ctx.send(content=f"{chat_resp}\n\n**Image:** {styled}", file=file)

                self.history[channel_id].append({"role": "user", "content": "I know what you're thinking, draw me it!"})
                self.history[channel_id].append({"role": "assistant", "content": chat_resp + f"\n[Image: {styled}]"})
            except Exception as e:
                await ctx.send(f"⚠️ Couldn't complete the draw-what-you-think flow: `{e}`")

    # -----------------------
    # Config commands (subset)
    # -----------------------
    @commands.group(name="aprilconfig", aliases=["aprilcfg"])
    @commands.is_owner()
    async def aprilconfig(self, ctx):
        if ctx.invoked_subcommand is None:
            await self.show_settings(ctx)

    @aprilconfig.command(name="settings")
    async def show_settings(self, ctx):
        cfg = await self.config.all()
        e = discord.Embed(title="AprilAI Configuration")
        deepseek_key = cfg['deepseek_key']
        anthropic_key = cfg['anthropic_key']
        tts_key = cfg['tts_key']
        openai_key = cfg.get('openai_key', '')
        e.add_field(name="DeepSeek Key", value=f"`...{deepseek_key[-4:]}`" if deepseek_key else "❌ Not set", inline=False)
        e.add_field(name="Anthropic Key", value=f"`...{anthropic_key[-4:]}`" if anthropic_key else "❌ Not set", inline=False)
        e.add_field(name="ElevenLabs Key", value=f"`...{tts_key[-4:]}`" if tts_key else "❌ Not set", inline=False)
        e.add_field(name="OpenAI Key", value=f"`...{openai_key[-4:]}`" if openai_key else "❌ Not set", inline=False)
        e.add_field(name="Voice ID", value=f"`{cfg['voice_id']}`", inline=True)
        e.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        e.add_field(name="Temperature", value=f"`{cfg['temperature']}`", inline=True)
        e.add_field(name="Max Tokens", value=f"`{cfg['max_tokens']}`", inline=True)
        e.add_field(name="Max History", value=f"`{cfg['max_history']} exchanges`", inline=True)
        e.add_field(name="Max Message Length", value=f"`{cfg['max_message_length']}`", inline=True)
        e.add_field(name="TTS Enabled", value="✅" if cfg['tts_enabled'] else "❌", inline=True)
        e.add_field(name="Text with Voice", value="✅" if cfg['text_response_when_voice'] else "❌", inline=True)
        e.add_field(name="Emotion GIFs", value="✅" if cfg['use_gifs'] else "❌", inline=True)
        e.add_field(name="Loki URL", value=f"{cfg['loki_url']}", inline=False)
        e.add_field(name="Loki Token", value=("✅ set" if cfg.get('loki_token') else "❌ not set"), inline=True)
        prompt_preview = cfg['system_prompt'][:200] + ("..." if len(cfg['system_prompt']) > 200 else "")
        e.add_field(name="System Prompt", value=f"```{prompt_preview}```", inline=False)
        smert_preview = cfg['smert_prompt'][:200] + ("..." if len(cfg['smert_prompt']) > 200 else "")
        e.add_field(name="Smert Prompt", value=f"```{smert_preview}```", inline=False)
        await ctx.send(embed=e)

async def setup(bot: Red):
    await bot.add_cog(AprilAI(bot))
