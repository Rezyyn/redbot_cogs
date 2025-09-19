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
from typing import Optional

from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import pagify

# -----------------------------------
# Logger
# -----------------------------------
tllogger = logging.getLogger("red.apriltalk")
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

# -----------------------------------
# Cog
# -----------------------------------
class AprilTalk(commands.Cog):
    """April: chat + voice + Loki recall (standalone; keeps your april_log untouched)"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=813984652)  # unique ID
        self.session = aiohttp.ClientSession()

        # per-channel conversation (deque of dicts)
        self.history = {}
        # per-channel recall cache {channel_id: {username: [lines ...]}}
        self.recall = {}
        # TTS temp files (if needed)
        self.tts_files = set()

        # Perf tunables
        self._api_sem = asyncio.Semaphore(3)   # limit concurrent external calls
        self._tts_sem = asyncio.Semaphore(1)   # serialize TTS synthesis
        self._edit_delay = 0.025               # stream edit delay
        self._chunk_size = 160                 # stream chunk chars

        # Files
        self.tts_dir = Path(cog_data_path(self)) / "tts"
        self.tts_dir.mkdir(exist_ok=True, parents=True)

        # Config
        self.config.register_global(
            deepseek_key="",
            anthropic_key="",
            tts_key="",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="You are April, a helpful AI assistant for Discord. Default to useful answers. Be concise and kind.",
            smert_prompt="You are April in 'smert' mode - an incredibly intelligent, witty, and creative AI assistant.",
            tts_enabled=True,
            text_response_when_voice=True,
            max_history=5,
            use_gifs=True,
            max_message_length=1800,
            openai_key="",
            # Loki (works with either base or push URL)
            loki_url="http://localhost:3100",
            loki_token="",
            # Sleep mode - default to your ID
            sleep_enabled=False,
            sleep_user_id="165548483128983552",
        )

        self.config.register_user(
            smert_mode=False,
            custom_anthropic_key="",
            custom_smert_prompt=""
        )

    # -----------------------------------
    # Lifecycle
    # -----------------------------------
    def cog_unload(self):
        try:
            self.bot.loop.create_task(self.session.close())
        except Exception:
            pass
        for p in list(self.tts_files):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass
            finally:
                self.tts_files.discard(p)

    # -----------------------------------
    # Internal helpers
    # -----------------------------------
    async def _with_limit(self, sem: asyncio.Semaphore, coro):
        async with sem:
            return await coro

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
        if any(w in t for w in ["happy", "great", "awesome", "excellent", "wonderful", "amazing"]): return "happy"
        if any(w in t for w in ["think", "consider", "ponder", "wonder", "hmm"]): return "thinking"
        if any(w in t for w in ["confused", "don't understand", "what", "huh", "unclear"]): return "confused"
        if any(w in t for w in ["excited", "can't wait", "wow", "amazing!"]): return "excited"
        if any(w in t for w in ["sad", "sorry", "unfortunately", "regret"]): return "sad"
        return None

    # Lavalink helpers
    def get_player(self, guild_id: int):
        try:
            return lavalink.get_player(guild_id)
        except Exception:
            return None

    def is_player_connected(self, player) -> bool:
        if not player:
            return False
        try:
            if hasattr(player, "channel_id"):
                return bool(player.channel_id)
            if hasattr(player, "channel"):
                return bool(player.channel)
        except Exception:
            return False
        return False

    def get_player_channel_id(self, player):
        try:
            if hasattr(player, "channel_id"):
                return player.channel_id
            if hasattr(player, "channel") and player.channel:
                return player.channel.id
        except Exception:
            pass
        return None

    async def _is_allowed_to_interact(self, ctx) -> bool:
        cfg = await self.config.all()
        if not cfg.get("sleep_enabled", False):
            return True
        allowed = cfg.get("sleep_user_id")
        try:
            allowed_int = int(allowed) if allowed else None
        except Exception:
            allowed_int = None
        try:
            if await self.bot.is_owner(ctx.author):
                return True
        except Exception:
            pass
        return allowed_int is not None and ctx.author.id == allowed_int

    # -----------------------------------
    # Commands
    # -----------------------------------
    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx: commands.Context, *, input: str):
        """Main April command (chat)"""
        if not await self._is_allowed_to_interact(ctx):
            return
        cmd = input.strip()
        if cmd.lower().startswith("i know what you're thinking"):
            return await self._draw_what_you_think(ctx)
        if cmd.lower() == "join":
            return await self.join_voice(ctx)
        if cmd.lower() == "leave":
            return await self.leave_voice(ctx)
        if cmd.lower() == "clearhistory":
            return await self.clear_history(ctx)
        if cmd.lower() == "smert":
            return await self.toggle_smert_mode(ctx)
        await self.process_query(ctx, input)

    # ---- quick image command within group
    @april.command(name="draw")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april_draw(self, ctx: commands.Context, *, prompt: str):
        if not await self._is_allowed_to_interact(ctx):
            return
        p = prompt.strip()
        if p.lower().startswith("me "):
            p = p[3:].strip()
        styled = self.style_prompt(p)
        try:
            async with ctx.typing():
                png = await self._with_limit(self._api_sem, self.generate_openai_image_png(styled, size="1024x1024"))
                await ctx.send(file=discord.File(BytesIO(png), filename="april_draw.png"), content=f"**Prompt:** {styled}")
        except Exception as e:
            await ctx.send(f"⚠️ I couldn't draw that: `{e}`")

    # ---- sleep controls
    @april.command(name="sleep")
    @commands.is_owner()
    async def april_sleep(self, ctx: commands.Context, enabled: bool):
        await self.config.sleep_enabled.set(bool(enabled))
        state = "😴 ON (only allowed user can chat)" if enabled else "💬 OFF (everyone can chat)"
        await ctx.send(f"✅ Sleep mode {state}.")

    @april.command(name="sleepuser")
    @commands.is_owner()
    async def april_sleepuser(self, ctx: commands.Context, user_id: Optional[int] = None):
        uid = user_id or ctx.author.id
        await self.config.sleep_user_id.set(str(uid))
        await ctx.send(f"🔒 Allowed user set to <@{uid}>.")

    # ---- Loki setup helpers
    @april.command(name="lokiurl")
    @commands.is_owner()
    async def lokiurl_cmd(self, ctx: commands.Context, url: str):
        """Set Loki base or push URL (both accepted)."""
        await self.config.loki_url.set(url)
        await ctx.tick()

    @april.command(name="lokitoken")
    @commands.is_owner()
    async def lokitoken_cmd(self, ctx: commands.Context, token: str):
        """Optional bearer token."""
        await self.config.loki_token.set(token)
        await ctx.tick()

    @april.command(name="lokiverify")
    @commands.is_owner()
    async def lokiverify_cmd(self, ctx: commands.Context):
        """Check Loki endpoints resolve + respond."""
        cfg = await self.config.all()
        base = self._normalize_loki_base(cfg.get("loki_url", ""))
        headers = {}
        if cfg.get("loki_token"):
            headers["Authorization"] = f"Bearer {cfg['loki_token']}"
        ready_url = f"{base}/ready"
        labels_url = f"{base}/loki/api/v1/labels"
        query_url = f"{base}/loki/api/v1/query"
        try:
            async with ctx.typing():
                async with self.session.get(ready_url, headers=headers, timeout=10) as r1:
                    r1s = r1.status
                async with self.session.get(labels_url, headers=headers, timeout=10) as r2:
                    r2s = r2.status
                params = {"query": '{app="discord-bot"}', "time": str(int(time.time() * 1e9))}
                async with self.session.get(query_url, params=params, headers=headers, timeout=10) as r3:
                    r3s = r3.status
            await ctx.send(f"✅ Base: `{base}`\n/ready → {r1s}\n/labels → {r2s}\n/query → {r3s}")
        except Exception as e:
            await ctx.send(f"❌ lokiverify failed: `{e}`")

    @april.command(name="lokitest")
    async def lokitest_cmd(self, ctx: commands.Context):
        """Quick smoke query over last 5m."""
        try:
            end_ns = int(time.time() * 1_000_000_000)
            start_ns = end_ns - 5 * 60 * 1_000_000_000
            data = await self._loki_query_range('{app="discord-bot"}', start_ns, end_ns, 5)
            count = sum(len(s.get("values", [])) for s in data.get("data", {}).get("result", []))
            await ctx.send(f"🧪 lokitest ok: {count} entries in last 5m.")
        except Exception as e:
            await ctx.send(f"❌ lokitest failed: `{e}`")

    # ---- Loki fetch + recall
    @april.command(name="fetch")
    @commands.guild_only()
    async def april_fetch(self, ctx: commands.Context, user_mention: str, since: Optional[str] = "24h", limit: Optional[int] = 20):
        """Fetch recent messages for @user from Loki and cache as recall memory.
        Usage: .april fetch @user [since] [limit]
        since: 1h, 6h, 24h, 7d  /  limit: 1..200
        """
        if not await self._is_allowed_to_interact(ctx):
            return
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

                rows = []
                for s in streams:
                    for ts, line in s.get("values", []):
                        rows.append((int(ts), line))
                rows.sort(key=lambda x: x[0], reverse=True)
                rows = rows[:limit]

                lines_out = []
                recall_lines = []
                display_name = None
                for ts, line in rows:
                    content = None
                    author_name = None
                    # Each line is the JSON payload you wrote
                    try:
                        obj = json.loads(line)
                        content = obj.get("content")
                        author = obj.get("author") or {}
                        author_name = author.get("name")
                    except Exception:
                        # fallback tiny regex
                        m = re.search(r'"content"\s*:\s*"([^"]*)"', line)
                        content = m.group(1) if m else line
                        m2 = re.search(r'"author"\s*:\s*{[^}]*"name"\s*:\s*"([^"]*)"', line)
                        author_name = m2.group(1) if m2 else None
                    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts / 1_000_000_000))
                    who = author_name or uname or uid or "user"
                    display_name = display_name or who
                    text = content if content is not None else line
                    lines_out.append(f"**{who}** at {when}: {text}")
                    recall_lines.append(text)

                ch = ctx.channel.id
                if ch not in self.recall:
                    self.recall[ch] = {}
                key = display_name or (uname or uid or "user")
                self.recall[ch][key] = recall_lines

                for page in pagify("\n".join(lines_out), delims=["\n"], page_length=1800):
                    await ctx.send(page)

                await ctx.send(f"🧠 Cached {len(recall_lines)} lines as recall for **{key}**. I’ll use them in next replies.")
        except Exception as e:
            await ctx.send(f"⚠️ Loki query failed: `{e}`")

    # ---- Voice helpers
    @april.command(name="join")
    async def join_voice(self, ctx: commands.Context):
        if not await self._is_allowed_to_interact(ctx):
            return
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a voice channel.")
        channel = ctx.author.voice.channel
        perms = channel.permissions_for(ctx.me)
        if not perms.connect or not perms.speak:
            return await ctx.send("❌ I need permissions to connect and speak!")
        try:
            if not hasattr(lavalink, "get_player"):
                return await ctx.send("❌ Lavalink not initialized. `[p]load audio`")
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                if self.get_player_channel_id(player) == channel.id:
                    return await ctx.send(f"✅ Already in {channel.name}")
                await player.disconnect()
            await lavalink.connect(channel)
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await ctx.send(f"🔊 Joined {channel.name}")
            else:
                await ctx.send("❌ Failed to connect")
        except Exception as e:
            await ctx.send(f"❌ Join failed: {e}")

    @april.command(name="leave")
    async def leave_voice(self, ctx: commands.Context):
        if not await self._is_allowed_to_interact(ctx):
            return
        try:
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await player.stop()
                await player.disconnect()
                await ctx.send("👋 Disconnected")
            else:
                await ctx.send("❌ Not connected.")
        except Exception as e:
            await ctx.send(f"❌ Leave failed: {e}")

    @april.command(name="clearhistory")
    async def clear_history(self, ctx: commands.Context):
        if not await self._is_allowed_to_interact(ctx):
            return
        ch = ctx.channel.id
        if ch in self.history:
            self.history[ch].clear()
        await ctx.send("✅ Channel history cleared.")

    @april.command(name="smert")
    async def toggle_smert_mode(self, ctx: commands.Context):
        if not await self._is_allowed_to_interact(ctx):
            return
        user_cfg = self.config.user(ctx.author)
        current = await user_cfg.smert_mode()
        if not current:
            custom = await user_cfg.custom_anthropic_key()
            globalk = await self.config.anthropic_key()
            if not custom and not globalk:
                return await ctx.send("❌ Smert mode needs an Anthropic API key. `[p]aprilconfig anthropickey <key>` or `[p]apriluser setkey <key>`")
            await user_cfg.smert_mode.set(True)
            await ctx.send("🧠 Smert mode ON.")
        else:
            await user_cfg.smert_mode.set(False)
            await ctx.send("💡 Smert mode OFF.")

    # -----------------------------------
    # Core chat flow
    # -----------------------------------
    async def process_query(self, ctx: commands.Context, input_text: str):
        use_voice = False
        if await self.config.tts_enabled():
            try:
                player = self.get_player(ctx.guild.id)
                use_voice = player and self.is_player_connected(player)
            except Exception:
                use_voice = False

        user_cfg = self.config.user(ctx.author)
        smert_mode = await user_cfg.smert_mode()

        async with ctx.typing():
            try:
                ch = ctx.channel.id
                if ch not in self.history:
                    max_hist = await self.config.max_history()
                    self.history[ch] = deque(maxlen=max_hist * 2)

                system_prompt = (await self.config.smert_prompt()) if smert_mode else (await self.config.system_prompt())

                messages = [{"role": "system", "content": system_prompt}]

                # Inject recall memory
                ch_recall = self.recall.get(ch, {})
                if ch_recall:
                    chunks = []
                    for uname, lines in ch_recall.items():
                        if not lines:
                            continue
                        subset = lines[:5]
                        chunks.append(f"{uname} recent: " + " | ".join(subset))
                    if chunks:
                        messages.append({"role": "system", "content": "[memory] " + " || ".join(chunks)})

                messages.extend(self.history[ch])
                messages.append({"role": "user", "content": input_text})

                if smert_mode:
                    resp = await self._with_limit(self._api_sem, self.query_anthropic(ctx.author.id, messages))
                else:
                    resp = await self._with_limit(self._api_sem, self.query_deepseek(ctx.author.id, messages))

                draw_prompt = self.maybe_extract_draw_prompt(resp)
                clean = re.sub(r"<draw>.*?</draw>", "", resp, flags=re.IGNORECASE | re.DOTALL).strip()

                self.history[ch].append({"role": "user", "content": input_text})
                self.history[ch].append({"role": "assistant", "content": clean})

                tasks = []
                if not (use_voice and not await self.config.text_response_when_voice()):
                    tasks.append(asyncio.create_task(self.send_streamed_response(ctx, clean)))
                if use_voice:
                    tasks.append(asyncio.create_task(self.speak_response(ctx, clean)))
                if draw_prompt:
                    styled = self.style_prompt(draw_prompt)

                    async def _draw_and_send():
                        try:
                            png = await self._with_limit(self._api_sem, self.generate_openai_image_png(styled, size="1024x1024"))
                            await ctx.send(file=discord.File(BytesIO(png), filename="april_draw.png"), content="pic related:")
                        except Exception as e:
                            await ctx.send(f"⚠️ Couldn't render image: `{e}`")

                    tasks.append(asyncio.create_task(_draw_and_send()))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                await ctx.send(f"❌ Error: {e}")

    async def send_streamed_response(self, ctx: commands.Context, resp: str):
        max_len = await self.config.max_message_length()
        use_gifs = await self.config.use_gifs()
        emotion = await self.detect_emotion(resp) if use_gifs else None
        gif_url = random.choice(EMOTION_GIFS[emotion]) if emotion and random.random() < 0.3 else None

        if len(resp) > max_len:
            # chunk by message length
            words = resp.split()
            chunks, cur, cur_len = [], [], 0
            for w in words:
                if cur_len + len(w) + 1 > max_len:
                    chunks.append(" ".join(cur))
                    cur, cur_len = [w], len(w)
                else:
                    cur.append(w)
                    cur_len += len(w) + 1
            if cur:
                chunks.append(" ".join(cur))
            for i, chunk in enumerate(chunks):
                suffix = f"\n{gif_url}" if gif_url and i == len(chunks) - 1 else ""
                await ctx.send(chunk + suffix)
                await asyncio.sleep(0.2)
        else:
            embed = discord.Embed(description="💭 *Thinking...*")
            msg = await ctx.send(embed=embed)
            size = getattr(self, "_chunk_size", 160)
            delay = getattr(self, "_edit_delay", 0.025)
            for i in range(0, len(resp), size):
                chunk = resp[:i + size]
                if i + size >= len(resp) and gif_url:
                    await msg.edit(content=chunk + f"\n{gif_url}", embed=None)
                else:
                    await msg.edit(content=chunk + "▌", embed=None)
                await asyncio.sleep(delay)
            await msg.edit(content=resp + (f"\n{gif_url}" if gif_url else ""), embed=None)

    # -----------------------------------
    # TTS + images + model backends
    # -----------------------------------
    def clean_text_for_tts(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1000].rsplit(' ', 1)[0] + "..." if len(text) > 1000 else text

    async def speak_response(self, ctx: commands.Context, text: str):
        tts_key = await self.config.tts_key()
        if not tts_key:
            return
        player = self.get_player(ctx.guild.id)
        if not self.is_player_connected(player):
            return
        clean = self.clean_text_for_tts(text)
        if not clean:
            return
        if len(clean) > 800:
            clean = clean[:800].rsplit(' ', 1)[0] + "..."
        audio = await self._with_limit(self._tts_sem, self.generate_tts_audio(clean, tts_key))
        if not audio:
            return
        localtrack_dir = cog_data_path(self).parent / "Audio" / "localtracks" / "april_tts"
        localtrack_dir.mkdir(parents=True, exist_ok=True)
        filename = f"tts_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
        filepath = localtrack_dir / filename
        with open(filepath, "wb") as f:
            f.write(audio)
        audio_cog = self.bot.get_cog("Audio")
        if audio_cog:
            await audio_cog.command_play(ctx, query=f"localtracks/april_tts/{filename}")

    async def generate_tts_audio(self, text: str, api_key: str) -> Optional[bytes]:
        voice_id = await self.config.voice_id()
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}
        async with self.session.post(url, json=payload, headers=headers, timeout=30) as r:
            if r.status == 200:
                return await r.read()
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
                msg = err.get("error", {}).get("message", f"HTTP {r.status}")
                raise RuntimeError(f"OpenAI Images API error: {msg}")
            data = await r.json()
            return base64.b64decode(data["data"][0]["b64_json"])

    async def query_deepseek(self, user_id: int, messages: list) -> str:
        key = await self.config.deepseek_key()
        if not key:
            raise Exception("DeepSeek API key not set. Use `[p]aprilconfig deepseekkey <key>`")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": messages,
            "temperature": await self.config.temperature(),
            "max_tokens": await self.config.max_tokens(),
            "user": str(user_id),
        }
        async with self.session.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=60) as r:
            if r.status != 200:
                data = await r.json()
                raise Exception(data.get("error", {}).get("message", f"HTTP {r.status}"))
            data = await r.json()
            return data["choices"][0]["message"]["content"].strip()

    async def query_anthropic(self, user_id: int, messages: list) -> str:
        # prefer user key if provided
        key = None
        user = self.bot.get_user(user_id)
        if user:
            custom = await self.config.user(user).custom_anthropic_key()
            key = custom or await self.config.anthropic_key()
        else:
            key = await self.config.anthropic_key()
        if not key:
            raise Exception("Anthropic API key not set. Use `[p]aprilconfig anthropickey <key>` or `[p]apriluser setkey <key>`")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        system_content = None
        convo = []
        for m in messages:
            if m["role"] == "system":
                system_content = m["content"]
            else:
                convo.append(m)
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": convo,
            "max_tokens": await self.config.max_tokens(),
            "temperature": await self.config.temperature(),
        }
        if system_content:
            payload["system"] = system_content
        async with self.session.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=60) as r:
            if r.status != 200:
                raise Exception(f"Anthropic API Error {r.status}: {await r.text()}")
            data = await r.json()
            return data["content"][0]["text"].strip()

    # -----------------------------------
    # Loki client (normalize + query)
    # -----------------------------------
    def _normalize_loki_base(self, configured_url: str) -> str:
        raw = (configured_url or "").rstrip("/")
        if raw.endswith("/loki/api/v1/push"):
            return raw[: -len("/loki/api/v1/push")]
        return raw

    def _since_to_ns(self, since: str) -> int:
        m = re.fullmatch(r"(\d+)([hd])", since or "24h")
        n = int(m.group(1)) if m else 24
        unit = m.group(2) if m else "h"
        sec = n * 3600 if unit == "h" else n * 86400
        return (int(time.time()) - sec) * 1_000_000_000

    def _mention_to_user(self, mention: str):
        m = re.fullmatch(r"<@!?(\d+)>", mention.strip())
        if m:
            uid = int(m.group(1))
            user = self.bot.get_user(uid)
            return str(uid), (user.name if user else None)
        return None, mention.lstrip("@")

    def _build_logql_for_user(self, *, guild_id: int, channel_id: int, user_id: Optional[str], user_name: Optional[str]) -> str:
        base = f'{{app="discord-bot",event_type="message",guild_id="{guild_id}",channel_id="{channel_id}"}} | json'
        if user_id:
            return base + f' | author.id="{user_id}"'
        if user_name:
            return base + f' | author.name="{user_name}"'
        return base

    async def _loki_query_range(self, query: str, start_ns: int, end_ns: int, limit: int = 50):
        cfg = await self.config.all()
        base = self._normalize_loki_base(cfg.get("loki_url", ""))
        url = f"{base}/loki/api/v1/query_range"
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
                body = await r.text()
                raise RuntimeError(f"Loki {r.status} at {url}: {body}")
            return await r.json()

    # -----------------------------------
    # Special flow (fun draw)
    # -----------------------------------
    async def _draw_what_you_think(self, ctx: commands.Context):
        async with ctx.typing():
            try:
                ch = ctx.channel.id
                if ch not in self.history:
                    max_hist = await self.config.max_history()
                    self.history[ch] = deque(maxlen=max_hist * 2)
                system_prompt = await self.config.system_prompt()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.history[ch])
                messages.append({"role": "user", "content": "User: I know what you're thinking, draw me it!"})
                chat_resp = await self._with_limit(self._api_sem, self.query_deepseek(ctx.author.id, messages))
                img_prompt = await self._with_limit(self._api_sem, self.query_deepseek(0, [
                    {"role": "system", "content": "You write exactly ONE single-line image prompt; no commentary, no quotes."}
                ] + messages[-10:] + [{"role": "assistant", "content": chat_resp}]))
                styled = self.style_prompt(img_prompt)
                png = await self._with_limit(self._api_sem, self.generate_openai_image_png(styled, size="1024x1024"))
                await ctx.send(file=discord.File(BytesIO(png), filename="april_draw.png"), content=f"{chat_resp}\n\n**Image:** {styled}")
                self.history[ch].append({"role": "user", "content": "I know what you're thinking, draw me it!"})
                self.history[ch].append({"role": "assistant", "content": chat_resp + f"\n[Image: {styled}]"})
            except Exception as e:
                await ctx.send(f"⚠️ Couldn't complete the draw-what-you-think flow: `{e}`")

    # -----------------------------------
    # Minimal config view (owner can still use your april_log's own commands)
    # -----------------------------------
    @commands.group(name="aprilconfig", aliases=["aprilcfg"])
    @commands.is_owner()
    async def aprilconfig(self, ctx: commands.Context):
        if ctx.invoked_subcommand is None:
            await self.show_settings(ctx)

    @aprilconfig.command(name="settings")
    async def show_settings(self, ctx: commands.Context):
        cfg = await self.config.all()
        e = discord.Embed(title="AprilTalk Configuration")
        def tail(v): return f"`...{v[-4:]}`" if v else "❌ Not set"
        e.add_field(name="DeepSeek Key", value=tail(cfg.get("deepseek_key")), inline=False)
        e.add_field(name="Anthropic Key", value=tail(cfg.get("anthropic_key")), inline=False)
        e.add_field(name="ElevenLabs Key", value=tail(cfg.get("tts_key")), inline=False)
        e.add_field(name="OpenAI Key", value=tail(cfg.get("openai_key")), inline=False)
        e.add_field(name="Voice ID", value=f"`{cfg['voice_id']}`", inline=True)
        e.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        e.add_field(name="Temperature", value=f"`{cfg['temperature']}`", inline=True)
        e.add_field(name="Max Tokens", value=f"`{cfg['max_tokens']}`", inline=True)
        e.add_field(name="Max History", value=f"`{cfg['max_history']} exchanges`", inline=True)
        e.add_field(name="Max Message Length", value=f"`{cfg['max_message_length']}`", inline=True)
        e.add_field(name="TTS Enabled", value="✅" if cfg['tts_enabled'] else "❌", inline=True)
        e.add_field(name="Text with Voice", value="✅" if cfg['text_response_when_voice'] else "❌", inline=True)
        e.add_field(name="Emotion GIFs", value="✅" if cfg['use_gifs'] else "❌", inline=True)
        e.add_field(name="Loki URL", value=f"{cfg.get('loki_url','')}", inline=False)
        e.add_field(name="Sleep Mode", value="😴 ON" if cfg.get("sleep_enabled") else "💬 OFF", inline=True)
        e.add_field(name="Sleep Allowed User", value=f"<@{cfg.get('sleep_user_id','')}>" if cfg.get("sleep_user_id") else "—", inline=True)
        sp = cfg['system_prompt'][:200] + ("..." if len(cfg['system_prompt']) > 200 else "")
        sm = cfg['smert_prompt'][:200] + ("..." if len(cfg['smert_prompt']) > 200 else "")
        e.add_field(name="System Prompt", value=f"```{sp}```", inline=False)
        e.add_field(name="Smert Prompt", value=f"```{sm}```", inline=False)
        await ctx.send(embed=e)

# -----------------------------------
# Setup
# -----------------------------------
async def setup(bot: Red):
    await bot.add_cog(AprilTalk(bot))
