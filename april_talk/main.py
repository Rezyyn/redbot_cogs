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
from datetime import timezone, datetime
from collections import deque
from pathlib import Path
from io import BytesIO
from typing import Optional, List, Dict

from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import pagify

log = logging.getLogger("red.apriltalk")
log.setLevel(logging.DEBUG)

STYLE_SUFFIX = ", in a futuristic neo-cyberpunk aesthetic"
TENOR_ENDPOINT = "https://tenor.googleapis.com/v2/search"

FALLBACK_GIFS = {
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
    """April: chat + voice + Loki recall + Tenor GIFs + Vision intake + context-aware drawing"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1398462)

        self.session = aiohttp.ClientSession()

        # Memory
        self.history: Dict[int, deque] = {}                 # channel_id -> deque of chat messages
        self.recall: Dict[int, Dict[str, List[str]]] = {}   # channel_id -> {user: [lines]}
        self.ctx_state: Dict[int, Dict] = {}                # channel_id -> {"last_subject": str, "last_image_prompt": str}

        # Secret short-term memory (from Loki)
        # channel_id -> { user_id: {"ts": int, "lines": [str], "fetched_at": float} }
        self.short_mem: Dict[int, Dict[int, Dict]] = {}
        self._short_mem_ttl = 20.0

        # Concurrency
        self._api_sem = asyncio.Semaphore(3)
        self._tts_sem = asyncio.Semaphore(1)
        self._edit_delay = 0.05
        self._chunk_size = 240

        # TTS files
        self.tts_dir = Path(cog_data_path(self)) / "tts"
        self.tts_dir.mkdir(exist_ok=True, parents=True)
        self.tts_files = set()

        # Defaults
        self.config.register_global(
            deepseek_key="",
            anthropic_key="",
            openai_key="",
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
            gif_prob=0.30,
            max_message_length=1800,
            # Loki (query)
            loki_url="http://localhost:3100",
            loki_token="",
            # Loki (push)
            loki_push_url="http://localhost:3100/loki/api/v1/push",
            loki_push_enabled=True,
            # Tenor
            tenor_key="",
            tenor_enabled=True,
            # Sleep
            sleep_enabled=False,
            sleep_user_id="165548483128983552",
            # Vision
            vision_provider="openai",
            # Audio localtracks config
            localtracks_root="",          # absolute path that matches .audioset localpath
            localtracks_subdir="april",   # subfolder under localtracks root for TTS files
            tts_archive_after=90,         # seconds before archiving the played TTS file
            tts_archive_keep=True, 
        )

        self.config.register_user(
            smert_mode=False,
            custom_anthropic_key="",
            custom_smert_prompt="",
        )

    # ---------------- Lifecycle ----------------

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

    # ---------------- Utils ----------------

    def _ctx(self, channel_id: int) -> Dict:
        if channel_id not in self.ctx_state:
            self.ctx_state[channel_id] = {"last_subject": None, "last_image_prompt": None}
        return self.ctx_state[channel_id]

    async def _with_limit(self, sem: asyncio.Semaphore, coro):
        async with sem:
            return await coro

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
            if hasattr(player, "channel") and player.channel:
                return True
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

    # Prompt / intent helpers
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

    def _is_draw_intent(self, text: str) -> bool:
        t = text.lower().strip()
        triggers = [
            "draw ", "generate an image", "make an image", "create an image",
            "paint ", "illustrate ", "render ", "/imagine", "<draw>", "image of ",
        ]
        return any(x in t for x in triggers)

    def _looks_like_vague_draw(self, text: str) -> bool:
        t = text.lower().strip()
        vague = ["draw it", "draw that", "draw this", "draw one", "draw something"]
        return any(t.startswith(v) or t == v for v in vague) or (len(t) <= 12 and "draw" in t)

    def _extract_draw_text(self, text: str) -> str:
        dp = self.maybe_extract_draw_prompt(text)
        if dp:
            return dp
        t = re.sub(r"^(april\s+)?(draw|/imagine)\s*", "", text.strip(), flags=re.IGNORECASE)
        return t.strip()

    def _looks_like_memory_request(self, text: str) -> bool:
        t = text.lower()
        keys = [
            "search your history", "search my history", "look in your logs",
            "what did i say", "what did we say", "what did", "remind me when",
            "find where i said", "from history", "from the logs", "from loki",
        ]
        return any(k in t for k in keys)

    async def detect_emotion(self, text: str) -> Optional[str]:
        t = text.lower()
        if any(w in t for w in ["happy", "great", "awesome", "excellent", "wonderful", "amazing"]): return "happy"
        if any(w in t for w in ["think", "consider", "ponder", "wonder", "hmm"]): return "thinking"
        if any(w in t for w in ["confused", "don't understand", "what", "huh", "unclear"]): return "confused"
        if any(w in t for w in ["excited", "can't wait", "wow", "amazing!"]): return "excited"
        if any(w in t for w in ["sad", "sorry", "unfortunately", "regret"]): return "sad"
        return None

    async def _pick_gif(self, emotion: Optional[str]) -> Optional[str]:
        cfg = await self.config.all()
        if not cfg.get("use_gifs", True):
            return None
        if random.random() > float(cfg.get("gif_prob", 0.3)):
            return None
        tenor_key = cfg.get("tenor_key") or ""
        tenor_enabled = bool(cfg.get("tenor_enabled", True) and tenor_key)
        if not emotion:
            return None

        if tenor_enabled:
            try:
                params = {
                    "q": emotion,
                    "key": tenor_key,
                    "limit": 25,
                    "media_filter": "minimal",
                    "random": "true",
                    "contentfilter": "high",
                }
                async with self.session.get(TENOR_ENDPOINT, params=params, timeout=10) as r:
                    if r.status == 200:
                        data = await r.json()
                        results = data.get("results", [])
                        if results:
                            cand = random.choice(results)
                            media = cand.get("media_formats") or {}
                            for k in ["gif", "tinygif", "nanogif", "mediumgif", "mp4", "tinygifpreview"]:
                                if k in media and "url" in media[k]:
                                    return media[k]["url"]
            except Exception as e:
                log.debug(f"Tenor fetch failed: {e}")

        arr = FALLBACK_GIFS.get(emotion, [])
        return random.choice(arr) if arr else None
    # --- Add this helper below other utils ---
    async def _ensure_localtracks_registered(self, audio_cog, ctx, root: str):
        """
        Make sure Audio cog sees new files:
        1) If Audio provides a scan command/function, call it.
        2) If not, attempt to run via command group if present.
        """
        try:
            # Newer Audio cog exposes a scan command method
            if hasattr(audio_cog, "localtracks_scan"):
                await audio_cog.localtracks_scan(ctx)
                return True
            # Some versions expose it under a command_ name
            if hasattr(audio_cog, "command_localtracks_scan"):
                await audio_cog.command_localtracks_scan(ctx)
                return True
        except Exception:
            pass
    
        # Worst case, send a hint (non-blocking)
        try:
            await ctx.send("🔎 Scanning localtracks so I can play the new TTS…")
            # As a user would: `localtracks scan` (if the command group is available)
            if hasattr(self.bot, "get_command"):
                cmd = self.bot.get_command("localtracks scan")
                if cmd:
                    await ctx.invoke(cmd)
                    return True
        except Exception:
            pass
        return False
    
    
    # --- Add this helper to archive (or delete) after playback ---
    async def _archive_or_delete_tts(self, fpath: Path):
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
                    # fallback copy+unlink
                    import shutil
    
                    shutil.copy2(fpath, target)
                    fpath.unlink(missing_ok=True)
            else:
                fpath.unlink(missing_ok=True)
        except Exception:
            pass

    # ---------------- Loki: query/push ----------------

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

    def _escape_regex(self, s: str) -> str:
        return re.sub(r'([.^$*+?{}\[\]\\|()])', r'\\\1', s)

    def _build_logql_for_user(self, *, guild_id: int, channel_id: int, user_id: Optional[str], user_name: Optional[str]) -> str:
        base = f'{{app="discord-bot",event_type="message",guild_id="{guild_id}",channel_id="{channel_id}"}} | json'
        if user_id:
            return base + f' | author_id = "{user_id}"'
        if user_name:
            safe = self._escape_regex(user_name)
            return base + f' | author_name =~ "^{safe}$"'
        return base

    def _build_recent_user_logql(self, guild_id: int, channel_id: int, user_id: int) -> str:
        return (
            f'{{app="discord-bot",event_type="message",guild_id="{guild_id}",channel_id="{channel_id}"}} '
            f'| json | author_id = "{user_id}"'
        )

    async def _get_loki_headers(self):
        headers = {"Content-Type": "application/json"}
        token = await self.config.loki_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _loki_query_range(self, query: str, start_ns: int, end_ns: int, limit: int = 50):
        cfg = await self.config.all()
        base = self._normalize_loki_base(cfg.get("loki_url", ""))
        url = f"{base}/loki/api/v1/query_range"
        headers = await self._get_loki_headers()
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

    async def _loki_push(self, streams):
        if not await self.config.loki_push_enabled():
            return
        push_url = await self.config.loki_push_url()
        if not push_url:
            return
        headers = await self._get_loki_headers()
        payload = {"streams": streams}
        try:
            async with self.session.post(push_url, json=payload, headers=headers, timeout=20) as r:
                if r.status != 204:
                    body = await r.text()
                    log.warning(f"Loki push {r.status}: {body}")
        except Exception as e:
            log.warning(f"Loki push failed: {e}")

    async def _loki_search_snippets(self, ctx: commands.Context, query_text: str, since: str = "30d", k: int = 20) -> List[Dict]:
        end_ns = int(time.time() * 1_000_000_000)
        start_ns = self._since_to_ns(since)
        terms = [re.escape(w) for w in re.findall(r"[A-Za-z0-9#@._-]+", query_text) if len(w) > 1]
        regex = "|".join(terms) if terms else "."
        logql = (
            f'{{app="discord-bot",event_type="message",guild_id="{ctx.guild.id}",channel_id="{ctx.channel.id}"}} '
            f'|~ "{regex}" | json'
        )
        try:
            data = await self._with_limit(self._api_sem, self._loki_query_range(logql, start_ns, end_ns, limit=max(50, k)))
        except Exception:
            return []
        rows = []
        for s in data.get("data", {}).get("result", []):
            for ts, line in s.get("values", []):
                try:
                    obj = json.loads(line)
                    content = obj.get("content") or ""
                    author = (obj.get("author") or {}).get("name") or "user"
                except Exception:
                    m = re.search(r'"content"\s*:\s*"([^"]*)"', line)
                    content = m.group(1) if m else line
                    m2 = re.search(r'"author"\s*:\s*{[^}]*"name"\s*:\s*"([^"]*)"', line)
                    author = m2.group(1) if m2 else "user"
                rows.append({"ts": int(ts), "who": author, "text": content})
        rows.sort(key=lambda r: r["ts"], reverse=True)
        seen = set()
        uniq = []
        for r in rows:
            key = r["text"].strip().lower()
            if key and key not in seen:
                seen.add(key)
                uniq.append(r)
            if len(uniq) >= k:
                break
        for r in uniq:
            r["when"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"] / 1_000_000_000))
        return uniq

    async def _summarize_snippets_into_facts(self, user_id: int, snippets: List[Dict], goal: str) -> str:
        if not snippets:
            return ""
        bullet_lines = "\n".join(f"- [{s['when']}] {s['who']}: {s['text']}" for s in snippets[:30])
        system = {"role": "system", "content": "You turn chat log snippets into 2–5 concise, relevant facts. Prefer concrete details and dates. No fluff."}
        user = {"role": "user", "content": f"Goal: {goal}\nHere are snippets:\n{bullet_lines}\n\nReturn only the facts as bullets."}
        try:
            return await self._with_limit(self._api_sem, self.query_deepseek(user_id, [system, user]))
        except Exception:
            return ""

    async def _prefetch_recent_user_history(self, guild_id: int, channel_id: int, user_id: int, max_items: int) -> List[str]:
        ch_map = self.short_mem.setdefault(channel_id, {})
        now = time.time()
        cached = ch_map.get(user_id)
        if cached and now - cached.get("fetched_at", 0) <= self._short_mem_ttl:
            return cached.get("lines", [])

        end_ns = int(time.time() * 1_000_000_000)
        start_ns = end_ns - 30 * 24 * 3600 * 1_000_000_000
        logql = self._build_recent_user_logql(guild_id, channel_id, user_id)

        try:
            data = await self._with_limit(self._api_sem, self._loki_query_range(logql, start_ns, end_ns, limit=max_items))
        except Exception:
            ch_map[user_id] = {"lines": [], "fetched_at": now}
            return []

        rows = []
        for s in data.get("data", {}).get("result", []):
            for ts, line in s.get("values", []):
                try:
                    obj = json.loads(line)
                    text = obj.get("content") or ""
                except Exception:
                    m = re.search(r'"content"\s*:\s*"([^"]*)"', line)
                    text = m.group(1) if m else line
                if text:
                    rows.append((int(ts), text))

        rows.sort(key=lambda x: x[0], reverse=True)
        lines = [t for _, t in rows[:max_items]]
        lines.reverse()

        ch_map[user_id] = {"lines": lines, "fetched_at": now}
        return lines

    # ---------------- Live push listeners ----------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        # Push to Loki
        if await self.config.loki_push_enabled():
            ns_timestamp = str(int(message.created_at.replace(tzinfo=timezone.utc).timestamp() * 1e9))
            message_data = {
                "content": message.content,
                "message_id": str(message.id),
                "author": {
                    "id": str(message.author.id),
                    "name": message.author.name,
                    "discriminator": message.author.discriminator,
                    "bot": message.author.bot,
                    "avatar": str(message.author.avatar.url) if message.author.avatar and message.author.avatar.url else None
                },
                "channel": {
                    "id": str(message.channel.id),
                    "name": message.channel.name if hasattr(message.channel, "name") else "DM",
                    "category": message.channel.category.name if hasattr(message.channel, "category") and message.channel.category else None
                },
                "guild": {
                    "id": str(message.guild.id),
                    "name": message.guild.name
                } if message.guild else None,
                "created_at": message.created_at.isoformat(),
                "attachments": [{
                    "url": a.url,
                    "filename": a.filename,
                    "size": a.size,
                    "content_type": a.content_type
                } for a in message.attachments],
                "embeds": [{
                    "type": e.type,
                    "title": e.title,
                    "description": e.description
                } for e in message.embeds]
            }
            stream_labels = {
                "app": "discord-bot",
                "event_type": "message",
                "channel_id": str(message.channel.id),
                "guild_id": str(message.guild.id) if message.guild else "DM",
                "source": "april"
            }
            asyncio.create_task(self._loki_push([{"stream": stream_labels, "values": [[ns_timestamp, json.dumps(message_data)]]}]))

        # Secretly prewarm short-term memory for this speaker
        try:
            if message.guild and not message.author.bot:
                max_hist = await self.config.max_history()
                asyncio.create_task(
                    self._prefetch_recent_user_history(
                        guild_id=message.guild.id,
                        channel_id=message.channel.id,
                        user_id=message.author.id,
                        max_items=max_hist
                    )
                )
        except Exception:
            pass

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        if after.author.bot or not await self.config.loki_push_enabled():
            return
        ns = str(int((after.edited_at or after.created_at).replace(tzinfo=timezone.utc).timestamp() * 1e9))
        edit_data = {
            "event_type": "edit",
            "message_id": str(after.id),
            "channel_id": str(after.channel.id),
            "guild_id": str(after.guild.id) if after.guild else "DM",
            "author_id": str(after.author.id),
            "old_content": before.content,
            "new_content": after.content,
            "edited_at": after.edited_at.isoformat() if after.edited_at else None
        }
        labels = {
            "app": "discord-bot",
            "event_type": "edit",
            "channel_id": str(after.channel.id),
            "guild_id": str(after.guild.id) if after.guild else "DM",
            "source": "april"
        }
        asyncio.create_task(self._loki_push([{"stream": labels, "values": [[ns, json.dumps(edit_data)]]}]))

    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message):
        if message.author.bot or not await self.config.loki_push_enabled():
            return
        ns = str(int(time.time() * 1e9))
        delete_data = {
            "event_type": "delete",
            "message_id": str(message.id),
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id) if message.guild else "DM",
            "author_id": str(message.author.id),
            "content": message.content,
            "created_at": message.created_at.isoformat(),
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }
        labels = {
            "app": "discord-bot",
            "event_type": "delete",
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id) if message.guild else "DM",
            "source": "april"
        }
        asyncio.create_task(self._loki_push([{"stream": labels, "values": [[ns, json.dumps(delete_data)]]}]))

    # ---------------- Commands ----------------

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april(self, ctx: commands.Context, *, input: str):
        if not await self._is_allowed_to_interact(ctx):
            return

        image_urls = self._collect_image_attachments(ctx.message)
        if image_urls:
            return await self._vision_describe(ctx, image_urls, input)

        if self._is_draw_intent(input) or self._looks_like_vague_draw(input):
            return await self._contextual_draw(ctx, input)

        return await self.process_query(ctx, input)

    @april.command(name="draw")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april_draw(self, ctx: commands.Context, *, prompt: str):
        if not await self._is_allowed_to_interact(ctx):
            return
        p = prompt.strip()
        if p.lower().startswith("me "):
            p = p[3:].strip()
        styled = self.style_prompt(p)
        await self._render_and_send_image(ctx, styled)

    @april.command(name="again")
    async def april_draw_again(self, ctx: commands.Context, *, tweak: str = ""):
        ctxobj = self._ctx(ctx.channel.id)
        last = ctxobj.get("last_image_prompt")
        if not last:
            return await ctx.send("I don’t have a previous image prompt in this channel yet.")
        styled = self.style_prompt((last + " " + tweak).strip())
        ctxobj["last_image_prompt"] = styled
        return await self._render_and_send_image(ctx, styled, prefix="again:")

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

    @april.command(name="lokiurl")
    @commands.is_owner()
    async def lokiurl_cmd(self, ctx: commands.Context, url: str):
        await self.config.loki_url.set(url.rstrip("/"))
        await ctx.send(f"✅ Loki URL set to `{url}`")

    @april.command(name="lokitoken")
    @commands.is_owner()
    async def lokitoken_cmd(self, ctx: commands.Context, token: str):
        await self.config.loki_token.set(token)
        await ctx.send("✅ Loki token set")

    @april.command(name="lokipush")
    @commands.is_owner()
    async def lokipush_cmd(self, ctx: commands.Context, url: str):
        await self.config.loki_push_url.set(url)
        await ctx.send(f"✅ Loki push URL set to `{url}`")

    @april.command(name="lokipushon")
    @commands.is_owner()
    async def lokipushon_cmd(self, ctx: commands.Context, enabled: bool):
        await self.config.loki_push_enabled.set(bool(enabled))
        await ctx.send(f"✅ Loki live logging {'ENABLED' if enabled else 'DISABLED'}")

    @april.command(name="lokiverify")
    @commands.is_owner()
    async def lokiverify_cmd(self, ctx: commands.Context):
        cfg = await self.config.all()
        base = self._normalize_loki_base(cfg.get("loki_url", ""))
        headers = await self._get_loki_headers()
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

    @april.command(name="fetch")
    @commands.guild_only()
    async def april_fetch(self, ctx: commands.Context, user_mention: str, since: Optional[str] = "24h", limit: Optional[int] = 20):
        if not await self._is_allowed_to_interact(ctx):
            return
        if not ctx.guild:
            return await ctx.send("❌ Must be used in a server channel.")

        uid, uname = self._mention_to_user(user_mention)
        if not uid and not uname:
            return await ctx.send("❌ I need an @mention or a username.")

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
                    q = (
                        f'{{app="discord-bot",event_type="message",guild_id="{ctx.guild.id}",channel_id="{ctx.channel.id}"}}'
                        f' |= "{uid or uname}"'
                    )
                    data = await self._with_limit(self._api_sem, self._loki_query_range(q, start_ns, end_ns, limit))
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
                    try:
                        obj = json.loads(line)
                        content = obj.get("content")
                        author = obj.get("author") or {}
                        author_name = author.get("name")
                    except Exception:
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

    @april.command(name="recall")
    async def april_recall(self, ctx: commands.Context, *, query: str = "interesting fact  OR plan OR todo"):
        try:
            async with ctx.typing():
                snippets = await self._loki_search_snippets(ctx, query, since="365d", k=30)
                if not snippets:
                    return await ctx.send("🔎 Nothing relevant found in history.")
                facts = await self._summarize_snippets_into_facts(ctx.author.id, snippets, goal=f"Summarize highlights for: {query}")
                if not facts:
                    return await ctx.send("I found entries, but couldn’t summarize them right now.")
                for page in pagify(facts, page_length=1800):
                    await ctx.send(page)
                self._ctx(ctx.channel.id)["last_subject"] = query
        except Exception as e:
            await ctx.send(f"⚠️ recall failed: `{e}`")

    @commands.command(name="prewarm")
    @commands.guild_only()
    async def prewarm(self, ctx: commands.Context, user: Optional[discord.Member] = None):
        user = user or ctx.author
        max_hist = await self.config.max_history()
        lines = await self._prefetch_recent_user_history(ctx.guild.id, ctx.channel.id, user.id, max_hist)
        await ctx.send(f"🔥 Prewarmed {len(lines)} recent lines for {user.display_name}.")

    # Voice
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
                return await ctx.send("❌ Smert mode needs an Anthropic API key. `[p]aprilcfg anthropickey <key>` or `[p]apriluser setkey <key>`")
            await user_cfg.smert_mode.set(True)
            await ctx.send("🧠 Smert mode ON.")
        else:
            await user_cfg.smert_mode.set(False)
            await ctx.send("💡 Smert mode OFF.")

    # ---------------- Chat flow ----------------

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

                # Short-term memory (secretly from Loki) for author
                author_lines = []
                try:
                    if ctx.guild:
                        max_hist = await self.config.max_history()
                        author_lines = await self._prefetch_recent_user_history(
                            guild_id=ctx.guild.id,
                            channel_id=ctx.channel.id,
                            user_id=ctx.author.id,
                            max_items=max_hist
                        )
                except Exception:
                    author_lines = []

                system_prompt = (await self.config.smert_prompt()) if smert_mode else (await self.config.system_prompt())
                messages = [{"role": "system", "content": system_prompt}]

                if author_lines:
                    mem_text = "Recent things this user said here:\n" + "\n".join(f"- {ln}" for ln in author_lines)
                    messages.append({"role": "system", "content": mem_text})

                # Deep history only when asked
                if self._looks_like_memory_request(input_text):
                    topic = re.sub(r"^(april|hey|hi|please|can you)\s+", "", input_text, flags=re.I).strip()
                    snippets = await self._loki_search_snippets(ctx, topic or "interesting|fact|todo|plan", since="365d", k=30)
                    loki_context_text = await self._summarize_snippets_into_facts(ctx.author.id, snippets, goal="Provide an interesting, relevant fact this user might care about.")
                    if loki_context_text:
                        messages.append({"role": "system", "content": f"[history-facts]\n{loki_context_text}"})
                        messages.append({"role": "assistant", "content": "I just looked back through your channel history and pulled some highlights; I’ll weave them into my answer naturally."})

                # Inject cached recall
                ch_recall = self.recall.get(ch, {})
                if ch_recall:
                    mems = []
                    for uname, lines in ch_recall.items():
                        if not lines:
                            continue
                        subset = lines[:5]
                        mems.append(f"{uname} recent: " + " | ".join(subset))
                    if mems:
                        messages.append({"role": "system", "content": "[memory] " + " || ".join(mems)})

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

                ctxobj = self._ctx(ch)
                m = re.search(r"([A-Z][^.!?\n]{5,60})", clean)
                if m:
                    ctxobj["last_subject"] = m.group(1).strip()

                # Async GIF selection, do not block streaming
                gif_url = None

                async def _gif():
                    nonlocal gif_url
                    try:
                        emotion = await self.detect_emotion(clean)
                        gif_url = await self._pick_gif(emotion)
                    except Exception:
                        gif_url = None

                task_gif = asyncio.create_task(_gif())

                tasks = []
                if not (use_voice and not await self.config.text_response_when_voice()):
                    tasks.append(asyncio.create_task(self.send_streamed_response(ctx, clean, task_gif=task_gif)))
                if use_voice:
                    tasks.append(asyncio.create_task(self.speak_response(ctx, clean)))
                if draw_prompt:
                    styled = self.style_prompt(draw_prompt)
                    ctxobj["last_image_prompt"] = styled
                    tasks.append(asyncio.create_task(self._render_and_send_image(ctx, styled, prefix="Let me sketch this")))
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                await ctx.send(f"❌ Error: {e}")

    async def send_streamed_response(self, ctx: commands.Context, resp: str, task_gif: Optional[asyncio.Task] = None):
        max_len = await self.config.max_message_length()
        gif_url = None

        if len(resp) > max_len:
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

            # Wait for GIF only before last send
            if task_gif:
                try:
                    await task_gif
                    gif_url = task_gif.result()
                except Exception:
                    gif_url = None

            for i, chunk in enumerate(chunks):
                suffix = f"\n{gif_url}" if gif_url and i == len(chunks) - 1 else ""
                await ctx.send(chunk + suffix)
                await asyncio.sleep(0.2)
        else:
            embed = discord.Embed(description="💭 *Thinking...*")
            msg = await ctx.send(embed=embed)
            size = getattr(self, "_chunk_size", 240)
            delay = getattr(self, "_edit_delay", 0.05)
            for i in range(0, len(resp), size):
                chunk = resp[:i + size]
                await msg.edit(content=chunk + ("▌" if i + size < len(resp) else ""), embed=None)
                await asyncio.sleep(delay)

            if task_gif:
                try:
                    await task_gif
                    gif_url = task_gif.result()
                except Exception:
                    gif_url = None

            await msg.edit(content=resp + (f"\n{gif_url}" if gif_url else ""), embed=None)

    # ---------------- TTS / Images / Models / Vision ----------------

    def clean_text_for_tts(self, text: str) -> str:
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1000].rsplit(' ', 1)[0] + "..." if len(text) > 1000 else text

async def speak_response(self, ctx: commands.Context, text: str):
    """
    Generate TTS, save under configured localtracks root/subdir, scan localtracks,
    then play explicitly using the Audio cog's play command with a localtracks/<...> query.
    Never fall back to generic search (which can hit YouTube).
    """
    tts_key = await self.config.tts_key()
    if not tts_key:
        return  # silently skip

    # must be in voice
    player = self.get_player(ctx.guild.id)
    if not self.is_player_connected(player):
        return

    clean = self.clean_text_for_tts(text)
    if not clean:
        return
    if len(clean) > 900:
        clean = clean[:900].rsplit(" ", 1)[0] + "..."

    # generate audio
    audio = await self._with_limit(
        self._tts_sem, self.generate_tts_audio(clean, tts_key)
    )
    if not audio:
        try:
            await ctx.send("⚠️ TTS failed to generate audio.")
        except Exception:
            pass
        return

    # determine localtracks location
    root = (await self.config.localtracks_root()).strip()
    subdir = (await self.config.localtracks_subdir()).strip() or "april"

    if not root:
        # hard fallback to standard Red path if user didn't configure a custom localtracks_root
        base = cog_data_path(self).parent / "Audio" / "localtracks"
        outdir = (
            base / "april_tts"
        )  # distinct from user's custom subdir to avoid confusion
        query_prefix = "localtracks/april_tts"
    else:
        outdir = Path(root) / subdir
        query_prefix = f"localtracks/{subdir}"

    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"tts_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
    fpath = outdir / fname

    try:
        with open(fpath, "wb") as f:
            f.write(audio)
    except Exception as e:
        try:
            await ctx.send(f"❌ Failed to write TTS file: `{e}`")
        except Exception:
            pass
        return

    audio_cog = self.bot.get_cog("Audio")
    if not audio_cog:
        try:
            await ctx.send(f"✅ TTS saved at `{fpath}` but Audio cog is not loaded.")
        except Exception:
            pass
        return

    # Ensure Audio has indexed the new file
    await self._ensure_localtracks_registered(
        audio_cog, ctx, root or str(outdir.parent)
    )

    # Build a strictly-localtracks query to avoid any provider search
    query = f"{query_prefix}/{fname}"

    # Try to use the Audio cog's 'play' command function directly if available
    try:
        if hasattr(audio_cog, "command_play"):
            await audio_cog.command_play(ctx, query=query)
        else:
            # Try to resolve the 'play' command and call it with our localtracks query.
            cmd = getattr(self.bot, "get_command", lambda *_: None)("play")
            if cmd:
                await ctx.invoke(cmd, query=query)
            else:
                # As a last resort, tell the user how to play it manually without risking YT
                await ctx.send(
                    f"▶️ To play the generated TTS manually, run:\n`{ctx.prefix}play {query}`"
                )
    except Exception as e:
        try:
            await ctx.send(
                f"❌ Couldn’t play `{query}`: `{e}`\n"
                f"• Ensure `.audioset localpath` is set to `{root or (cog_data_path(self).parent / 'Audio' / 'localtracks')}`\n"
                f"• File exists: `{fpath}`"
            )
        except Exception:
            pass
        # still archive/delete later
        asyncio.create_task(self._archive_or_delete_tts(fpath))
        return

    # Archive/delete the file later (gives Lavalink time to start/finish)
    asyncio.create_task(self._archive_or_delete_tts(fpath))


    async def generate_tts_audio(self, text: str, api_key: str) -> Optional[bytes]:
        """
        ElevenLabs TTS:
        - Use Accept: audio/mpeg so the API returns raw audio.
        - Provide a model_id (turbo or multilingual) to avoid 400s.
        - Return None on failure.
        """
        voice_id = await self.config.voice_id()
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "model_id": "eleven_turbo_v2",  # or "eleven_multilingual_v2" if you need
            "text": text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
        }
        try:
            async with self.session.post(
                url, json=payload, headers=headers, timeout=45
            ) as r:
                if r.status == 200:
                    return await r.read()
                else:
                    # read the error body so the probe can display it
                    err = await r.text()
                    log.warning(f"ElevenLabs TTS error {r.status}: {err}")
                    return None
        except Exception as e:
            log.warning(f"ElevenLabs call failed: {e}")
            return None


    async def generate_openai_image_png(self, prompt: str, size: str = "1024x1024") -> bytes:
        key = await self.config.openai_key()
        if not key:
            raise RuntimeError("OpenAI API key not set. Use `[p]aprilcfg openaikey <key>`.")
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
            raise Exception("DeepSeek API key not set. Use `[p]aprilcfg deepseekkey <key>`")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": messages,
            "temperature": await self.config.temperature(),
            "max_tokens": min(1024, await self.config.max_tokens()),
            "user": str(user_id),
        }
        async with self.session.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers, timeout=60) as r:
            if r.status != 200:
                try:
                    data = await r.json()
                    msg = data.get("error", {}).get("message", f"HTTP {r.status}")
                except Exception:
                    msg = f"HTTP {r.status}"
                raise Exception(msg)
            data = await r.json()
            return data["choices"][0]["message"]["content"].strip()

    async def query_anthropic(self, user_id: int, messages: list) -> str:
        user = self.bot.get_user(user_id)
        if user:
            custom = await self.config.user(user).custom_anthropic_key()
            key = custom or await self.config.anthropic_key()
        else:
            key = await self.config.anthropic_key()
        if not key:
            raise Exception("Anthropic API key not set. Use `[p]aprilcfg anthropickey <key>` or `[p]apriluser setkey <key>`")
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
            "max_tokens": min(1024, await self.config.max_tokens()),
            "temperature": await self.config.temperature(),
        }
        if system_content:
            payload["system"] = system_content
        async with self.session.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=60) as r:
            if r.status != 200:
                raise Exception(f"Anthropic API Error {r.status}: {await r.text()}")
            data = await r.json()
            return data["content"][0]["text"].strip()

    # Vision
    def _collect_image_attachments(self, message: discord.Message) -> List[str]:
        urls = []
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                urls.append(att.url)
            else:
                if any(att.filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    urls.append(att.url)
        return urls

    async def _vision_describe(self, ctx: commands.Context, image_urls: List[str], user_text: str):
        provider = (await self.config.vision_provider()) or "openai"
        if provider != "openai":
            return await ctx.send("⚠️ Vision provider not supported yet. Set `[p]aprilcfg vision openai`.")
        key = await self.config.openai_key()
        if not key:
            return await ctx.send("⚠️ Set an OpenAI key first: `[p]aprilcfg openaikey <key>`")

        messages = [
            {"role": "system", "content": "You are April, a helpful AI assistant. Briefly and helpfully describe the image(s), then answer the user's question if any. Be concise and kind."},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text or "Describe the image."}] + [{"type": "image_url", "image_url": {"url": u}} for u in image_urls]
            }
        ]
        model = "gpt-4o-mini"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": messages, "max_tokens": 600}
        try:
            async with ctx.typing():
                async with self.session.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=60) as r:
                    if r.status != 200:
                        txt = await r.text()
                        return await ctx.send(f"❌ Vision error: {txt}")
                    data = await r.json()
                    text = data["choices"][0]["message"]["content"].strip()
                    await self.send_streamed_response(ctx, text)
        except Exception as e:
            await ctx.send(f"❌ Vision failed: {e}")

    async def _contextual_draw(self, ctx: commands.Context, user_text: str):
        ch = ctx.channel.id
        ctxobj = self._ctx(ch)
        last_subject = ctxobj.get("last_subject") or ""

        explicit = self.maybe_extract_draw_prompt(user_text)
        raw_user_prompt = explicit or self._extract_draw_text(user_text)

        loki_bullets = ""
        if "history" in user_text.lower() or "from loki" in user_text.lower():
            snippets = await self._loki_search_snippets(ctx, raw_user_prompt or last_subject or "topic", since="365d", k=20)
            loki_bullets = await self._summarize_snippets_into_facts(ctx.author.id, snippets, goal="Provide visual details to include in an illustration prompt.")

        context_lines = []
        if last_subject:
            context_lines.append(f"Conversation subject: {last_subject}")
        if raw_user_prompt:
            context_lines.append(f"User wants: {raw_user_prompt}")
        if loki_bullets:
            context_lines.append(f"Relevant history facts:\n{loki_bullets}")

        crafter_sys = {
            "role": "system",
            "content": (
                "You write exactly ONE single-line image prompt for a text-to-image model. "
                "Use the given conversation subject and facts. Be concrete and vivid. "
                "No quotes, no commentary. 30 words max."
            )
        }
        crafter_user = {"role": "user", "content": "\n".join(context_lines) or "Create an evocative image prompt from recent conversation."}

        try:
            crafted = await self._with_limit(self._api_sem, self.query_deepseek(ctx.author.id, [crafter_sys, crafter_user]))
        except Exception:
            base = raw_user_prompt or last_subject or "the current topic"
            crafted = f"{base} in a detailed illustrative scene"

        styled = self.style_prompt(re.sub(r"\s+", " ", crafted).strip())
        ctxobj["last_image_prompt"] = styled
        return await self._render_and_send_image(ctx, styled, prefix="Let me sketch this")

    async def _render_and_send_image(self, ctx: commands.Context, styled_prompt: str, prefix: Optional[str] = None):
        # 1) start render ASAP (in parallel)
        render_task = asyncio.create_task(
            self._with_limit(self._api_sem, self.generate_openai_image_png(styled_prompt, size="1024x1024"))
        )

        # 2) artist narration via DeepSeek
        artist_system = {
            "role": "system",
            "content": (
                "You are April, a friendly concept artist narrating in real time as you start sketching. "
                "Keep it brief (2–4 short lines). Present tense. No asterisks. No markdown. "
                "Mention 1–2 vivid details from the prompt and a quick technique note (composition, brush, palette). "
                "End with: '— ok, exporting the piece.'"
            ),
        }
        artist_user = {
            "role": "user",
            "content": (
                f"I'm about to draw this:\n{styled_prompt}\n"
                "Narrate your first moments laying it out and the feel you're going for."
            ),
        }

        artist_text = None
        try:
            artist_text = await self._with_limit(self._api_sem, self.query_deepseek(ctx.author.id, [artist_system, artist_user]))
            artist_text = artist_text.strip()
            if len(artist_text) > 600:
                artist_text = artist_text[:600].rsplit(" ", 1)[0] + "…"
        except Exception:
            artist_text = f"Ok — I’ll sketch this: **{styled_prompt}**\nLaying in the shapes and light… — ok, exporting the piece."

        try:
            await self.send_streamed_response(ctx, artist_text)
        except Exception:
            await ctx.send(artist_text)

        # 3) when render completes, send the image (or error)
        try:
            png = await render_task
            await ctx.send(
                content="Here you go 🎨",
                file=discord.File(BytesIO(png), filename="april_draw.png")
            )
        except Exception as e:
            await ctx.send(f"⚠️ Image failed: `{e}`")

    # ---------------- Config (.aprilcfg) ----------------

    @commands.group(name="aprilconfig", aliases=["aprilcfg"])
    @commands.is_owner()
    async def aprilconfig(self, ctx: commands.Context):
        if ctx.invoked_subcommand is None:
            await self.show_settings(ctx)

    @aprilconfig.command()
    async def deepseekkey(self, ctx: commands.Context, key: str):
        await self.config.deepseek_key.set(key)
        await ctx.send("✅ DeepSeek key set.")
        try: await ctx.message.delete()
        except: pass

    @aprilconfig.command()
    async def anthropickey(self, ctx: commands.Context, key: str):
        await self.config.anthropic_key.set(key)
        await ctx.send("✅ Anthropic key set.")
        try: await ctx.message.delete()
        except: pass

    @aprilconfig.command()
    async def elevenlabs(self, ctx: commands.Context, key: str):
        await self.config.tts_key.set(key)
        await ctx.send("✅ ElevenLabs key set.")
        try: await ctx.message.delete()
        except: pass

    @aprilconfig.command()
    async def openaikey(self, ctx: commands.Context, key: str):
        await self.config.openai_key.set(key)
        await ctx.send("✅ OpenAI key set.")
        try: await ctx.message.delete()
        except: pass

    @aprilconfig.command()
    async def voice(self, ctx: commands.Context, voice_id: str):
        await self.config.voice_id.set(voice_id)
        await ctx.send(f"✅ Voice ID set to `{voice_id}`")

    @aprilconfig.command()
    async def model(self, ctx: commands.Context, model_name: str):
        await self.config.model.set(model_name.lower())
        await ctx.send(f"✅ Model set to `{model_name}`")

    @aprilconfig.command()
    async def temperature(self, ctx: commands.Context, value: float):
        if 0.0 <= value <= 1.0:
            await self.config.temperature.set(value)
            await ctx.send(f"✅ Temperature set to `{value}`")
        else:
            await ctx.send("❌ Value must be between 0.0 and 1.0")

    @aprilconfig.command()
    async def tokens(self, ctx: commands.Context, num: int):
        if 100 <= num <= 4096:
            await self.config.max_tokens.set(num)
            await ctx.send(f"✅ Max tokens set to `{num}`")
        else:
            await ctx.send("❌ Value must be between 100 and 4096")

    @aprilconfig.command()
    async def tts(self, ctx: commands.Context, enabled: bool):
        await self.config.tts_enabled.set(enabled)
        await ctx.send(f"✅ TTS {'enabled' if enabled else 'disabled'}")

    @aprilconfig.command()
    async def textresponse(self, ctx: commands.Context, enabled: bool):
        await self.config.text_response_when_voice.set(enabled)
        await ctx.send(f"✅ Text responses will be {'shown' if enabled else 'hidden'} when using voice")

    @aprilconfig.command()
    async def maxhistory(self, ctx: commands.Context, num: int):
        if 1 <= num <= 20:
            await self.config.max_history.set(num)
            for ch in self.history:
                self.history[ch] = deque(self.history[ch], maxlen=num * 2)
            await ctx.send(f"✅ Max history set to `{num}` exchanges")
        else:
            await ctx.send("❌ Value must be between 1 and 20")

    @aprilconfig.command()
    async def gifs(self, ctx: commands.Context, enabled: bool):
        await self.config.use_gifs.set(enabled)
        await ctx.send(f"✅ Emotion GIFs {'enabled' if enabled else 'disabled'}")

    @aprilconfig.command()
    async def gifprob(self, ctx: commands.Context, value: float):
        if 0.0 <= value <= 1.0:
            await self.config.gif_prob.set(value)
            await ctx.send(f"✅ GIF probability set to `{value}`")
        else:
            await ctx.send("❌ Value must be between 0 and 1")

    @aprilconfig.command(name="tenorkey")
    async def cfg_tenorkey(self, ctx: commands.Context, key: str):
        await self.config.tenor_key.set(key)
        await ctx.send("✅ Tenor key set.")
        try: await ctx.message.delete()
        except: pass

    @aprilconfig.command(name="tenor")
    async def cfg_tenor(self, ctx: commands.Context, enabled: bool):
        await self.config.tenor_enabled.set(bool(enabled))
        await ctx.send(f"✅ Tenor {'enabled' if enabled else 'disabled'}")

    @aprilconfig.command(name="vision")
    async def cfg_vision(self, ctx: commands.Context, provider: str):
        provider = provider.lower().strip()
        if provider not in ("openai",):
            return await ctx.send("❌ Supported providers: openai")
        await self.config.vision_provider.set(provider)
        await ctx.send(f"✅ Vision provider set to `{provider}`")

    @aprilconfig.command(name="lokiurl")
    async def cfg_lokiurl(self, ctx: commands.Context, url: str):
        await self.config.loki_url.set(url.rstrip("/"))
        await ctx.send(f"✅ Loki URL set to `{url}`")

    @aprilconfig.command(name="lokitoken")
    async def cfg_lokitoken(self, ctx: commands.Context, token: str):
        await self.config.loki_token.set(token)
        await ctx.send("✅ Loki token set")

    @aprilconfig.command(name="lokipush")
    async def cfg_lokipush(self, ctx: commands.Context, url: str):
        await self.config.loki_push_url.set(url)
        await ctx.send(f"✅ Loki push URL set to `{url}`")

    @aprilconfig.command(name="lokipushon")
    async def cfg_lokipushon(self, ctx: commands.Context, enabled: bool):
        await self.config.loki_push_enabled.set(bool(enabled))
        await ctx.send(f"✅ Loki live logging {'ENABLED' if enabled else 'DISABLED'}")

    @aprilconfig.command(name="lokiverify")
    async def cfg_lokiverify(self, ctx: commands.Context):
        await self.lokiverify_cmd(ctx)

    @aprilconfig.command(name="localtracks")
    @commands.is_owner()
    async def cfg_localtracks(self, ctx: commands.Context, root: str, subdir: str = "april"):
        root = root.rstrip("/")
        await self.config.localtracks_root.set(root)
        await self.config.localtracks_subdir.set(subdir)
        await ctx.send(f"✅ Localtracks root set to `{root}`, subdir `{subdir}`.\n"
                       f"I will save TTS into: `{root}/{subdir}` and play with: `localtracks/{subdir}/<file.mp3>`")

    @aprilconfig.command(name="ttsdebug")
    @commands.is_owner()
    async def cfg_ttsdebug(self, ctx: commands.Context, *, text: str = "This is a TTS test from April."):
        tts_key = await self.config.tts_key()
        if not tts_key:
            return await ctx.send("❌ No ElevenLabs key set. Use `[p]aprilcfg elevenlabs <key>`")

        root = (await self.config.localtracks_root()).strip()
        subdir = (await self.config.localtracks_subdir()).strip() or "april"

        if not root:
            return await ctx.send("❌ `localtracks_root` not set. Configure it with `[p]aprilcfg localtracks <abs_path> [subdir]`.\n"
                                  "It must match your `.audioset localpath` value.")

        outdir = Path(root) / subdir
        outdir.mkdir(parents=True, exist_ok=True)

        try:
            audio = await self.generate_tts_audio(text, tts_key)
            if not audio:
                return await ctx.send("❌ TTS generation failed.")
            fname = f"ttsdebug_{int(time.time())}_{random.randint(1000,9999)}.mp3"
            fpath = outdir / fname
            with open(fpath, "wb") as f:
                f.write(audio)
        except Exception as e:
            return await ctx.send(f"❌ Error writing TTS file: `{e}`")

        query = f"localtracks/{subdir}/{fname}"
        audio_cog = self.bot.get_cog("Audio")
        if not audio_cog:
            return await ctx.send(f"✅ File saved at `{fpath}` but Audio cog not loaded. Query would be: `{query}`")

        try:
            if hasattr(audio_cog, "command_play"):
                await audio_cog.command_play(ctx, query=query)
            else:
                await ctx.invoke(audio_cog.play, query=query)
            await ctx.send(f"▶️ Playing `{query}`")
        except Exception as e:
            await ctx.send(f"❌ Couldn’t play `{query}`: `{e}`\n"
                           f"Confirm `.audioset localpath` == `{root}` and file exists: `{fpath}`")

    @aprilconfig.command(name="settings")
    async def show_settings(self, ctx: commands.Context):
        cfg = await self.config.all()
        e = discord.Embed(title="AprilTalk Configuration")
        def tail(v): return f"`...{v[-4:]}`" if v else "❌ Not set"
        e.add_field(name="DeepSeek Key", value=tail(cfg.get("deepseek_key")), inline=True)
        e.add_field(name="Anthropic Key", value=tail(cfg.get("anthropic_key")), inline=True)
        e.add_field(name="OpenAI Key", value=tail(cfg.get("openai_key")), inline=True)
        e.add_field(name="ElevenLabs Key", value=tail(cfg.get("tts_key")), inline=True)
        e.add_field(name="Tenor Key", value=tail(cfg.get("tenor_key")), inline=True)
        e.add_field(name="Tenor Enabled", value="✅" if cfg.get("tenor_enabled") else "❌", inline=True)
        e.add_field(name="GIF Prob", value=f"{cfg.get('gif_prob')}", inline=True)
        e.add_field(name="Vision Provider", value=f"{cfg.get('vision_provider')}", inline=True)
        e.add_field(name="Voice ID", value=f"`{cfg['voice_id']}`", inline=True)
        e.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        e.add_field(name="Temperature", value=f"`{cfg['temperature']}`", inline=True)
        e.add_field(name="Max Tokens", value=f"`{cfg['max_tokens']}`", inline=True)
        e.add_field(name="Max History", value=f"`{cfg['max_history']} exchanges`", inline=True)
        e.add_field(name="Max Message Length", value=f"`{cfg['max_message_length']}`", inline=True)
        e.add_field(name="TTS Enabled", value="✅" if cfg['tts_enabled'] else "❌", inline=True)
        e.add_field(name="Text with Voice", value="✅" if cfg['text_response_when_voice'] else "❌", inline=True)
        e.add_field(name="Loki URL", value=f"{cfg.get('loki_url','')}", inline=False)
        e.add_field(name="Loki Push URL", value=f"{cfg.get('loki_push_url','')}", inline=False)
        e.add_field(name="Loki Push Enabled", value="✅" if cfg.get("loki_push_enabled") else "❌", inline=True)
        e.add_field(name="Sleep Mode", value="😴 ON" if cfg.get("sleep_enabled") else "💬 OFF", inline=True)
        e.add_field(name="Sleep Allowed User", value=f"<@{cfg.get('sleep_user_id','')}>" if cfg.get("sleep_user_id") else "—", inline=True)
        e.add_field(name="Localtracks Root", value=f"{cfg.get('localtracks_root') or '—'}", inline=False)
        e.add_field(name="Localtracks Subdir", value=f"{cfg.get('localtracks_subdir') or 'april'}", inline=True)
        sp = cfg['system_prompt'][:200] + ("..." if len(cfg['system_prompt']) > 200 else "")
        sm = cfg['smert_prompt'][:200] + ("..." if len(cfg['smert_prompt']) > 200 else "")
        e.add_field(name="System Prompt", value=f"```{sp}```", inline=False)
        e.add_field(name="Smert Prompt", value=f"```{sm}```", inline=False)
        await ctx.send(embed=e)

    # ---------------- Voice events ----------------

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        if member.bot:
            return
        try:
            player = self.get_player(member.guild.id)
            if player and self.is_player_connected(player):
                channel_id = self.get_player_channel_id(player)
                if channel_id:
                    voice_channel = self.bot.get_channel(channel_id)
                    if voice_channel:
                        human_members = [m for m in voice_channel.members if not m.bot]
                        if len(human_members) == 0:
                            await player.disconnect()
                            log.debug(f"Left voice in {member.guild} (empty channel)")
        except Exception as e:
            log.error(f"Error handling voice state update: {e}")


async def setup(bot: Red):
    await bot.add_cog(AprilTalk(bot))
