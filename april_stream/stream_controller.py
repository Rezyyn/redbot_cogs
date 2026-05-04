"""
stream_controller.py
Red-Bot cog — sends commands to stream_daemon.py via local TCP socket.
The daemon holds the Discord user token and does the actual streaming.

Commands:
  !stream join [channel]         — daemon joins a voice channel
  !stream vlc <title|path>       — open VLC fullscreen + stream audio
  !stream firefox <url>          — open Firefox kiosk + stream audio
  !stream stop                   — stop stream and disconnect
  !stream status                 — show active stream info
  !stream config dbpath <path>   — set movie DB JSON path (per guild)
  !stream config daemon <h> <p>  — set daemon host:port (owner only)

  !movies                        — paginated movie list
  !movies search <query>         — fuzzy search
  !movies add <path> [title]     — add single entry
  !movies remove <title>         — remove by title
  !movies scan <directory>       — recursive scan, auto-populate DB
  !movies info <title>           — show path / metadata
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import discord
from redbot.core import Config, checks, commands
from redbot.core.utils.chat_formatting import box, pagify
from redbot.core.utils.menus import DEFAULT_CONTROLS, menu


# ─────────────────────────────────────────────────────────────
#  Cog
# ─────────────────────────────────────────────────────────────

class StreamController(commands.Cog):
    """Stream movies and web content to Discord voice channels."""

    VIDEO_EXTENSIONS = {
        ".mkv", ".mp4", ".avi", ".mov", ".wmv",
        ".m4v", ".flv", ".webm", ".ts", ".m2ts",
    }

    def __init__(self, bot):
        self.bot = bot

        self.config = Config.get_conf(
            self, identifier=994455667788, force_registration=True
        )

        # Global (owner-only) — daemon network location
        self.config.register_global(
            daemon_host="127.0.0.1",
            daemon_port=7734,
        )

        # Per-guild settings
        self.config.register_guild(
            movie_db_path=str(Path.home() / "movies.json"),
        )

    # ─────────────────────────── IPC ────────────────────────────

    async def _send_daemon(self, payload: dict) -> dict:
        """Send one JSON command to the daemon, return its response."""
        host = await self.config.daemon_host()
        port = await self.config.daemon_port()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=5
            )
            writer.write(json.dumps(payload).encode() + b"\n")
            await writer.drain()
            raw = await asyncio.wait_for(reader.readline(), timeout=10)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return json.loads(raw.decode())
        except asyncio.TimeoutError:
            return {"ok": False, "error": "Daemon timed out"}
        except (ConnectionRefusedError, OSError):
            return {"ok": False, "error": "Daemon not running — start stream_daemon.py"}

    # ──────────────────────── Movie DB ──────────────────────────

    @staticmethod
    def _load_db(path: str) -> list:
        p = Path(path)
        if not p.exists():
            return []
        try:
            with open(p) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    @staticmethod
    def _save_db(path: str, db: list):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(db, f, indent=2)

    @staticmethod
    def _find(db: list, query: str) -> Optional[dict]:
        q = query.lower()
        for m in db:
            if m["title"].lower() == q:
                return m
        for m in db:
            if q in m["title"].lower():
                return m
        return None

    def _scan(self, directory: str) -> list:
        results = []
        for p in Path(directory).rglob("*"):
            if p.suffix.lower() in self.VIDEO_EXTENSIONS:
                results.append({
                    "title": p.stem,
                    "path": str(p),
                    "year": None,
                    "genre": None,
                })
        return results

    # ──────────────────────── !stream ───────────────────────────

    @commands.group(name="stream", invoke_without_command=True)
    @checks.mod_or_permissions(manage_guild=True)
    async def stream(self, ctx):
        """Stream controller — see subcommands."""
        await ctx.send_help()

    @stream.command(name="join")
    async def stream_join(self, ctx, *, channel: discord.VoiceChannel = None):
        """Join a voice channel. Defaults to your current channel."""
        if channel is None:
            if ctx.author.voice:
                channel = ctx.author.voice.channel
            else:
                return await ctx.send("❌ Join a voice channel or specify one.")

        resp = await self._send_daemon({
            "action": "join",
            "guild_id": ctx.guild.id,
            "channel_id": channel.id,
            "channel_name": channel.name,
        })

        if resp.get("ok"):
            await ctx.send(f"✅ Joined **{channel.name}**")
        else:
            await ctx.send(f"❌ {resp.get('error')}")

    @stream.command(name="vlc")
    async def stream_vlc(self, ctx, *, query: str):
        """
        Stream via VLC.

        Accepts a movie title from the DB or an absolute file path.
        Example: !stream vlc Inception
                 !stream vlc /mnt/movies/inception.mkv
        """
        db_path = await self.config.guild(ctx.guild).movie_db_path()
        db = self._load_db(db_path)

        movie = self._find(db, query)
        if movie:
            path, title = movie["path"], movie["title"]
        elif Path(query).exists():
            path, title = query, Path(query).stem
        else:
            return await ctx.send(
                f"❌ `{query}` not found in movie DB or filesystem.\n"
                f"Use `!movies search <query>` to check available titles."
            )

        resp = await self._send_daemon({
            "action": "stream_vlc",
            "guild_id": ctx.guild.id,
            "path": path,
            "title": title,
        })

        if resp.get("ok"):
            await ctx.send(f"▶️ Streaming **{title}** via VLC")
        else:
            await ctx.send(f"❌ {resp.get('error')}")

    @stream.command(name="firefox")
    async def stream_firefox(self, ctx, url: str):
        """
        Open Firefox in kiosk mode and stream it.

        Example: !stream firefox https://twitch.tv/someone
                 !stream firefox youtube.com/watch?v=xxx
        """
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        resp = await self._send_daemon({
            "action": "stream_firefox",
            "guild_id": ctx.guild.id,
            "url": url,
        })

        if resp.get("ok"):
            await ctx.send(f"🌐 Streaming Firefox → `{url}`")
        else:
            await ctx.send(f"❌ {resp.get('error')}")

    @stream.command(name="stop")
    async def stream_stop(self, ctx):
        """Stop the current stream and disconnect from voice."""
        resp = await self._send_daemon({
            "action": "stop",
            "guild_id": ctx.guild.id,
        })
        if resp.get("ok"):
            await ctx.send("⏹️ Stream stopped.")
        else:
            await ctx.send(f"❌ {resp.get('error')}")

    @stream.command(name="status")
    async def stream_status(self, ctx):
        """Show current stream status."""
        resp = await self._send_daemon({"action": "status"})

        if not resp.get("ok"):
            return await ctx.send("❌ Daemon offline or not responding.")

        guild_status = resp.get("status", {}).get(str(ctx.guild.id))
        if not guild_status:
            return await ctx.send("💤 No active stream for this server.")

        e = discord.Embed(title="📡 Stream Status", color=discord.Color.green())
        e.add_field(name="Mode",    value=guild_status.get("mode", "?"),    inline=True)
        e.add_field(name="Content", value=guild_status.get("title", "?"),   inline=True)
        e.add_field(name="Channel", value=guild_status.get("channel", "?"), inline=True)
        e.add_field(name="Audio",   value="🔊 Playing" if guild_status.get("playing") else "🔇 Silent", inline=True)
        await ctx.send(embed=e)

    # ─────────────────── !stream config ─────────────────────────

    @stream.group(name="config")
    async def stream_config(self, ctx):
        """Configure stream settings."""
        if ctx.invoked_subcommand is None:
            await ctx.send_help()

    @stream_config.command(name="dbpath")
    @checks.admin()
    async def config_dbpath(self, ctx, path: str):
        """Set movie DB JSON path for this server."""
        await self.config.guild(ctx.guild).movie_db_path.set(path)
        await ctx.send(f"✅ Movie DB path → `{path}`")

    @stream_config.command(name="daemon")
    @checks.is_owner()
    async def config_daemon(self, ctx, host: str, port: int):
        """(Owner) Set daemon host and port."""
        await self.config.daemon_host.set(host)
        await self.config.daemon_port.set(port)
        await ctx.send(f"✅ Daemon → `{host}:{port}`")

    @stream_config.command(name="show")
    @checks.admin()
    async def config_show(self, ctx):
        """Show current config."""
        host = await self.config.daemon_host()
        port = await self.config.daemon_port()
        db_path = await self.config.guild(ctx.guild).movie_db_path()

        e = discord.Embed(title="⚙️ Stream Config", color=discord.Color.blurple())
        e.add_field(name="Daemon",   value=f"`{host}:{port}`", inline=False)
        e.add_field(name="Movie DB", value=f"`{db_path}`",     inline=False)
        await ctx.send(embed=e)

    # ──────────────────────── !movies ───────────────────────────

    @commands.group(name="movies", invoke_without_command=True)
    async def movies(self, ctx):
        """Movie database — lists all titles. See subcommands for more."""
        await self.movies_list(ctx)

    @movies.command(name="list")
    async def movies_list(self, ctx):
        """Paginated list of all movies."""
        db_path = await self.config.guild(ctx.guild).movie_db_path()
        db = self._load_db(db_path)

        if not db:
            return await ctx.send(
                "📭 Database is empty.\n"
                "Use `!movies scan <dir>` to auto-populate or `!movies add <path>` to add manually."
            )

        db_sorted = sorted(db, key=lambda m: m["title"].lower())
        lines = []
        for i, m in enumerate(db_sorted, 1):
            year  = f" ({m['year']})"  if m.get("year")  else ""
            genre = f" [{m['genre']}]" if m.get("genre") else ""
            lines.append(f"`{i:>3}.` **{m['title']}**{year}{genre}")

        pages = list(pagify("\n".join(lines), page_length=1800))
        embeds = []
        for idx, chunk in enumerate(pages):
            e = discord.Embed(
                title=f"🎬 Movie Database — {len(db)} titles",
                description=chunk,
                color=discord.Color.blue(),
            )
            e.set_footer(text=f"Page {idx + 1}/{len(pages)} • !movies search <query> to filter")
            embeds.append(e)

        if len(embeds) == 1:
            await ctx.send(embed=embeds[0])
        else:
            await menu(ctx, embeds, DEFAULT_CONTROLS)

    @movies.command(name="search")
    async def movies_search(self, ctx, *, query: str):
        """Search movie titles (case-insensitive substring)."""
        db_path = await self.config.guild(ctx.guild).movie_db_path()
        db = self._load_db(db_path)

        q = query.lower()
        results = [m for m in db if q in m["title"].lower()]

        if not results:
            return await ctx.send(f"🔍 No results for `{query}`")

        desc = []
        for m in results[:25]:
            year = f" ({m['year']})" if m.get("year") else ""
            desc.append(f"• **{m['title']}**{year} — `{m['path']}`")
        if len(results) > 25:
            desc.append(f"*…and {len(results) - 25} more*")

        e = discord.Embed(
            title=f"🔍 '{query}' — {len(results)} result(s)",
            description="\n".join(desc),
            color=discord.Color.blue(),
        )
        await ctx.send(embed=e)

    @movies.command(name="add")
    @checks.mod_or_permissions(manage_guild=True)
    async def movies_add(self, ctx, path: str, *, title: str = None):
        """
        Add a movie to the DB.

        Usage: !movies add /path/to/file.mkv [optional title override]
        """
        if title is None:
            title = Path(path).stem

        db_path = await self.config.guild(ctx.guild).movie_db_path()
        db = self._load_db(db_path)

        if any(m["path"] == path for m in db):
            return await ctx.send(f"⚠️ Already in DB: **{title}**")

        db.append({"title": title, "path": path, "year": None, "genre": None})
        self._save_db(db_path, db)
        await ctx.send(f"✅ Added **{title}**")

    @movies.command(name="remove")
    @checks.mod_or_permissions(manage_guild=True)
    async def movies_remove(self, ctx, *, title: str):
        """Remove a movie by exact title."""
        db_path = await self.config.guild(ctx.guild).movie_db_path()
        db = self._load_db(db_path)

        before = len(db)
        db = [m for m in db if m["title"].lower() != title.lower()]
        if len(db) == before:
            return await ctx.send(f"❌ Not found: **{title}**")

        self._save_db(db_path, db)
        await ctx.send(f"🗑️ Removed **{title}**")

    @movies.command(name="scan")
    @checks.mod_or_permissions(manage_guild=True)
    async def movies_scan(self, ctx, directory: str):
        """
        Recursively scan a directory and add all video files.

        Skips paths already in the DB.
        Example: !movies scan /mnt/nas/movies
                 !movies scan "D:\\Movies"
        """
        if not Path(directory).is_dir():
            return await ctx.send(f"❌ Not a directory: `{directory}`")

        async with ctx.typing():
            found = await asyncio.get_event_loop().run_in_executor(
                None, self._scan, directory
            )

        if not found:
            return await ctx.send("📭 No video files found.")

        db_path = await self.config.guild(ctx.guild).movie_db_path()
        db = self._load_db(db_path)

        existing = {m["path"] for m in db}
        new = [m for m in found if m["path"] not in existing]

        db.extend(new)
        self._save_db(db_path, db)

        await ctx.send(
            f"✅ Scanned `{directory}` — "
            f"**{len(new)} new** added, {len(found) - len(new)} already in DB. "
            f"Total: {len(db)}"
        )

    @movies.command(name="info")
    async def movies_info(self, ctx, *, title: str):
        """Show details for a specific movie."""
        db_path = await self.config.guild(ctx.guild).movie_db_path()
        db = self._load_db(db_path)
        movie = self._find(db, title)

        if not movie:
            return await ctx.send(f"❌ Not found: **{title}**")

        e = discord.Embed(title=f"🎬 {movie['title']}", color=discord.Color.blue())
        e.add_field(name="Path",  value=f"`{movie['path']}`", inline=False)
        if movie.get("year"):
            e.add_field(name="Year",  value=str(movie["year"]),  inline=True)
        if movie.get("genre"):
            e.add_field(name="Genre", value=movie["genre"],       inline=True)
        await ctx.send(embed=e)
