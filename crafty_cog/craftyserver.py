"""
CraftyServer - Red-Bot Cog
Creates Minecraft servers in Crafty Controller from a CurseForge modpack link.

Flow:
  1. User runs [p]createserver <curseforge_url> [name] [port] [min_ram] [max_ram]
  2. Bot resolves the CF slug -> mod ID -> server pack download URL via CF API
  3. Downloads the server pack zip to Crafty's import directory
  4. Calls Crafty API to create the server from the zip
  5. Reports back with server ID and connection info

Config keys (set with [p]craftyset):
  crafty_url       - e.g. https://localhost:8443
  crafty_token     - Crafty superuser API token
  crafty_import    - Filesystem path to Crafty's import dir (e.g. /opt/crafty/import)
  cf_api_key       - CurseForge API key (get one at console.curseforge.com)
  default_min_ram  - Default min RAM in GB (default: 2)
  default_max_ram  - Default max RAM in GB (default: 6)
  default_port     - Default starting port (auto-incremented if taken, default: 25565)
  allow_roles      - List of role IDs allowed to use createserver (empty = everyone)
"""

import asyncio
import os
import re
import tempfile
import aiohttp
import aiofiles

from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import box, humanize_list
import discord

CURSEFORGE_API = "https://api.curseforge.com/v1"
CF_GAME_ID = 432  # Minecraft

# Regex patterns for CurseForge URLs
RE_CF_MODPACK = re.compile(
    r"(?:https?://)?(?:www\.)?curseforge\.com/minecraft/modpacks/([a-z0-9\-]+)(?:/files/(\d+))?",
    re.IGNORECASE,
)
RE_CF_DIRECT_FILE = re.compile(
    r"(?:https?://)?(?:www\.)?curseforge\.com/minecraft/modpacks/[a-z0-9\-]+/files/(\d+)",
    re.IGNORECASE,
)


class CraftyServer(commands.Cog):
    """Create Minecraft servers in Crafty Controller from a CurseForge modpack link."""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=0x637261667479, force_registration=True)
        self.config.register_global(
            crafty_url="",
            crafty_token="",
            crafty_import="",
            cf_api_key="",
            default_min_ram=2,
            default_max_ram=6,
            default_port=25565,
            allow_roles=[],
        )

    # -------------------------------------------------------------------------
    # Permission check
    # -------------------------------------------------------------------------

    async def _can_use(self, ctx: commands.Context) -> bool:
        """Returns True if the user is allowed to create servers."""
        if await self.bot.is_owner(ctx.author):
            return True
        allow_roles = await self.config.allow_roles()
        if not allow_roles:
            return True  # open to everyone
        return any(r.id in allow_roles for r in ctx.author.roles)

    # -------------------------------------------------------------------------
    # Config commands
    # -------------------------------------------------------------------------

    @commands.group(name="craftyset", aliases=["cset"])
    @commands.is_owner()
    async def craftyset(self, ctx: commands.Context):
        """Configure the CraftyServer cog."""

    @craftyset.command(name="url")
    async def set_url(self, ctx, url: str):
        """Set the Crafty Controller base URL (e.g. https://localhost:8443)."""
        await self.config.crafty_url.set(url.rstrip("/"))
        await ctx.tick()

    @craftyset.command(name="token")
    async def set_token(self, ctx, token: str):
        """Set the Crafty superuser API token."""
        await self.config.crafty_token.set(token)
        await ctx.message.delete()
        await ctx.send("Token saved. (Message deleted for security.)", delete_after=5)

    @craftyset.command(name="import")
    async def set_import(self, ctx, path: str):
        """Set the filesystem path to Crafty's import directory."""
        if not os.path.isdir(path):
            return await ctx.send(f"Path `{path}` does not exist or is not a directory.")
        await self.config.crafty_import.set(path)
        await ctx.tick()

    @craftyset.command(name="cfkey")
    async def set_cfkey(self, ctx, key: str):
        """Set your CurseForge API key."""
        await self.config.cf_api_key.set(key)
        await ctx.message.delete()
        await ctx.send("CF API key saved. (Message deleted for security.)", delete_after=5)

    @craftyset.command(name="ram")
    async def set_ram(self, ctx, min_gb: int, max_gb: int):
        """Set default RAM allocation in GB. e.g. `ram 2 8`"""
        await self.config.default_min_ram.set(min_gb)
        await self.config.default_max_ram.set(max_gb)
        await ctx.tick()

    @craftyset.command(name="port")
    async def set_port(self, ctx, port: int):
        """Set the default starting port for new servers."""
        await self.config.default_port.set(port)
        await ctx.tick()

    @craftyset.command(name="roles")
    async def set_roles(self, ctx, *roles: discord.Role):
        """Set which roles can create servers. Leave empty to allow everyone."""
        await self.config.allow_roles.set([r.id for r in roles])
        if roles:
            await ctx.send(f"Allowed roles: {humanize_list([r.mention for r in roles])}")
        else:
            await ctx.send("All users can now create servers.")

    @craftyset.command(name="show")
    async def show_config(self, ctx):
        """Show current configuration (tokens redacted)."""
        cfg = await self.config.all()
        allow_roles = cfg["allow_roles"]
        role_names = (
            humanize_list([f"<@&{r}>" for r in allow_roles]) if allow_roles else "Everyone"
        )
        msg = (
            f"**Crafty URL:** {cfg['crafty_url'] or 'not set'}\n"
            f"**Crafty Token:** {'set' if cfg['crafty_token'] else 'not set'}\n"
            f"**Import Path:** {cfg['crafty_import'] or 'not set'}\n"
            f"**CF API Key:** {'set' if cfg['cf_api_key'] else 'not set'}\n"
            f"**Default RAM:** {cfg['default_min_ram']}G – {cfg['default_max_ram']}G\n"
            f"**Default Port:** {cfg['default_port']}\n"
            f"**Allowed Roles:** {role_names}"
        )
        await ctx.send(msg)

    # -------------------------------------------------------------------------
    # Main command
    # -------------------------------------------------------------------------

    @commands.command(name="createserver", aliases=["mkserver"])
    @commands.guild_only()
    async def create_server(
        self,
        ctx: commands.Context,
        cf_url: str,
        name: str = None,
        port: int = None,
        min_ram: int = None,
        max_ram: int = None,
    ):
        """
        Create a Minecraft server from a CurseForge modpack link.

        Usage:
          [p]createserver <curseforge_url> [name] [port] [min_ram_gb] [max_ram_gb]

        Examples:
          [p]createserver https://www.curseforge.com/minecraft/modpacks/all-the-mods-9
          [p]createserver https://www.curseforge.com/minecraft/modpacks/atm9 "ATM9 Server" 25570 2 8
        """
        if not await self._can_use(ctx):
            return await ctx.send("You don't have permission to create servers.")

        cfg = await self.config.all()
        if not all([cfg["crafty_url"], cfg["crafty_token"], cfg["crafty_import"], cfg["cf_api_key"]]):
            return await ctx.send(
                "CraftyServer isn't fully configured yet. Ask the bot owner to run `craftyset`."
            )

        port = port or cfg["default_port"]
        min_ram = min_ram or cfg["default_min_ram"]
        max_ram = max_ram or cfg["default_max_ram"]

        # -- Parse CurseForge URL --
        match = RE_CF_MODPACK.match(cf_url)
        if not match:
            return await ctx.send("Couldn't parse that URL. Expected a CurseForge modpack link.")

        slug = match.group(1)
        specific_file_id = match.group(2)  # May be None

        msg = await ctx.send(
            embed=self._embed("🔍 Resolving modpack...", f"Slug: `{slug}`", discord.Color.blurple())
        )

        async with aiohttp.ClientSession() as session:
            # 1. Resolve slug -> mod ID
            try:
                mod_id, mod_name, mod_version = await self._resolve_mod(
                    session, cfg["cf_api_key"], slug
                )
            except Exception as e:
                return await msg.edit(embed=self._embed("❌ CF Error", str(e), discord.Color.red()))

            server_name = name or f"{mod_name} ({mod_version})"

            await msg.edit(
                embed=self._embed(
                    "📦 Found modpack",
                    f"**{mod_name}** `{mod_version}` (ID: {mod_id})\nLooking for server pack...",
                    discord.Color.blurple(),
                )
            )

            # 2. Get server pack download URL
            try:
                download_url, file_name = await self._get_server_pack(
                    session, cfg["cf_api_key"], mod_id, specific_file_id
                )
            except Exception as e:
                return await msg.edit(embed=self._embed("❌ No Server Pack", str(e), discord.Color.red()))

            await msg.edit(
                embed=self._embed(
                    "⬇️ Downloading server pack...",
                    f"`{file_name}`\nThis may take a while for large packs.",
                    discord.Color.blurple(),
                )
            )

            # 3. Download zip to Crafty's import directory
            import_path = cfg["crafty_import"]
            dest_path = os.path.join(import_path, file_name)

            try:
                await self._download_file(session, download_url, dest_path, msg)
            except Exception as e:
                return await msg.edit(
                    embed=self._embed("❌ Download Failed", str(e), discord.Color.red())
                )

            await msg.edit(
                embed=self._embed(
                    "🔧 Creating server in Crafty...",
                    f"Server: **{server_name}**\nPort: `{port}` | RAM: `{min_ram}G – {max_ram}G`",
                    discord.Color.blurple(),
                )
            )

            # 4. Create server in Crafty via API
            try:
                server_id, server_uuid = await self._crafty_create_server(
                    session,
                    cfg["crafty_url"],
                    cfg["crafty_token"],
                    server_name=server_name,
                    zip_filename=file_name,
                    port=port,
                    min_ram=min_ram,
                    max_ram=max_ram,
                )
            except Exception as e:
                return await msg.edit(
                    embed=self._embed("❌ Crafty API Error", str(e), discord.Color.red())
                )

        # 5. Done
        embed = discord.Embed(
            title="✅ Server Created",
            color=discord.Color.green(),
        )
        embed.add_field(name="Name", value=server_name, inline=False)
        embed.add_field(name="Server UUID", value=f"`{server_uuid}`", inline=False)
        embed.add_field(name="Port", value=str(port), inline=True)
        embed.add_field(name="RAM", value=f"{min_ram}G / {max_ram}G", inline=True)
        embed.add_field(
            name="Crafty Panel",
            value=f"{cfg['crafty_url']}/panel/server_detail?id={server_uuid}",
            inline=False,
        )
        embed.set_footer(text="Remember to agree to the EULA in Crafty before starting the server.")
        await msg.edit(embed=embed)

    # -------------------------------------------------------------------------
    # CurseForge helpers
    # -------------------------------------------------------------------------

    async def _resolve_mod(self, session: aiohttp.ClientSession, cf_key: str, slug: str):
        """Resolve a CF modpack slug to (mod_id, mod_name, latest_version)."""
        headers = {"x-api-key": cf_key}
        params = {
            "gameId": CF_GAME_ID,
            "classId": 4471,  # Modpacks class
            "slug": slug,
            "pageSize": 1,
        }
        async with session.get(f"{CURSEFORGE_API}/mods/search", headers=headers, params=params) as r:
            r.raise_for_status()
            data = await r.json()

        results = data.get("data", [])
        if not results:
            raise ValueError(f"No modpack found for slug `{slug}`. Check the URL.")

        mod = results[0]
        mod_id = mod["id"]
        mod_name = mod["name"]
        latest = mod.get("latestFilesIndexes", [{}])[0].get("gameVersion", "unknown")
        return mod_id, mod_name, latest

    async def _get_server_pack(
        self,
        session: aiohttp.ClientSession,
        cf_key: str,
        mod_id: int,
        specific_file_id: str = None,
    ):
        """
        Return (download_url, filename) for the server pack.
        Prefers explicit file ID if given, otherwise picks the newest server pack.
        """
        headers = {"x-api-key": cf_key}

        if specific_file_id:
            url = f"{CURSEFORGE_API}/mods/{mod_id}/files/{specific_file_id}"
            async with session.get(url, headers=headers) as r:
                r.raise_for_status()
                data = await r.json()
            file = data["data"]
            return await self._resolve_file_download(session, headers, file)

        # Fetch all files and find the latest server pack
        url = f"{CURSEFORGE_API}/mods/{mod_id}/files"
        params = {"pageSize": 50}
        async with session.get(url, headers=headers, params=params) as r:
            r.raise_for_status()
            data = await r.json()

        files = data.get("data", [])
        server_packs = [f for f in files if f.get("isServerPack")]

        if not server_packs:
            # Some packs don't flag isServerPack — look for filename hints
            server_packs = [
                f for f in files
                if "server" in f.get("fileName", "").lower()
                and f.get("fileName", "").endswith(".zip")
            ]

        if not server_packs:
            raise ValueError(
                "No server pack found for this modpack.\n"
                "Some modpacks don't include a server pack — you may need to build it manually."
            )

        # Sort by date, newest first
        server_packs.sort(key=lambda f: f.get("fileDate", ""), reverse=True)
        return await self._resolve_file_download(session, headers, server_packs[0])

    async def _resolve_file_download(self, session, headers, file: dict):
        """
        Returns (download_url, filename).
        CF sometimes returns None for downloadUrl — falls back to constructing it.
        """
        download_url = file.get("downloadUrl")
        file_name = file.get("fileName", "serverpack.zip")

        if not download_url:
            # Construct the CDN URL manually
            file_id = file["id"]
            id_str = str(file_id)
            part1 = id_str[:4]
            part2 = id_str[4:]
            download_url = (
                f"https://mediafilez.forgecdn.net/files/{part1}/{part2}/{file_name}"
            )

        return download_url, file_name

    # -------------------------------------------------------------------------
    # Download helper
    # -------------------------------------------------------------------------

    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        dest: str,
        status_msg: discord.Message = None,
    ):
        """Stream-download a file to dest, updating status_msg with progress."""
        async with session.get(url) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            last_update = 0

            async with aiofiles.open(dest, "wb") as f:
                async for chunk in r.content.iter_chunked(1024 * 512):  # 512KB chunks
                    await f.write(chunk)
                    downloaded += len(chunk)

                    # Update Discord message every ~5MB
                    if status_msg and total and (downloaded - last_update) >= 5 * 1024 * 1024:
                        last_update = downloaded
                        pct = int((downloaded / total) * 100)
                        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        try:
                            await status_msg.edit(
                                embed=self._embed(
                                    "⬇️ Downloading server pack...",
                                    f"`[{bar}]` {pct}% ({downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB)",
                                    discord.Color.blurple(),
                                )
                            )
                        except Exception:
                            pass

    # -------------------------------------------------------------------------
    # Crafty API helpers
    # -------------------------------------------------------------------------

    async def _crafty_create_server(
        self,
        session: aiohttp.ClientSession,
        crafty_url: str,
        token: str,
        server_name: str,
        zip_filename: str,
        port: int,
        min_ram: int,
        max_ram: int,
    ):
        """
        POST /api/v2/servers with import_zip create type.
        Returns (server_id, server_uuid).
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {
            "name": server_name,
            "monitoring_type": "minecraft_java",
            "minecraft_java_monitoring_data": {
                "host": "127.0.0.1",
                "port": port,
            },
            "create_type": "minecraft_java",
            "minecraft_java_create_data": {
                "create_type": "import_zip",
                "import_zip_create_data": {
                    "zip": zip_filename,
                    "zip_root": "/",
                    "agree_to_eula": False,
                    "server_properties_port": port,
                    "mem_min": min_ram,
                    "mem_max": max_ram,
                },
            },
        }

        url = f"{crafty_url}/api/v2/servers"
        async with session.post(url, json=payload, headers=headers, ssl=False) as r:
            body = await r.json()
            if r.status not in (200, 201) or body.get("status") != "ok":
                error = body.get("error", r.status)
                detail = body.get("error_data", "")
                raise ValueError(f"Crafty API returned error: `{error}` — {detail}")

        server_uuid = body["data"].get("new_server_uuid") or body["data"].get("new_server_id")
        server_id = body["data"].get("new_server_id", server_uuid)
        return server_id, server_uuid

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    @staticmethod
    def _embed(title: str, description: str, color: discord.Color) -> discord.Embed:
        return discord.Embed(title=title, description=description, color=color)

    # -------------------------------------------------------------------------
    # Server list / start / stop
    # -------------------------------------------------------------------------

    @commands.command(name="listservers", aliases=["allservers"])
    @commands.guild_only()
    async def list_servers(self, ctx: commands.Context):
        """List all servers registered in Crafty."""
        if not await self._can_use(ctx):
            return await ctx.send("You don't have permission to do that.")

        cfg = await self.config.all()
        if not cfg["crafty_url"] or not cfg["crafty_token"]:
            return await ctx.send("CraftyServer isn't configured yet.")

        async with aiohttp.ClientSession() as session:
            try:
                servers = await self._crafty_list_servers(
                    session, cfg["crafty_url"], cfg["crafty_token"]
                )
            except Exception as e:
                return await ctx.send(f"❌ Crafty API error: {e}")

        if not servers:
            return await ctx.send("No servers found in Crafty.")

        embed = discord.Embed(title="Crafty Servers", color=discord.Color.blurple())
        for s in servers:
            status = s.get("running", False)
            icon = "🟢" if status else "🔴"
            embed.add_field(
                name=f"{icon} {s['server_name']}",
                value=f"`{s['server_id']}`  •  port `{s.get('server_port', '?')}`",
                inline=False,
            )
        embed.set_footer(text="Use the server ID with startserver / stopserver.")
        await ctx.send(embed=embed)

    @commands.command(name="startserver")
    @commands.guild_only()
    async def start_server_cmd(self, ctx: commands.Context, server_id: str):
        """Start a Crafty server by its UUID."""
        await self._run_server_action(ctx, server_id, "start_server")

    @commands.command(name="stopserver")
    @commands.guild_only()
    async def stop_server_cmd(self, ctx: commands.Context, server_id: str):
        """Stop a Crafty server by its UUID."""
        await self._run_server_action(ctx, server_id, "stop_server")

    async def _run_server_action(self, ctx: commands.Context, server_id: str, action: str):
        if not await self._can_use(ctx):
            return await ctx.send("You don't have permission to do that.")

        cfg = await self.config.all()
        if not cfg["crafty_url"] or not cfg["crafty_token"]:
            return await ctx.send("CraftyServer isn't configured yet.")

        label = "Starting" if action == "start_server" else "Stopping"
        msg = await ctx.send(
            embed=self._embed(f"⏳ {label}...", f"Server: `{server_id}`", discord.Color.blurple())
        )

        async with aiohttp.ClientSession() as session:
            try:
                await self._crafty_server_action(
                    session, cfg["crafty_url"], cfg["crafty_token"], server_id, action
                )
            except Exception as e:
                return await msg.edit(
                    embed=self._embed("❌ Action Failed", str(e), discord.Color.red())
                )

        done_label = "Started" if action == "start_server" else "Stopped"
        icon = "✅" if action == "start_server" else "🛑"
        await msg.edit(
            embed=self._embed(
                f"{icon} {done_label}",
                f"Server `{server_id}` has been {done_label.lower()}.",
                discord.Color.green() if action == "start_server" else discord.Color.orange(),
            )
        )

    # -------------------------------------------------------------------------
    # Crafty API — list & action
    # -------------------------------------------------------------------------

    async def _crafty_list_servers(
        self, session: aiohttp.ClientSession, crafty_url: str, token: str
    ):
        """GET /api/v2/servers — returns list of server dicts."""
        headers = {"Authorization": f"Bearer {token}"}
        async with session.get(
            f"{crafty_url}/api/v2/servers", headers=headers, ssl=False
        ) as r:
            r.raise_for_status()
            body = await r.json()

        if body.get("status") != "ok":
            raise ValueError(body.get("error", "Unknown error"))

        # Each item may nest server details under a 'server_id' object
        results = []
        for item in body.get("data", []):
            if isinstance(item.get("server_id"), dict):
                srv = item["server_id"]
            else:
                srv = item
            results.append(srv)
        return results

    async def _crafty_server_action(
        self,
        session: aiohttp.ClientSession,
        crafty_url: str,
        token: str,
        server_id: str,
        action: str,  # "start_server" | "stop_server"
    ):
        """POST /api/v2/servers/{server_id}/action/{action}"""
        valid = {"start_server", "stop_server", "restart_server", "kill_server"}
        if action not in valid:
            raise ValueError(f"Invalid action `{action}`. Valid: {valid}")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        url = f"{crafty_url}/api/v2/servers/{server_id}/action/{action}"
        async with session.post(url, headers=headers, ssl=False) as r:
            body = await r.json()
            if r.status not in (200, 201) or body.get("status") != "ok":
                error = body.get("error", r.status)
                detail = body.get("error_data", "")
                raise ValueError(f"`{error}` — {detail}")

    # -------------------------------------------------------------------------
    # Info command
    # -------------------------------------------------------------------------

    @commands.command(name="serverhelp")
    async def server_help(self, ctx: commands.Context):
        """Show how to create a server using a CurseForge link."""
        embed = discord.Embed(
            title="Creating a Minecraft Server",
            description=(
                "Paste a CurseForge modpack link to spin up a server automatically.\n\n"
                "**Command:**\n"
                "```\n"
                f"{ctx.prefix}createserver <curseforge_url> [name] [port] [min_ram] [max_ram]\n"
                "```\n"
                "**Examples:**\n"
                f"`{ctx.prefix}createserver https://www.curseforge.com/minecraft/modpacks/all-the-mods-9`\n"
                f"`{ctx.prefix}createserver https://www.curseforge.com/minecraft/modpacks/atm9 \"ATM9\" 25570 2 8`\n\n"
                "**Other commands:**\n"
                f"`{ctx.prefix}listservers` — list all servers and their IDs\n"
                f"`{ctx.prefix}startserver <id>` — start a server\n"
                f"`{ctx.prefix}stopserver <id>` — stop a server\n\n"
                "**Notes:**\n"
                "• The modpack must have a server pack uploaded to CurseForge.\n"
                "• After creation, a superuser must accept the EULA in Crafty before starting.\n"
                "• RAM values are in GB."
            ),
            color=discord.Color.blurple(),
        )
        await ctx.send(embed=embed)
