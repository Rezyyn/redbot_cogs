"""
AMPServers - Red-Bot Cog
Lists and controls game servers hosted on a CubeCoders AMP (Application Management Panel) instance.

Commands:
  [p]ampservers            - List all instances with status, game type and player count
  [p]ampserver  <name>     - Detailed view of a single instance (metrics, uptime, players)
  [p]ampstart   <name>     - Start an instance  (admin / allowed roles)
  [p]ampstop    <name>     - Stop an instance   (admin / allowed roles)
  [p]amprestart <name>     - Restart an instance(admin / allowed roles)
  [p]ampset     <key> <val>- Configure the cog  (bot owner / guild admin only)

Config keys (set with [p]ampset):
  url            - AMP panel base URL, e.g. http://192.168.1.10:8080
  username       - AMP username (should be a dedicated bot account)
  password       - AMP password
  allow_roles    - Comma-separated role IDs allowed to start/stop servers (empty = admins only)
  public_ip      - Your public-facing IP/hostname shown in server detail embeds (optional)
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import discord
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import box

log = logging.getLogger("red.rezcog.ampservers")

# ─── AMP state code → human label + embed colour ───────────────────────────
_STATE = {
    -1: ("Unknown",     discord.Colour.greyple()),
     0: ("Stopped",     discord.Colour.red()),
     5: ("PreStart",    discord.Colour.orange()),
     7: ("Configuring", discord.Colour.orange()),
    10: ("Running",     discord.Colour.green()),
    20: ("Restarting",  discord.Colour.gold()),
    30: ("Stopping",    discord.Colour.orange()),
    45: ("Undefined",   discord.Colour.greyple()),
    50: ("Hibernating", discord.Colour.blue()),
    60: ("Offline",     discord.Colour.dark_grey()),
    70: ("Sleeping",    discord.Colour.blue()),
}

# Status emoji
_EMOJI = {
    "Running":     "🟢",
    "Stopped":     "🔴",
    "Stopping":    "🟡",
    "Restarting":  "🟡",
    "PreStart":    "🟡",
    "Configuring": "🟡",
    "Hibernating": "💤",
    "Sleeping":    "💤",
    "Offline":     "⚫",
    "Unknown":     "❓",
    "Undefined":   "❓",
}


def _state_label(code: int) -> tuple[str, discord.Colour]:
    return _STATE.get(code, ("Unknown", discord.Colour.greyple()))


def _fmt_uptime(seconds: Optional[float]) -> str:
    if not seconds or seconds < 0:
        return "—"
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    parts  = []
    if d: parts.append(f"{d}d")
    if h: parts.append(f"{h}h")
    if m: parts.append(f"{m}m")
    if not parts or s:
        parts.append(f"{s}s")
    return " ".join(parts)


# ─── Low-level AMP HTTP client ───────────────────────────────────────────────

class AMPClient:
    """Thin async wrapper around the AMP JSON API."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url  = base_url.rstrip("/")
        self.username  = username
        self.password  = password
        self._session_id: Optional[str] = None
        self._http: Optional[aiohttp.ClientSession] = None

    # ── Session management ────────────────────────────────────────────────

    def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                connector=aiohttp.TCPConnector(ssl=False),   # AMP is often self-signed
            )
        return self._http

    async def close(self):
        if self._http and not self._http.closed:
            await self._http.close()

    async def _login(self) -> str:
        """Authenticate and cache the session ID."""
        http = self._ensure_http()
        url  = f"{self.base_url}/API/Core/Login"
        payload = {
            "username":   self.username,
            "password":   self.password,
            "token":      "",
            "rememberMe": False,
        }
        async with http.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        if not data.get("success"):
            raise RuntimeError(f"AMP login failed: {data.get('resultReason', 'unknown reason')}")

        self._session_id = data["sessionID"]
        return self._session_id

    async def _call(self, endpoint: str, extra: Optional[dict] = None, *, retried: bool = False) -> dict:
        """
        POST to an AMP API endpoint.
        Automatically re-authenticates once on session expiry.
        """
        if not self._session_id:
            await self._login()

        http    = self._ensure_http()
        url     = f"{self.base_url}/API/{endpoint}"
        payload = {"sessionId": self._session_id, **(extra or {})}

        async with http.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        # AMP returns {"result": ..., "status": false} on bad sessions
        if isinstance(data, dict) and data.get("status") is False and not retried:
            self._session_id = None
            return await self._call(endpoint, extra, retried=True)

        return data

    async def _instance_call(self, instance_id: str, endpoint: str, extra: Optional[dict] = None) -> dict:
        """
        Call an endpoint on a specific child instance through the ADS proxy.
        URL pattern: ADSModule/Servers/{id}/API/{endpoint}
        """
        proxy_endpoint = f"ADSModule/Servers/{instance_id}/API/{endpoint}"
        return await self._call(proxy_endpoint, extra)

    # ── ADS / Instance API ────────────────────────────────────────────────

    async def get_instances(self) -> list[dict]:
        """
        Return a flat list of all instances (across all ADS targets).
        Each dict is the raw AMP instance object.
        """
        data = await self._call("ADSModule/GetInstances")
        instances: list[dict] = []

        # GetInstances returns a list of ADS controller objects, each containing
        # an "AvailableInstances" list of the actual game server instances.
        result = data if isinstance(data, list) else data.get("result", [])
        for controller in result:
            for inst in controller.get("AvailableInstances", []):
                instances.append(inst)

        return instances

    async def get_instance_status(self, instance_id: str) -> dict:
        """Fetch live Core/GetStatus for a single instance."""
        try:
            return await self._instance_call(instance_id, "Core/GetStatus")
        except Exception as exc:
            log.debug("get_instance_status(%s) failed: %s", instance_id, exc)
            return {}

    async def start_instance(self, instance_id: str) -> dict:
        return await self._call("ADSModule/StartInstance", {"InstanceId": instance_id})

    async def stop_instance(self, instance_id: str) -> dict:
        return await self._call("ADSModule/StopInstance", {"InstanceId": instance_id})

    async def restart_instance(self, instance_id: str) -> dict:
        return await self._call("ADSModule/RestartInstance", {"InstanceId": instance_id})


# ─── Cog ─────────────────────────────────────────────────────────────────────

class AMPServers(commands.Cog):
    """Manage and list game servers hosted on a CubeCoders AMP panel."""

    def __init__(self, bot: Red):
        self.bot   = bot
        self.config = Config.get_conf(self, identifier=0x414D504353, force_registration=True)
        self.config.register_global(
            url="",
            username="",
            password="",
            allow_roles=[],
            public_ip="",
        )
        self._client: Optional[AMPClient] = None

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _get_client(self) -> AMPClient:
        """Return a cached (or freshly created) AMPClient."""
        url      = await self.config.url()
        username = await self.config.username()
        password = await self.config.password()

        if not url or not username or not password:
            raise RuntimeError(
                "AMP is not configured yet. Run `[p]ampset url`, "
                "`[p]ampset username`, and `[p]ampset password`."
            )

        # Recreate if credentials changed
        if (
            self._client is None
            or self._client.base_url != url.rstrip("/")
            or self._client.username != username
            or self._client.password != password
        ):
            if self._client:
                await self._client.close()
            self._client = AMPClient(url, username, password)

        return self._client

    async def _find_instance(self, instances: list[dict], name: str) -> Optional[dict]:
        """Case-insensitive search by FriendlyName or InstanceName."""
        name_lower = name.lower()
        for inst in instances:
            if (
                inst.get("FriendlyName", "").lower() == name_lower
                or inst.get("InstanceName", "").lower() == name_lower
            ):
                return inst
        return None

    async def _can_control(self, ctx: commands.Context) -> bool:
        """Returns True if the invoker may start/stop/restart servers."""
        if await ctx.bot.is_owner(ctx.author):
            return True
        if ctx.author.guild_permissions.administrator:
            return True
        allow_roles = await self.config.allow_roles()
        if allow_roles:
            author_role_ids = {r.id for r in ctx.author.roles}
            return bool(author_role_ids.intersection(set(allow_roles)))
        return False

    # ── Commands ──────────────────────────────────────────────────────────

    # -- Configuration -------------------------------------------------------

    @commands.group(name="ampset", invoke_without_command=True)
    @commands.admin_or_permissions(administrator=True)
    async def ampset(self, ctx: commands.Context):
        """Configure the AMP Servers cog."""
        cfg = {
            "url":         await self.config.url() or "(not set)",
            "username":    await self.config.username() or "(not set)",
            "password":    "••••••••" if await self.config.password() else "(not set)",
            "allow_roles": ", ".join(str(r) for r in await self.config.allow_roles()) or "(everyone with admin)",
            "public_ip":   await self.config.public_ip() or "(not set)",
        }
        embed = discord.Embed(title="AMP Servers — Configuration", colour=discord.Colour.blurple())
        for k, v in cfg.items():
            embed.add_field(name=k, value=v, inline=False)
        embed.set_footer(text="Use [p]ampset <key> <value> to update")
        await ctx.send(embed=embed)

    @ampset.command(name="url")
    @commands.is_owner()
    async def ampset_url(self, ctx: commands.Context, url: str):
        """Set the AMP panel base URL (e.g. http://192.168.1.10:8080)."""
        await self.config.url.set(url.rstrip("/"))
        self._client = None   # force reconnect
        await ctx.tick()

    @ampset.command(name="username")
    @commands.is_owner()
    async def ampset_username(self, ctx: commands.Context, username: str):
        """Set the AMP login username."""
        await self.config.username.set(username)
        self._client = None
        await ctx.tick()

    @ampset.command(name="password")
    @commands.is_owner()
    async def ampset_password(self, ctx: commands.Context, password: str):
        """Set the AMP login password. Run this in a DM to avoid exposing it in chat."""
        await self.config.password.set(password)
        self._client = None
        try:
            await ctx.message.delete()
        except (discord.Forbidden, discord.HTTPException):
            pass
        await ctx.tick()

    @ampset.command(name="public_ip")
    @commands.admin_or_permissions(administrator=True)
    async def ampset_public_ip(self, ctx: commands.Context, ip: str):
        """Set the public IP / hostname shown in server detail embeds."""
        await self.config.public_ip.set(ip)
        await ctx.tick()

    @ampset.command(name="allow_roles")
    @commands.admin_or_permissions(administrator=True)
    async def ampset_allow_roles(self, ctx: commands.Context, *roles: discord.Role):
        """Set which roles can start/stop/restart servers (mention or IDs). Pass nothing to clear."""
        ids = [r.id for r in roles]
        await self.config.allow_roles.set(ids)
        if ids:
            await ctx.send(f"Control roles set: {', '.join(r.mention for r in roles)}")
        else:
            await ctx.send("Control roles cleared — only server admins may control instances.")

    # -- Listing -------------------------------------------------------------

    @commands.command(name="ampservers")
    async def ampservers(self, ctx: commands.Context):
        """List all game server instances on the AMP panel."""
        async with ctx.typing():
            try:
                client    = await self._get_client()
                instances = await client.get_instances()
            except RuntimeError as exc:
                return await ctx.send(f"❌ {exc}")
            except aiohttp.ClientError as exc:
                return await ctx.send(f"❌ Could not reach AMP panel: {exc}")

        if not instances:
            return await ctx.send("No instances found on the AMP panel.")

        embed = discord.Embed(
            title="🖥️  AMP — Game Servers",
            colour=discord.Colour.blurple(),
            timestamp=datetime.now(tz=timezone.utc),
        )

        for inst in instances:
            name       = inst.get("FriendlyName") or inst.get("InstanceName", "Unknown")
            module     = inst.get("ModuleDisplayName") or inst.get("Module", "Unknown")
            state_code = inst.get("AppState", -1)
            label, _   = _state_label(state_code)
            emoji      = _EMOJI.get(label, "❓")
            running    = inst.get("IsRunning", False)

            # Player counts come from the Metrics block (populated when running)
            metrics    = inst.get("Metrics", {})
            players    = metrics.get("Active Users", {})
            cur_p      = players.get("RawValue", "—")
            max_p      = players.get("MaxValue",  "—")
            pcount     = f"{cur_p}/{max_p}" if running else "—"

            value = f"{emoji} **{label}**  ·  👥 {pcount}  ·  🎮 {module}"
            embed.add_field(name=name, value=value, inline=False)

        embed.set_footer(text=f"{len(instances)} instance(s)")
        await ctx.send(embed=embed)

    @commands.command(name="ampserver")
    async def ampserver(self, ctx: commands.Context, *, name: str):
        """Show detailed status for a named AMP instance."""
        async with ctx.typing():
            try:
                client    = await self._get_client()
                instances = await client.get_instances()
            except RuntimeError as exc:
                return await ctx.send(f"❌ {exc}")
            except aiohttp.ClientError as exc:
                return await ctx.send(f"❌ Could not reach AMP panel: {exc}")

            inst = await self._find_instance(instances, name)
            if inst is None:
                names = [i.get("FriendlyName") or i.get("InstanceName", "?") for i in instances]
                return await ctx.send(
                    f"❌ No instance named **{name}** found.\n"
                    f"Available: {', '.join(f'`{n}`' for n in names)}"
                )

            # Fetch live status from the instance itself
            inst_id = inst.get("InstanceID") or inst.get("InstanceId", "")
            live    = {}
            if inst_id:
                live = await client.get_instance_status(inst_id)

        display_name = inst.get("FriendlyName") or inst.get("InstanceName", "Unknown")
        module       = inst.get("ModuleDisplayName") or inst.get("Module", "Unknown")
        state_code   = inst.get("AppState", -1)
        label, colour = _state_label(state_code)
        emoji        = _EMOJI.get(label, "❓")

        embed = discord.Embed(
            title=f"{emoji}  {display_name}",
            colour=colour,
            timestamp=datetime.now(tz=timezone.utc),
        )
        embed.add_field(name="Game / Module", value=module, inline=True)
        embed.add_field(name="Status",        value=label,  inline=True)

        # IP info
        public_ip = await self.config.public_ip()
        port      = inst.get("Port") or live.get("Port", "")
        if public_ip and port:
            embed.add_field(name="Connect", value=f"`{public_ip}:{port}`", inline=True)
        elif port:
            embed.add_field(name="Port", value=str(port), inline=True)

        # Live metrics from Core/GetStatus
        if live:
            metrics = live.get("Metrics", {})

            # Players
            players = metrics.get("Active Users", {})
            if players:
                cur_p = players.get("RawValue", "—")
                max_p = players.get("MaxValue",  "—")
                embed.add_field(name="Players", value=f"{cur_p} / {max_p}", inline=True)

            # CPU
            cpu = metrics.get("CPU Usage", {})
            if cpu:
                embed.add_field(
                    name="CPU",
                    value=f"{cpu.get('RawValue', '—')} {cpu.get('Units', '%')}",
                    inline=True,
                )

            # Memory
            mem = metrics.get("Memory Usage", {})
            if mem:
                raw  = mem.get("RawValue", 0)
                maxv = mem.get("MaxValue",  0)
                unit = mem.get("Units", "MB")
                embed.add_field(
                    name="Memory",
                    value=f"{raw} / {maxv} {unit}",
                    inline=True,
                )

            # Uptime
            uptime_secs = live.get("Uptime") or live.get("UptimeSeconds")
            if uptime_secs is not None:
                embed.add_field(name="Uptime", value=_fmt_uptime(uptime_secs), inline=True)

        # Instance ID footer
        embed.set_footer(text=f"InstanceID: {inst_id}")
        await ctx.send(embed=embed)

    # -- Control commands (admin / allowed roles) ----------------------------

    async def _control(self, ctx: commands.Context, name: str, action: str):
        """Shared logic for start / stop / restart."""
        if not await self._can_control(ctx):
            return await ctx.send("❌ You don't have permission to control server instances.")

        async with ctx.typing():
            try:
                client    = await self._get_client()
                instances = await client.get_instances()
            except RuntimeError as exc:
                return await ctx.send(f"❌ {exc}")
            except aiohttp.ClientError as exc:
                return await ctx.send(f"❌ Could not reach AMP panel: {exc}")

            inst = await self._find_instance(instances, name)
            if inst is None:
                return await ctx.send(f"❌ No instance named **{name}** found.")

            inst_id = inst.get("InstanceID") or inst.get("InstanceId", "")
            if not inst_id:
                return await ctx.send("❌ Could not determine instance ID.")

            try:
                if action == "start":
                    await client.start_instance(inst_id)
                elif action == "stop":
                    await client.stop_instance(inst_id)
                elif action == "restart":
                    await client.restart_instance(inst_id)
            except aiohttp.ClientError as exc:
                return await ctx.send(f"❌ AMP API error: {exc}")

        verbs = {"start": "Starting", "stop": "Stopping", "restart": "Restarting"}
        display = inst.get("FriendlyName") or inst.get("InstanceName", name)
        await ctx.send(f"✅ {verbs[action]} **{display}**…")

    @commands.command(name="ampstart")
    async def ampstart(self, ctx: commands.Context, *, name: str):
        """Start a named AMP game server instance."""
        await self._control(ctx, name, "start")

    @commands.command(name="ampstop")
    async def ampstop(self, ctx: commands.Context, *, name: str):
        """Stop a named AMP game server instance."""
        await self._control(ctx, name, "stop")

    @commands.command(name="amprestart")
    async def amprestart(self, ctx: commands.Context, *, name: str):
        """Restart a named AMP game server instance."""
        await self._control(ctx, name, "restart")

    # ── Cog lifecycle ─────────────────────────────────────────────────────

    def cog_unload(self):
        if self._client:
            asyncio.create_task(self._client.close())


async def setup(bot: Red):
    await bot.add_cog(AMPServers(bot))
