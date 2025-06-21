import aiohttp
import json
import time
from datetime import datetime
from redbot.core import commands, Config
import discord

class LokiHelper:
    """Handles Loki communication"""
    def __init__(self, loki_url, auth_token=None):
        self.loki_url = loki_url
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
    
    async def _send_to_loki(self, stream, values):
        payload = {"streams": [{"stream": stream, "values": values}]}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.loki_url,
                    data=json.dumps(payload),
                    headers=self.headers,
                    timeout=10
                ) as response:
                    if response.status != 204:
                        error = await response.text()
                        return f"Loki error: {response.status} - {error}"
                    return "Success"
        except Exception as e:
            return f"Connection error: {str(e)}"

    async def log_message(self, message):
        """Log message with all metadata"""
        ns_timestamp = str(int(time.time() * 1e9))
        message_data = {
            "content": message.content,
            "message_id": str(message.id),
            "author": {
                "id": str(message.author.id),
                "name": message.author.name,
                "discriminator": message.author.discriminator,
                "bot": message.author.bot,
                "avatar": str(message.author.avatar.url) if message.author.avatar else None
            },
            "channel": {
                "id": str(message.channel.id),
                "name": message.channel.name if hasattr(message.channel, 'name') else "DM",
                "category": message.channel.category.name if hasattr(message.channel, 'category') and message.channel.category else None
            },
            "guild": {
                "id": str(message.guild.id),
                "name": message.guild.name
            } if message.guild else None,
            "created_at": message.created_at.isoformat(),
            "attachments": [{
                "url": att.url,
                "filename": att.filename,
                "size": att.size,
                "content_type": att.content_type
            } for att in message.attachments],
            "embeds": [{
                "type": embed.type,
                "title": embed.title,
                "description": embed.description
            } for embed in message.embeds]
        }
        
        stream = {
            "app": "discord-bot",
            "event_type": "message",
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id) if message.guild else "DM"
        }
        
        return await self._send_to_loki(stream, [[ns_timestamp, json.dumps(message_data)]])
    
    # Similar methods for log_edit, log_delete, log_reaction would go here
    # (Full implementations omitted for brevity but follow same pattern)

class LokiLogger(commands.Cog):
    """Logs Discord events to Grafana Loki"""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1234567890)
        self.config.register_global(
            loki_url="",
            auth_token="",
            enabled=False
        )
    
    async def get_loki_helper(self):
        """Create LokiHelper instance with current config"""
        config = await self.config.all()
        if not config["loki_url"]:
            return None
        return LokiHelper(config["loki_url"], config["auth_token"])
    
    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author.bot:
            return
        if not await self.config.enabled():
            return
            
        helper = await self.get_loki_helper()
        if helper:
            await helper.log_message(message)
    
    # Similar listeners for:
    # on_message_edit, on_message_delete, on_reaction_add, on_reaction_remove
    # (Implement following same pattern as on_message)
    
    @commands.group()
    @commands.is_owner()
    async def lokiset(self, ctx):
        """Configure Loki logging"""
        pass
        
    @lokiset.command()
    async def url(self, ctx, url: str):
        """Set Loki endpoint URL"""
        if not url.startswith("http"):
            return await ctx.send("❌ Invalid URL format")
        await self.config.loki_url.set(url)
        await ctx.tick()
        
    @lokiset.command()
    async def token(self, ctx, token: str):
        """Set authentication token (if required)"""
        await self.config.auth_token.set(token)
        await ctx.send("Token set!")
        
    @lokiset.command()
    async def toggle(self, ctx):
        """Enable/disable logging"""
        current = await self.config.enabled()
        await self.config.enabled.set(not current)
        status = "ENABLED" if not current else "DISABLED"
        await ctx.send(f"✅ Logging {status}")

    @lokiset.command()
    async def test(self, ctx):
        """Test Loki connection"""
        helper = await self.get_loki_helper()
        if not helper:
            return await ctx.send("❌ Loki URL not configured!")
        
        test_message = type('Obj', (object,), {'content': 'Test', 'id': '123', 'author': ctx.author,
                                              'channel': ctx.channel, 'guild': ctx.guild,
                                              'attachments': [], 'embeds': []})
        result = await helper.log_message(test_message)
        if "Success" in result:
            await ctx.send("✅ Connection successful!")
        else:
            await ctx.send(f"❌ Error: {result}")
    
    @lokiset.command()
    async def settings(self, ctx):
        """Show current Loki configuration"""
        config = await self.config.all()
        status = "ENABLED" if config["enabled"] else "DISABLED"
        msg = (
            f"**Loki URL:** `{config['loki_url'] or 'Not set'}`\n"
            f"**Auth Token:** `{'Set' if config['auth_token'] else 'Not set'}`\n"
            f"**Status:** {status}"
        )
        await ctx.send(msg)