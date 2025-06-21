import aiohttp
import json
import time
from datetime import datetime, timezone
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
                "avatar": str(message.author.avatar.url) if message.author.avatar and message.author.avatar.url else None
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
        
        return await self._send_to_loki(stream, [[ns_timestamp, json.dumps(message_data)]]
    
    async def log_edit(self, before, after):
        """Log message edits"""
        ns_timestamp = str(int(time.time() * 1e9))
        
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
        
        stream = {
            "app": "discord-bot",
            "event_type": "edit",
            "channel_id": str(after.channel.id),
            "guild_id": str(after.guild.id) if after.guild else "DM"
        }
        
        return await self._send_to_loki(stream, [[ns_timestamp, json.dumps(edit_data)]])
    
    async def log_delete(self, message):
        """Log message deletions"""
        ns_timestamp = str(int(time.time() * 1e9))
        
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
        
        stream = {
            "app": "discord-bot",
            "event_type": "delete",
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id) if message.guild else "DM"
        }
        
        return await self._send_to_loki(stream, [[ns_timestamp, json.dumps(delete_data)]])
    
    async def log_reaction(self, reaction, user, action_type):
        """Log reaction add/remove"""
        ns_timestamp = str(int(time.time() * 1e9))
        
        reaction_data = {
            "event_type": action_type,
            "message_id": str(reaction.message.id),
            "channel_id": str(reaction.message.channel.id),
            "guild_id": str(reaction.message.guild.id) if reaction.message.guild else "DM",
            "user": {
                "id": str(user.id),
                "name": user.name
            },
            "emoji": {
                "name": reaction.emoji.name,
                "id": str(reaction.emoji.id) if hasattr(reaction.emoji, 'id') else None,
                "custom": isinstance(reaction.emoji, discord.Emoji)
            }
        }
        
        stream = {
            "app": "discord-bot",
            "event_type": "reaction",
            "channel_id": str(reaction.message.channel.id),
            "guild_id": str(reaction.message.guild.id) if reaction.message.guild else "DM"
        }
        
        return await self._send_to_loki(stream, [[ns_timestamp, json.dumps(reaction_data)]])

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
    
    @commands.Cog.listener()
    async def on_message_edit(self, before, after):
        if before.content == after.content or after.author.bot:
            return
        if not await self.config.enabled():
            return
            
        helper = await self.get_loki_helper()
        if helper:
            await helper.log_edit(before, after)

    @commands.Cog.listener()
    async def on_message_delete(self, message):
        if message.author.bot:
            return
        if not await self.config.enabled():
            return
            
        helper = await self.get_loki_helper()
        if helper:
            await helper.log_delete(message)

    @commands.Cog.listener()
    async def on_reaction_add(self, reaction, user):
        if user.bot:
            return
        if not await self.config.enabled():
            return
            
        helper = await self.get_loki_helper()
        if helper:
            await helper.log_reaction(reaction, user, "reaction_add")

    @commands.Cog.listener()
    async def on_reaction_remove(self, reaction, user):
        if user.bot:
            return
        if not await self.config.enabled():
            return
            
        helper = await self.get_loki_helper()
        if helper:
            await helper.log_reaction(reaction, user, "reaction_remove")
    
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
        
        # Create a proper test message object
        class TestMessage:
            content = "Loki Logger Test Message"
            id = 123456789
            
            class Author:
                id = 987654321
                name = "TestUser"
                discriminator = "0001"
                bot = False
                avatar = None
                
                class Avatar:
                    url = "https://example.com/avatar.png"
                
                @property
                def avatar(self):
                    return self.Avatar() if hasattr(self, 'Avatar') else None
                    
            author = Author()
            created_at = datetime.now(timezone.utc)
            
            class Channel:
                id = 1122334455
                name = "test-channel"
                
                class Category:
                    name = "Test Category"
                
                @property
                def category(self):
                    return self.Category()
            
            channel = Channel()
            
            class Guild:
                id = 5544332211
                name = "Test Guild"
            
            guild = Guild()
            attachments = []
            embeds = []
        
        test_message = TestMessage()
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