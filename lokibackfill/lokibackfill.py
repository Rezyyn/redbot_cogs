import discord
import aiohttp
import asyncio
import json
import logging
from datetime import timezone, datetime
from redbot.core import commands, Config, checks
from typing import Optional

logger = logging.getLogger("red.lokibackfill")

async def backfill_channel_to_loki(channel, loki_url, headers, limit=None, batch_size=50, delay=0.1):
    """Backfill channel messages to Loki with batching and rate limiting"""
    message_batch = []
    total_sent = 0
    
    async for message in channel.history(limit=limit, oldest_first=True):
        try:
            # Create message data
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
                    "name": message.channel.name,
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
            
            # Create stream labels
            stream = {
                "app": "discord-bot",
                "event_type": "message",
                "channel_id": str(message.channel.id),
                "guild_id": str(message.guild.id) if message.guild else "DM",
                "source": "backfill"
            }
            
            # Add to batch
            message_batch.append({
                "stream": stream,
                "values": [[ns_timestamp, json.dumps(message_data)]]
            })
            
            # Send batch when full
            if len(message_batch) >= batch_size:
                await send_batch_to_loki(message_batch, loki_url, headers, channel.name)
                total_sent += len(message_batch)
                message_batch = []
                await asyncio.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error processing message {message.id} in #{channel.name}: {e}")
    
    # Send remaining messages in batch
    if message_batch:
        await send_batch_to_loki(message_batch, loki_url, headers, channel.name)
        total_sent += len(message_batch)
    
    return total_sent

async def send_batch_to_loki(batch, loki_url, headers, channel_name):
    """Send a batch of messages to Loki"""
    payload = {"streams": batch}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                loki_url,
                json=payload,
                headers=headers,
                timeout=30
            ) as response:
                if response.status != 204:
                    error = await response.text()
                    logger.error(f"Loki error ({channel_name}): {response.status} - {error}")
                else:
                    logger.info(f"Sent batch of {len(batch)} messages from #{channel_name}")
    except Exception as e:
        logger.error(f"Connection error ({channel_name}): {str(e)}")

class LokiBackfill(commands.Cog):
    """Backfill Discord messages to Loki with enhanced features"""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)
        self.config.register_global(
            loki_url="",
            auth_token="",
            batch_size=50,
            delay=0.1
        )
        self.active_backfills = {}

    async def get_loki_headers(self):
        """Get Loki headers with authentication"""
        auth_token = await self.config.auth_token()
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        return headers

    @commands.group()
    @checks.is_owner()
    async def lokibackfill(self, ctx):
        """Backfill messages to Loki"""
        pass

    @lokibackfill.command(name="config")
    async def backfill_config(self, ctx):
        """Show current backfill configuration"""
        config = await self.config.all()
        embed = discord.Embed(
            title="Loki Backfill Configuration",
            color=discord.Color.blue()
        )
        embed.add_field(name="Loki URL", value=config["loki_url"] or "Not set", inline=False)
        embed.add_field(name="Auth Token", value="Set" if config["auth_token"] else "Not set", inline=False)
        embed.add_field(name="Batch Size", value=config["batch_size"], inline=False)
        embed.add_field(name="Delay", value=f"{config['delay']}s", inline=False)
        await ctx.send(embed=embed)

    @lokibackfill.command(name="seturl")
    async def set_loki_url(self, ctx, url: str):
        """Set Loki push endpoint URL"""
        if not url.startswith("http"):
            return await ctx.send("❌ Invalid URL format")
        await self.config.loki_url.set(url)
        await ctx.send(f"✅ Loki URL set to `{url}`")

    @lokibackfill.command(name="settoken")
    async def set_auth_token(self, ctx, token: str):
        """Set Loki authentication token"""
        await self.config.auth_token.set(token)
        await ctx.send("✅ Auth token set")

    @lokibackfill.command(name="setbatch")
    async def set_batch_size(self, ctx, size: int):
        """Set number of messages per batch (1-500)"""
        if size < 1 or size > 500:
            return await ctx.send("❌ Batch size must be between 1 and 500")
        await self.config.batch_size.set(size)
        await ctx.send(f"✅ Batch size set to {size}")

    @lokibackfill.command(name="setdelay")
    async def set_delay(self, ctx, delay: float):
        """Set delay between batches in seconds (0.01-5.0)"""
        if delay < 0.01 or delay > 5.0:
            return await ctx.send("❌ Delay must be between 0.01 and 5.0 seconds")
        await self.config.delay.set(delay)
        await ctx.send(f"✅ Delay set to {delay} seconds")

    @lokibackfill.command(name="guild")
    async def backfill_guild(self, ctx, limit: Optional[int] = None):
        """
        Backfill all accessible text channels in this guild
        
        Usage: [p]lokibackfill guild [limit]
        """
        if ctx.guild is None:
            return await ctx.send("❌ This command must be used in a server")
        
        # Check if backfill already running
        if ctx.guild.id in self.active_backfills:
            return await ctx.send("⚠️ Backfill already running for this guild")
        
        # Get config
        loki_url = await self.config.loki_url()
        if not loki_url:
            return await ctx.send("❌ Loki URL not configured. Use `lokibackfill seturl`")
        
        headers = await self.get_loki_headers()
        batch_size = await self.config.batch_size()
        delay = await self.config.delay()
        
        # Start backfill
        self.active_backfills[ctx.guild.id] = True
        guild = ctx.guild
        text_channels = [
            ch for ch in guild.text_channels 
            if ch.permissions_for(guild.me).read_message_history
        ]
        
        total_channels = len(text_channels)
        await ctx.send(f"🚀 Starting backfill for {total_channels} channels. This may take a while.")
        
        total_messages = 0
        start_time = datetime.utcnow()
        
        for idx, channel in enumerate(text_channels, 1):
            try:
                if ctx.guild.id not in self.active_backfills:
                    await ctx.send("⏹️ Backfill cancelled")
                    break
                    
                status = await ctx.send(f"📥 Backfilling #{channel.name} ({idx}/{total_channels})...")
                count = await backfill_channel_to_loki(
                    channel, loki_url, headers, 
                    limit=limit, 
                    batch_size=batch_size,
                    delay=delay
                )
                total_messages += count
                await status.edit(content=f"✅ #{channel.name}: {count} messages backfilled")
            except Exception as e:
                await ctx.send(f"❌ Error in #{channel.name}: {str(e)}")
            finally:
                if channel != text_channels[-1]:
                    await asyncio.sleep(1)  # Brief pause between channels
        
        # Cleanup and report
        duration = (datetime.utcnow() - start_time).total_seconds()
        if ctx.guild.id in self.active_backfills:
            del self.active_backfills[ctx.guild.id]
            await ctx.send(
                f"🏁 Backfill complete! "
                f"Processed {total_messages} messages "
                f"in {duration:.1f} seconds "
                f"({total_messages/duration:.1f} msg/s)"
            )

    @lokibackfill.command(name="channel")
    async def backfill_channel(self, ctx, channel: discord.TextChannel, limit: Optional[int] = None):
        """
        Backfill a specific channel
        
        Usage: [p]lokibackfill channel <channel> [limit]
        """
        # Get config
        loki_url = await self.config.loki_url()
        if not loki_url:
            return await ctx.send("❌ Loki URL not configured. Use `lokibackfill seturl`")
        
        headers = await self.get_loki_headers()
        batch_size = await self.config.batch_size()
        delay = await self.config.delay()
        
        # Check permissions
        if not channel.permissions_for(ctx.guild.me).read_message_history:
            return await ctx.send(f"❌ Missing permissions in #{channel.name}")
        
        # Start backfill
        self.active_backfills[channel.id] = True
        start_time = datetime.utcnow()
        
        try:
            status = await ctx.send(f"📥 Starting backfill for #{channel.name}...")
            count = await backfill_channel_to_loki(
                channel, loki_url, headers, 
                limit=limit, 
                batch_size=batch_size,
                delay=delay
            )
            await status.edit(content=f"✅ #{channel.name}: {count} messages backfilled")
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")
        finally:
            if channel.id in self.active_backfills:
                del self.active_backfills[channel.id]
                duration = (datetime.utcnow() - start_time).total_seconds()
                await ctx.send(
                    f"🏁 Channel backfill complete! "
                    f"{count} messages in {duration:.1f} seconds "
                    f"({count/duration:.1f} msg/s)"
                )

    @lokibackfill.command(name="cancel")
    async def cancel_backfill(self, ctx, target: Optional[str] = None):
        """Cancel an active backfill"""
        if target == "guild" and ctx.guild:
            if ctx.guild.id in self.active_backfills:
                del self.active_backfills[ctx.guild.id]
                await ctx.send("⏹️ Guild backfill cancelled")
            else:
                await ctx.send("ℹ️ No active backfill for this guild")
        elif target and target.startswith("<#") and target.endswith(">"):
            channel_id = int(target[2:-1])
            if channel_id in self.active_backfills:
                del self.active_backfills[channel_id]
                await ctx.send(f"⏹️ Channel backfill cancelled")
            else:
                await ctx.send("ℹ️ No active backfill for this channel")
        else:
            await ctx.send("❌ Please specify 'guild' or a #channel")

    @lokibackfill.command(name="dmbackfill")
    async def dm_backfill(self, ctx, guild_id: Optional[int] = None):
        """
        Start backfill via DM
        
        Usage: [p]lokibackfill dmbackfill [guild_id]
        """
        if guild_id is None:
            # Get all guilds the bot is in
            guilds = "\n".join(f"{g.id} - {g.name}" for g in self.bot.guilds)
            return await ctx.send(
                f"🔍 Please specify a guild ID:\n"
                f"{guilds}\n\n"
                f"Usage: `{ctx.prefix}lokibackfill dmbackfill <guild_id>`"
            )
        
        # Find guild
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return await ctx.send("❌ Guild not found")
        
        # Start backfill process
        await ctx.send(f"🔁 Starting backfill for **{guild.name}**...")
        fake_context = await self.bot.get_context(ctx.message)
        fake_context.guild = guild
        fake_context.channel = guild.system_channel or guild.text_channels[0]
        await self.backfill_guild(fake_context)

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CheckFailure):
            await ctx.send("❌ You don't have permission to use this command")
        elif isinstance(error, commands.CommandInvokeError):
            logger.error(f"Command error: {str(error.original)}")
            await ctx.send(f"⚠️ Error: {str(error.original)}")