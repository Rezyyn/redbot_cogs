import discord
import aiohttp
import asyncio
import json
import logging
from datetime import timezone, datetime
from redbot.core import commands, Config, checks
from typing import Optional, Dict

logger = logging.getLogger("red.lokibackfill")

class SilentProgress:
    """Silent progress tracker that only logs to console"""
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.message_count = 0
        self.channel_count = 0
        self.current_channel = None
    
    def update(self, channel_name, count):
        self.current_channel = channel_name
        self.message_count += count
        self.channel_count += 1
        
        # Log progress every 1000 messages or new channel
        if count > 0 and (self.message_count % 1000 == 0 or count < 100):
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            rate = self.message_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"Backfilled #{channel_name} | Total: {self.message_count} msgs "
                f"| {rate:.1f} msg/s | {self.channel_count} channels"
            )

async def backfill_channel_to_loki(
    channel, 
    loki_url, 
    headers, 
    progress: SilentProgress,
    limit=None, 
    batch_size=50, 
    delay=0.1
):
    """Backfill channel messages to Loki with batching and rate limiting"""
    message_batch = []
    total_sent = 0
    
    try:
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
                    await send_batch_to_loki(message_batch, loki_url, headers)
                    total_sent += len(message_batch)
                    progress.update(channel.name, len(message_batch))
                    message_batch = []
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing message {message.id}: {e}")
        
        # Send remaining messages in batch
        if message_batch:
            await send_batch_to_loki(message_batch, loki_url, headers)
            total_sent += len(message_batch)
            progress.update(channel.name, len(message_batch))
    
    except discord.Forbidden:
        logger.warning(f"Missing permissions in #{channel.name}")
    except discord.HTTPException as e:
        logger.error(f"Discord API error in #{channel.name}: {e}")
    
    return total_sent

async def send_batch_to_loki(batch, loki_url, headers):
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
                    logger.error(f"Loki error: {response.status} - {error}")
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")

class LokiBackfill(commands.Cog):
    """Silent backfill of Discord messages to Loki"""

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)
        self.config.register_global(
            loki_url="",
            auth_token="",
            batch_size=50,
            delay=0.1
        )
        self.active_backfills: Dict[int, asyncio.Task] = {}

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

    @lokibackfill.command(name="dmbackfill")
    async def dm_backfill(self, ctx, guild_id: Optional[int] = None, limit: Optional[int] = None):
        """
        Start silent backfill via DM
        
        Usage: [p]lokibackfill dmbackfill [guild_id] [limit]
        """
        # Get all guilds if no ID specified
        if guild_id is None:
            guild_list = "\n".join(f"{g.id} - {g.name}" for g in self.bot.guilds)
            return await ctx.send(
                f"🔍 Available guilds:\n"
                f"{guild_list}\n\n"
                f"Usage: `{ctx.prefix}lokibackfill dmbackfill <guild_id> [limit]`"
            )
        
        # Check if backfill already running
        if guild_id in self.active_backfills:
            return await ctx.send("⚠️ Backfill already running for this guild")
        
        # Find guild
        guild = self.bot.get_guild(guild_id)
        if not guild:
            return await ctx.send("❌ Guild not found")
        
        # Get config
        loki_url = await self.config.loki_url()
        if not loki_url:
            return await ctx.send("❌ Loki URL not configured. Use `lokibackfill seturl`")
        
        headers = await self.get_loki_headers()
        batch_size = await self.config.batch_size()
        delay = await self.config.delay()
        
        # Start silent backfill
        progress = SilentProgress()
        text_channels = [
            ch for ch in guild.text_channels 
            if ch.permissions_for(guild.me).read_message_history
        ]
        
        # Create and store backfill task
        task = asyncio.create_task(
            self.silent_backfill_guild(
                guild, text_channels, loki_url, headers, 
                progress, limit, batch_size, delay
            )
        )
        self.active_backfills[guild_id] = task
        
        # Send initial confirmation
        await ctx.send(
            f"🔁 Starting SILENT backfill for **{guild.name}** "
            f"({len(text_channels)} channels). You'll get a DM when complete."
        )
        
        # Add callback to notify when done
        task.add_done_callback(
            lambda t: asyncio.create_task(
                self.notify_backfill_complete(ctx, guild_id, t)
            )
        )

    async def silent_backfill_guild(
        self,
        guild,
        channels,
        loki_url,
        headers,
        progress,
        limit=None,
        batch_size=50,
        delay=0.1
    ):
        """Perform silent backfill without sending any messages"""
        total_messages = 0
        logger.info(f"Starting silent backfill for {guild.name} ({len(channels)} channels)")
        
        for channel in channels:
            try:
                if guild.id not in self.active_backfills:
                    logger.info("Backfill cancelled")
                    break
                    
                count = await backfill_channel_to_loki(
                    channel, loki_url, headers, progress,
                    limit=limit, 
                    batch_size=batch_size,
                    delay=delay
                )
                total_messages += count
                logger.info(f"Completed #{channel.name}: {count} messages")
                
            except Exception as e:
                logger.error(f"Error in #{channel.name}: {str(e)}")
        
        return total_messages, (datetime.utcnow() - progress.start_time).total_seconds()

    async def notify_backfill_complete(self, ctx, guild_id, task):
        """Notify user via DM when backfill completes"""
        try:
            # Get task results
            total_messages, duration = await task
            
            # Prepare DM message
            guild = self.bot.get_guild(guild_id)
            guild_name = guild.name if guild else f"Guild {guild_id}"
            rate = total_messages / duration if duration > 0 else 0
            
            try:
                dm_channel = ctx.author.dm_channel or await ctx.author.create_dm()
                await dm_channel.send(
                    f"🏁 **{guild_name}** backfill complete!\n"
                    f"• {total_messages} messages processed\n"
                    f"• Took {duration:.1f} seconds\n"
                    f"• Average {rate:.1f} msg/s"
                )
            except discord.Forbidden:
                logger.warning(f"Could not DM user {ctx.author.id}")
            
            # Clean up
            if guild_id in self.active_backfills:
                del self.active_backfills[guild_id]
                
        except asyncio.CancelledError:
            # Backfill was cancelled
            try:
                dm_channel = ctx.author.dm_channel or await ctx.author.create_dm()
                await dm_channel.send(f"⏹️ Backfill cancelled for **{guild_name}**")
            except discord.Forbidden:
                logger.warning(f"Could not DM user {ctx.author.id}")
            
            if guild_id in self.active_backfills:
                del self.active_backfills[guild_id]
                
        except Exception as e:
            logger.error(f"Error notifying backfill completion: {str(e)}")
            if guild_id in self.active_backfills:
                del self.active_backfills[guild_id]

    @lokibackfill.command(name="cancel")
    async def cancel_backfill(self, ctx, guild_id: int):
        """Cancel an active backfill"""
        if guild_id not in self.active_backfills:
            return await ctx.send("ℹ️ No active backfill for this guild")
        
        task = self.active_backfills[guild_id]
        task.cancel()
        
        try:
            await task  # Wait for cancellation to complete
        except asyncio.CancelledError:
            pass
            
        del self.active_backfills[guild_id]
        await ctx.send(f"⏹️ Backfill cancelled for guild {guild_id}")

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CheckFailure):
            await ctx.send("❌ You don't have permission to use this command")
        elif isinstance(error, commands.CommandInvokeError):
            logger.error(f"Command error: {str(error.original)}")