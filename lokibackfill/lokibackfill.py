import discord
import aiohttp
import asyncio
import json
from datetime import timezone
from redbot.core import commands

async def backfill_channel_to_loki(channel, loki_url, headers, limit=None):
    async for message in channel.history(limit=limit, oldest_first=True):
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
        stream = {
            "app": "discord-bot",
            "event_type": "message",
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id) if message.guild else "DM"
        }
        payload = {
            "streams": [
                {
                    "stream": stream,
                    "values": [
                        [ns_timestamp, json.dumps(message_data)]
                    ]
                }
            ]
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                loki_url,
                data=json.dumps(payload),
                headers=headers,
                timeout=10
            ) as response:
                if response.status != 204:
                    error = await response.text()
                    print(f"Loki error: {response.status} - {error}")
                else:
                    print(f"Sent message {message.id} from #{channel.name}")
        await asyncio.sleep(0.1)  # Adjust to avoid hitting Discord rate limits

class LokiBackfill(commands.Cog):
    """Backfill all text channels' messages in a guild to Loki"""

    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    @commands.guild_only()
    @commands.is_owner()
    async def lokibackfillall(self, ctx, limit: int = None):
        """
        Backfill all accessible text channels in this guild to Loki.
        Optionally limit messages per channel (default: all).
        Usage: [p]lokibackfillall [limit]
        """
        loki_url = "http://192.168.1.41:3100/loki/api/v1/push"  # Change as needed
        headers = {"Content-Type": "application/json"}
        # If needed: headers["Authorization"] = "Bearer <token>"
        guild = ctx.guild
        text_channels = [ch for ch in guild.text_channels if ch.permissions_for(guild.me).read_message_history]
        total_channels = len(text_channels)
        await ctx.send(f"Starting backfill for {total_channels} channels. This may take a while.")
        for idx, channel in enumerate(text_channels, 1):
            await ctx.send(f"Backfilling #{channel.name} ({idx}/{total_channels})...")
            try:
                await backfill_channel_to_loki(channel, loki_url, headers, limit=limit)
            except Exception as e:
                await ctx.send(f"Error in #{channel.name}: {e}")
        await ctx.send("Backfill complete for all accessible channels.")
