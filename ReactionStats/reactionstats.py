import aiohttp
import json
import math
import time  # Added missing import
from datetime import datetime, timedelta
import dateparser
import discord
from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import pagify

class ReactionStats(commands.Cog):
    """Analyze message reaction statistics from Loki logs"""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=987654321)
        self.config.register_global(
            loki_url="",
            auth_token=""
        )
        
    async def get_loki_config(self):
        """Get Loki configuration from LokiLogger cog"""
        # Try to get config from LokiLogger
        lokilogger_cog = self.bot.get_cog("LokiLogger")
        if lokilogger_cog:
            loki_config = await lokilogger_cog.config.all()
            return loki_config.get("loki_url", ""), loki_config.get("auth_token", "")
        
        # Fallback to our own config
        config = await self.config.all()
        return config.get("loki_url", ""), config.get("auth_token", "")
    
    async def query_loki(self, query, hours=24):
        """Query Loki using LogQL"""
        loki_url, auth_token = await self.get_loki_config()
        if not loki_url:
            return None, "Loki URL not configured"
        
        # Convert hours to nanoseconds
        end_ns = int(time.time() * 1e9)
        start_ns = end_ns - int(hours * 3600 * 1e9)
        
        # Build query URL
        query_url = f"{loki_url.rstrip('/')}/loki/api/v1/query_range?query={query}&start={start_ns}&end={end_ns}&limit=1000"
        
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(query_url, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        error = await response.text()
                        return None, f"Loki error: {response.status} - {error}"
                    
                    data = await response.json()
                    return data, None
        except Exception as e:
            return None, f"Connection error: {str(e)}"
    
    async def get_top_reactions(self, hours=24, limit=10):
        """Get top reacted messages"""
        # Query for reaction_add events
        query = (
            'sum by (message_id, channel_id) ('
            '  count_over_time({event_type="reaction_add"} | json | __error__="" [1h])'
            ')'
        )
        
        data, error = await self.query_loki(query, hours)
        if error:
            return None, error
        
        # Parse results
        results = []
        for stream in data.get("data", {}).get("result", []):
            try:
                message_id = stream["metric"]["message_id"]
                channel_id = stream["metric"]["channel_id"]
                reaction_count = int(stream["value"][1])
                
                # Query for message content
                msg_query = (
                    f'{{event_type="message", message_id="{message_id}", channel_id="{channel_id}"}}'
                    '| json | line_format "{{.content}}"'
                )
                msg_data, msg_error = await self.query_loki(msg_query, hours)
                
                content = "Content not available"
                author = "Unknown"
                timestamp = 0
                
                if not msg_error and msg_data["data"]["result"]:
                    msg = msg_data["data"]["result"][0]
                    content = msg.get("value", ["", "Content not available"])[1]
                    
                    # Extract metadata from stream labels
                    author_id = msg["stream"].get("author_id", "")
                    timestamp_ns = int(msg["values"][0][0])
                    timestamp = timestamp_ns / 1e9  # Convert to seconds
                    
                    if author_id:
                        author = f"<@{author_id}>"
                
                results.append({
                    "message_id": message_id,
                    "channel_id": channel_id,
                    "reactions": reaction_count,
                    "content": content,
                    "author": author,
                    "timestamp": timestamp
                })
            except (KeyError, ValueError):
                continue
        
        # Sort and limit results
        results.sort(key=lambda x: x["reactions"], reverse=True)
        return results[:limit], None
    
    @commands.group()
    async def reactionstats(self, ctx):
        """Message reaction statistics"""
        pass
    
    @reactionstats.command()
    async def top(self, ctx, hours: int = 24, limit: int = 5):
        """
        Show top reacted messages
        
        Parameters:
        hours: Time window to analyze (default: 24 hours)
        limit: Number of results to show (default: 5)
        """
        if hours > 168:
            await ctx.send("⚠️ Maximum time window is 1 week (168 hours)")
            return
        if limit > 15:
            await ctx.send("⚠️ Maximum results limit is 15")
            return
        
        async with ctx.typing():
            results, error = await self.get_top_reactions(hours, limit)
            
            if error:
                await ctx.send(f"❌ Error: {error}")
                return
                
            if not results:
                await ctx.send("ℹ️ No reaction data found for the specified time period")
                return
            
            # Build embed
            embed = discord.Embed(
                title=f"🔥 Top {len(results)} Most Reacted Messages",
                description=f"From the last {hours} hours",
                color=discord.Color.gold()
            )
            
            for i, item in enumerate(results, 1):
                timestamp = datetime.utcfromtimestamp(item["timestamp"])
                content = item["content"][:100] + "..." if len(item["content"]) > 100 else item["content"]
                
                embed.add_field(
                    name=f"#{i} - {item['reactions']} reactions",
                    value=(
                        f"**Author:** {item['author']}\n"
                        f"**Channel:** <#{item['channel_id']}>\n"
                        f"**Posted:** {discord.utils.format_dt(timestamp, 'R')}\n"
                        f"**Content:** {content}\n"
                        f"[Jump to Message](https://discord.com/channels/{ctx.guild.id}/{item['channel_id']}/{item['message_id']})"
                    ),
                    inline=False
                )
            
            embed.set_footer(text=f"Message IDs: {' '.join(r['message_id'] for r in results)}")
            await ctx.send(embed=embed)
    
    @reactionstats.command()
    async def config(self, ctx):
        """Show current Loki configuration"""
        loki_url, auth_token = await self.get_loki_config()
        msg = (
            f"**Loki URL:** `{loki_url or 'Not set'}`\n"
            f"**Auth Token:** `{'Set' if auth_token else 'Not set'}`"
        )
        await ctx.send(msg)

# If not using LokiLogger, these commands allow manual configuration
    @reactionstats.command()
    @commands.is_owner()
    async def seturl(self, ctx, url: str):
        """Set Loki endpoint URL (if not using LokiLogger)"""
        if not url.startswith("http"):
            return await ctx.send("❌ Invalid URL format")
        await self.config.loki_url.set(url)
        await ctx.tick()
        
    @reactionstats.command()
    @commands.is_owner()
    async def settoken(self, ctx, token: str):
        """Set authentication token (if not using LokiLogger)"""
        await self.config.auth_token.set(token)
        await ctx.send("Token set!")