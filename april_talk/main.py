import aiohttp
import discord
from redbot.core import commands, Config
from redbot.core.bot import Red

class AprilAI(commands.Cog):
    """DeepSeek AI integration for Red-DiscordBot"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1398462)
        self.session = aiohttp.ClientSession()
        
        default_global = {
            "api_key": "",
            "system_prompt": "You are a helpful assistant integrated with Discord.",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2048
        }
        self.config.register_global(**default_global)

    def cog_unload(self):
        self.bot.loop.create_task(self.session.close())

    @commands.group()
    @commands.is_owner()
    async def deepseek(self, ctx):
        """Configure DeepSeek settings"""
        pass

    @deepseek.command()
    async def apikey(self, ctx, api_key: str):
        """Set DeepSeek API key"""
        await self.config.api_key.set(api_key)
        await ctx.send("API key set successfully!")

    @deepseek.command()
    async def prompt(self, ctx, *, system_prompt: str):
        """Set custom system prompt"""
        await self.config.system_prompt.set(system_prompt)
        await ctx.send("System prompt updated!")

    @deepseek.command()
    async def model(self, ctx, model_name: str):
        """Set model (deepseek-chat/deepseek-coder)"""
        await self.config.model.set(model_name.lower())
        await ctx.send(f"Model set to {model_name}")

    @deepseek.command()
    async def settings(self, ctx):
        """Show current configuration"""
        config = await self.config.all()
        embed = discord.Embed(title="April AI Configuration", color=await ctx.embed_color())
        embed.add_field(name="Model", value=config["model"])
        embed.add_field(name="System Prompt", value=f'```{config["system_prompt"][:1000]}```', inline=False)
        embed.add_field(name="Temperature", value=config["temperature"])
        embed.add_field(name="Max Tokens", value=config["max_tokens"])
        await ctx.send(embed=embed)

    @commands.command()
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, question: str):
        """Ask April AI a question"""
        api_key = await self.config.api_key()
        if not api_key:
            return await ctx.send("API key not set! Bot owner must configure with `[p]deepseek apikey`")
        
        async with ctx.typing():
            try:
                response = await self.query_deepseek(
                    user_id=ctx.author.id,
                    question=question,
                    ctx=ctx
                )
                await self.send_response(ctx, response)
            except Exception as e:
                await ctx.send(f"❌ Error: {str(e)}")

    async def query_deepseek(self, user_id: int, question: str, ctx: commands.Context):
        config = await self.config.all()
        
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": config["system_prompt"]},
                {"role": "user", "content": f"[User: {ctx.author.display_name}] {question}"}
            ],
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "user": str(user_id)
        }
        
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        async with self.session.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        ) as response:
            data = await response.json()
            
            if response.status != 200:
                error = data.get("error", {}).get("message", "Unknown API error")
                raise Exception(f"API Error {response.status}: {error}")
            
            return data["choices"][0]["message"]["content"].strip()

    async def send_response(self, ctx, response: str):
        """Send response with smart formatting"""
        # Clean potential tokens
        response = response.replace("`", "'")
        
        # Send as file if over 1900 characters
        if len(response) > 1900:
            await ctx.send(file=discord.File(
                filename="april_response.txt",
                fp=response.encode('utf-8')
            ))
        else:
            await ctx.send(response)

async def setup(bot):
    await bot.add_cog(AprilAI(bot))
