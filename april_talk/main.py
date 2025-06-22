import asyncio
import aiohttp
import json
import discord
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import pagify

class AprilAI(commands.Cog):
    """Unified AI assistant with text and voice capabilities"""
    
    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1398462)
        self.session = aiohttp.ClientSession()
        
        default_global = {
            "deepseek_key": "",
            "tts_key": "",
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2048,
            "system_prompt": "You are April, a helpful AI assistant. Respond cryptically when appropriate.",
            "tts_enabled": True,
            "text_response_when_voice": True
        }
        self.config.register_global(**default_global)
        self.voice_clients = {}
        self.active_tts_tasks = {}

    def cog_unload(self):
        self.bot.loop.create_task(self.session.close())
        for vc in self.voice_clients.values():
            self.bot.loop.create_task(vc.disconnect())

    @commands.group(name="april", invoke_without_command=True)
    async def april(self, ctx, *, input: str):
        """Interact with April AI - text or voice based on context"""
        # Handle "join" command
        if input.strip().lower() == "join":
            return await self.join_voice(ctx)
        
        # Handle regular queries
        await self.process_query(ctx, input)

    async def process_query(self, ctx, input_text):
        """Process user input with dual text/voice response"""
        # Check if we should use voice response
        use_voice = await self.config.tts_enabled() and ctx.guild and ctx.guild.voice_client
        
        async with ctx.typing():
            try:
                # Get AI response
                response = await self.query_deepseek(
                    user_id=ctx.author.id,
                    question=input_text,
                    ctx=ctx
                )
                
                # Send text response first
                await self.send_text_response(ctx, response)
                
                # Add voice response if in voice channel
                if use_voice:
                    await self.speak_response(ctx.guild, response)
                    
            except Exception as e:
                await ctx.send(f"❌ Error: {str(e)}")

    async def join_voice(self, ctx):
        """Join the user's voice channel"""
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You need to be in a voice channel!")
        
        # Join voice channel
        voice_channel = ctx.author.voice.channel
        try:
            vc = await voice_channel.connect()
            self.voice_clients[ctx.guild.id] = vc
            await ctx.send(f"🔊 Joined {voice_channel.name}")
        except discord.ClientException:
            await ctx.send("✅ Already in your voice channel")

    async def speak_response(self, guild, text: str):
        """Convert text to speech using ElevenLabs"""
        tts_key = await self.config.tts_key()
        voice_id = await self.config.voice_id()
        vc = self.voice_clients.get(guild.id)
        
        if not vc or not tts_key:
            return
        
        # Cancel any ongoing TTS task
        if guild.id in self.active_tts_tasks:
            self.active_tts_tasks[guild.id].cancel()
        
        # Create new TTS task
        self.active_tts_tasks[guild.id] = asyncio.create_task(
            self.play_tts(vc, tts_key, voice_id, text)
        )

    async def play_tts(self, vc, tts_key, voice_id, text: str):
        """Play TTS audio in voice channel"""
        # Split long text into chunks
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            headers = {
                "xi-api-key": tts_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": chunk,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                }
            }
            
            # Get TTS audio
            async with self.session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    # Create in-memory audio source
                    audio_data = await response.read()
                    source = discord.FFmpegPCMAudio(
                        source="pipe:0", 
                        before_options="-f mp3",
                        options="-loglevel warning",
                        pipe=True
                    )
                    
                    # Play audio
                    vc.play(source)
                    while vc.is_playing():
                        await asyncio.sleep(0.1)

    @april.command()
    @commands.is_owner()
    async def setkey(self, ctx, key: str):
        """Set DeepSeek API key"""
        await self.config.deepseek_key.set(key)
        await ctx.send("🔑 DeepSeek API key set!")

    @april.command()
    @commands.is_owner()
    async def setttskey(self, ctx, key: str):
        """Set ElevenLabs API key"""
        await self.config.tts_key.set(key)
        await ctx.send("🔊 ElevenLabs API key set!")

    @april.command()
    @commands.is_owner()
    async def setvoice(self, ctx, voice_id: str):
        """Set ElevenLabs voice ID"""
        await self.config.voice_id.set(voice_id)
        await ctx.send(f"🗣️ Voice ID set to `{voice_id}`")

    @april.command()
    @commands.is_owner()
    async def togglevoice(self, ctx):
        """Toggle voice responses"""
        current = await self.config.tts_enabled()
        await self.config.tts_enabled.set(not current)
        status = "ENABLED" if not current else "DISABLED"
        await ctx.send(f"🔊 Voice responses {status}")

    @april.command()
    async def join(self, ctx):
        """Join your voice channel (alternative to 'april join')"""
        await self.join_voice(ctx)

    @april.command()
    async def leave(self, ctx):
        """Leave voice channel"""
        if ctx.guild.id in self.voice_clients:
            await self.voice_clients[ctx.guild.id].disconnect()
            del self.voice_clients[ctx.guild.id]
            await ctx.send("👋 Left voice channel")
        else:
            await ctx.send("❌ Not in a voice channel")

    async def query_deepseek(self, user_id: int, question: str, ctx: commands.Context):
        """Query DeepSeek API"""
        deepseek_key = await self.config.deepseek_key()
        model = await self.config.model()
        system_prompt = await self.config.system_prompt()
        
        if not deepseek_key:
            raise Exception("DeepSeek API key not set")
        
        headers = {
            "Authorization": f"Bearer {deepseek_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"[User: {ctx.author.display_name}] {question}"}
            ],
            "temperature": await self.config.temperature(),
            "max_tokens": await self.config.max_tokens(),
            "user": str(user_id)
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

    async def send_text_response(self, ctx, response: str):
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

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        """Auto-disconnect when alone in voice channel"""
        if member.bot:
            return
        
        guild_id = member.guild.id
        if guild_id in self.voice_clients:
            vc = self.voice_clients[guild_id]
            if len(vc.channel.members) == 1:  # Only bot remains
                await vc.disconnect()
                del self.voice_clients[guild_id]
                if guild_id in self.active_tts_tasks:
                    self.active_tts_tasks[guild_id].cancel()
                    del self.active_tts_tasks[guild_id]

async def setup(bot):
    await bot.add_cog(AprilAI(bot))