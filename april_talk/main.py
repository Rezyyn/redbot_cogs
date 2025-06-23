import asyncio
import aiohttp
import discord
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import pagify
from redbot.core.utils.menus import close_menu, menu, DEFAULT_CONTROLS

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
        self.active_tts_tasks = {}
        self.voice_lock = asyncio.Lock()

    def cog_unload(self):
        self.bot.loop.create_task(self.session.close())
        for task in self.active_tts_tasks.values():
            task.cancel()

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
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
                
                # Conditionally send text response
                if not (use_voice and not await self.config.text_response_when_voice()):
                    await self.send_text_response(ctx, response)
                
                # Add voice response if in voice channel
                if use_voice and ctx.guild.voice_client:
                    await self.speak_response(ctx.guild, response)
                    
            except Exception as e:
                await ctx.send(f"❌ Error: {str(e)}")

    async def join_voice(self, ctx):
        """Join the user's voice channel"""
        if not ctx.guild:
            return await ctx.send("❌ This command only works in servers!")
        
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You need to be in a voice channel!")
        
        # Join voice channel using Red's standard method
        try:
            # Get existing voice client or create new one
            voice_client = ctx.guild.voice_client
            if not voice_client:
                voice_client = await ctx.author.voice.channel.connect()
                await ctx.send(f"🔊 Joined {ctx.author.voice.channel.name}")
            elif voice_client.channel != ctx.author.voice.channel:
                await voice_client.move_to(ctx.author.voice.channel)
                await ctx.send(f"🔊 Moved to {ctx.author.voice.channel.name}")
            else:
                await ctx.send("✅ Already in your voice channel")
        except Exception as e:
            await ctx.send(f"❌ Failed to join voice: {str(e)}")

    async def speak_response(self, guild, text: str):
        """Convert text to speech using ElevenLabs"""
        tts_key = await self.config.tts_key()
        voice_id = await self.config.voice_id()
        voice_client = guild.voice_client
        
        if not voice_client or not tts_key:
            return
        
        # Cancel any ongoing TTS task
        if guild.id in self.active_tts_tasks:
            try:
                self.active_tts_tasks[guild.id].cancel()
            except:
                pass
        
        # Create new TTS task
        self.active_tts_tasks[guild.id] = asyncio.create_task(
            self.play_tts(voice_client, tts_key, voice_id, text)
        )

   async def play_tts(self, voice_client, tts_key, voice_id, text: str):
    """Play TTS audio in voice channel from ElevenLabs (MP3)"""
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
        try:
            async with self.session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                json=payload,
                headers=headers,
                timeout=30
            ) as response:
                if response.status == 200:
                    audio_data = await response.read()
                    audio_stream = io.BytesIO(audio_data)
                    # Play audio from in-memory stream
                    source = discord.FFmpegPCMAudio(
                        audio_stream,
                        pipe=True,
                        before_options="-f mp3",
                        options="-loglevel warning"
                    )
                    if not voice_client.is_playing():
                        voice_client.play(source)
                        while voice_client.is_playing():
                            await asyncio.sleep(0.1)
                else:
                    error = await response.text()
                    print(f"ElevenLabs API error: {error}")
        except Exception as e:
            print(f"TTS playback error: {str(e)}")
       
    @commands.group(name="deepseek", aliases=["ds"])
    @commands.is_owner()
    async def deepseek(self, ctx):
        """Configure DeepSeek AI settings"""
        pass

    @deepseek.command()
    async def apikey(self, ctx, key: str):
        """Set DeepSeek API key"""
        await self.config.deepseek_key.set(key)
        await ctx.tick()

    @deepseek.command()
    async def prompt(self, ctx, *, system_prompt: str):
        """Set custom system prompt"""
        await self.config.system_prompt.set(system_prompt)
        await ctx.tick()

    @deepseek.command()
    async def model(self, ctx, model_name: str):
        """Set model (deepseek-chat/deepseek-coder)"""
        await self.config.model.set(model_name.lower())
        await ctx.tick()

    @deepseek.command()
    async def temperature(self, ctx, value: float):
        """Set temperature (0.0 to 1.0)"""
        if 0.0 <= value <= 1.0:
            await self.config.temperature.set(value)
            await ctx.tick()
        else:
            await ctx.send("❌ Temperature must be between 0.0 and 1.0")

    @deepseek.command()
    async def tokens(self, ctx, max_tokens: int):
        """Set maximum tokens (100 to 4096)"""
        if 100 <= max_tokens <= 4096:
            await self.config.max_tokens.set(max_tokens)
            await ctx.tick()
        else:
            await ctx.send("❌ Max tokens must be between 100 and 4096")

    @deepseek.command()
    async def settings(self, ctx):
        """Show current configuration"""
        config = await self.config.all()
        embed = discord.Embed(title="April AI Configuration", color=await ctx.embed_color())
        
        # API keys (masked)
        embed.add_field(name="DeepSeek Key", value=f"`...{config['deepseek_key'][-4:]}`" if config['deepseek_key'] else "❌ Not set")
        embed.add_field(name="TTS Key", value=f"`...{config['tts_key'][-4:]}`" if config['tts_key'] else "❌ Not set")
        
        # Voice settings
        embed.add_field(name="Voice ID", value=config['voice_id'])
        embed.add_field(name="TTS Enabled", value=("✅" if config['tts_enabled'] else "❌"))
        embed.add_field(name="Text w/Voice", value=("✅" if config['text_response_when_voice'] else "❌"))
        
        # AI settings
        embed.add_field(name="Model", value=config['model'])
        embed.add_field(name="Temperature", value=config['temperature'])
        embed.add_field(name="Max Tokens", value=config['max_tokens'])
        embed.add_field(name="System Prompt", value=f"```{config['system_prompt'][:1000]}```", inline=False)
        
        await ctx.send(embed=embed)

    @april.command()
    @commands.is_owner()
    async def ttskey(self, ctx, key: str):
        """Set ElevenLabs API key"""
        await self.config.tts_key.set(key)
        await ctx.tick()

    @april.command()
    @commands.is_owner()
    async def voiceid(self, ctx, voice_id: str):
        """Set ElevenLabs voice ID"""
        await self.config.voice_id.set(voice_id)
        await ctx.tick()

    @april.command()
    @commands.is_owner()
    async def togglevoice(self, ctx):
        """Toggle voice responses"""
        current = await self.config.tts_enabled()
        await self.config.tts_enabled.set(not current)
        status = "✅ Enabled" if not current else "❌ Disabled"
        await ctx.send(f"🔊 Voice responses {status}")

    @april.command()
    @commands.is_owner()
    async def toggletxt(self, ctx):
        """Toggle text response when using voice"""
        current = await self.config.text_response_when_voice()
        await self.config.text_response_when_voice.set(not current)
        status = "✅ Enabled" if not current else "❌ Disabled"
        await ctx.send(f"📝 Text response with voice {status}")

    @april.command()
    async def join(self, ctx):
        """Join your voice channel"""
        await self.join_voice(ctx)

    @april.command()
    async def leave(self, ctx):
        """Leave voice channel"""
        if not ctx.guild:
            return await ctx.send("❌ This command only works in servers!")
            
        if ctx.guild.voice_client:
            await ctx.guild.voice_client.disconnect()
            # Cancel any active TTS task
            if ctx.guild.id in self.active_tts_tasks:
                try:
                    self.active_tts_tasks[ctx.guild.id].cancel()
                except:
                    pass
                del self.active_tts_tasks[ctx.guild.id]
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
        # For very long responses, use pagination
        if len(response) > 1900:
            pages = list(pagify(response, delims=["\n", " "], priority=True, page_length=1500))
            await menu(ctx, pages, DEFAULT_CONTROLS)
        else:
            await ctx.send(response)

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        """Auto-disconnect when alone in voice channel"""
        if member.bot:
            return
        
        guild = member.guild
        voice_client = guild.voice_client
        
        if voice_client and voice_client.channel:
            # Check if only the bot is left in the channel
            if len(voice_client.channel.members) == 1:
                await voice_client.disconnect()
                # Cancel any active TTS task
                if guild.id in self.active_tts_tasks:
                    try:
                        self.active_tts_tasks[guild.id].cancel()
                    except:
                        pass
                    del self.active_tts_tasks[guild.id]

async def setup(bot):
    await bot.add_cog(AprilAI(bot))