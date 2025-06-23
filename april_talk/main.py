import asyncio
import aiohttp
import discord
import tempfile
import os
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import pagify
from redbot.core.utils.menus import menu, DEFAULT_CONTROLS

class AprilAI(commands.Cog):
    """Unified AI assistant with text and voice via Redbot’s Audio cog"""

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

    def cog_unload(self):
        # close aiohttp session
        self.bot.loop.create_task(self.session.close())
        for task in self.active_tts_tasks.values():
            task.cancel()

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, input: str):
        """Interact with April AI – text or voice based on context."""
        if input.strip().lower() == "join":
            return await self.join_voice(ctx)
        await self.process_query(ctx, input)

    async def join_voice(self, ctx):
        """Delegate to Redbot’s Audio cog to join the channel."""
        if not ctx.guild:
            return await ctx.send("❌ This command only works in servers!")
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You need to be in a voice channel!")

        audio_cog = self.bot.get_cog("Audio")
        if audio_cog:
            # Call Redbot Audio's summon
            await ctx.invoke(self.bot.get_command("summon"))
            return
        await ctx.send("❌ Audio cog not found. Install Redbot Audio to use voice features.")

    async def process_query(self, ctx, input_text):
        """Handle text vs. voice response logic."""
        use_voice = await self.config.tts_enabled() and ctx.guild and ctx.guild.voice_client
        async with ctx.typing():
            try:
                resp = await self.query_deepseek(ctx.author.id, input_text, ctx)
                if not (use_voice and not await self.config.text_response_when_voice()):
                    await self.send_text_response(ctx, resp)
                if use_voice:
                    await self.speak_response(ctx.guild, resp)
            except Exception as e:
                await ctx.send(f"❌ Error: {e}")

    async def speak_response(self, guild, text: str):
        """Schedule TTS playback via temporary file."""
        tts_key = await self.config.tts_key()
        voice_id = await self.config.voice_id()
        vc = guild.voice_client
        if not vc or not tts_key:
            return
        if guild.id in self.active_tts_tasks:
            self.active_tts_tasks[guild.id].cancel()
        self.active_tts_tasks[guild.id] = asyncio.create_task(
            self._play_tts(vc, tts_key, voice_id, text)
        )

    async def _play_tts(self, voice_client, tts_key, voice_id, text: str):
        """Fetch MP3 from ElevenLabs, save to temp file, and play."""
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        for chunk in chunks:
            if not chunk.strip():
                continue
            headers = {"xi-api-key": tts_key, "Content-Type": "application/json"}
            payload = {"text": chunk, "voice_settings": {"stability":0.5,"similarity_boost":0.8}}
            try:
                async with self.session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    json=payload, headers=headers, timeout=30
                ) as resp:
                    if resp.status != 200:
                        print("TTS API error:", await resp.text())
                        continue
                    data = await resp.read()
            except Exception as e:
                print("TTS fetch error:", e)
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                tf.write(data)
                path = tf.name

            source = discord.FFmpegPCMAudio(path, before_options="-loglevel warning")
            voice_client.play(source)
            while voice_client.is_playing():
                await asyncio.sleep(0.1)
            try:
                os.unlink(path)
            except:
                pass

    @commands.group(name="deepseek", aliases=["ds"])
    @commands.is_owner()
    async def deepseek(self, ctx):
        """Configure DeepSeek AI settings"""
        pass

    @deepseek.command()
    async def apikey(self, ctx, key: str):
        await self.config.deepseek_key.set(key); await ctx.tick()
    @deepseek.command()
    async def prompt(self, ctx, *, system_prompt: str):
        await self.config.system_prompt.set(system_prompt); await ctx.tick()
    @deepseek.command()
    async def model(self, ctx, model_name: str):
        await self.config.model.set(model_name.lower()); await ctx.tick()
    @deepseek.command()
    async def temperature(self, ctx, value: float):
        if 0.0 <= value <= 1.0: await self.config.temperature.set(value); await ctx.tick()
        else: await ctx.send("❌ Temp must be 0.0–1.0")
    @deepseek.command()
    async def tokens(self, ctx, max_tokens: int):
        if 100 <= max_tokens <= 4096: await self.config.max_tokens.set(max_tokens); await ctx.tick()
        else: await ctx.send("❌ Tokens must be 100–4096")
    @deepseek.command()
    async def settings(self, ctx):
        cfg = await self.config.all()
        e = discord.Embed(title="April AI Settings", color=await ctx.embed_color())
        e.add_field("DeepSeek Key", f"`...{cfg['deepseek_key'][-4:]}`" if cfg["deepseek_key"] else "❌ Not set")
        e.add_field("TTS Key", f"`...{cfg['tts_key'][-4:]}`" if cfg["tts_key"] else "❌ Not set")
        e.add_field("Voice ID", cfg["voice_id"])  
        e.add_field("TTS Enabled", "✅" if cfg["tts_enabled"] else "❌")
        e.add_field("Text w/Voice", "✅" if cfg["text_response_when_voice"] else "❌")
        e.add_field("Model", cfg["model"])
        e.add_field("Temperature", str(cfg["temperature"]))
        e.add_field("Max Tokens", str(cfg["max_tokens"]))
        e.add_field("System Prompt", f"```{cfg['system_prompt']}```", inline=False)
        await ctx.send(embed=e)

    async def query_deepseek(self, user_id: int, question: str, ctx):
        key = await self.config.deepseek_key()
        if not key:
            raise Exception("DeepSeek key not set.")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": [
                {"role":"system","content":await self.config.system_prompt()},
                {"role":"user","content":f"[User:{ctx.author.display_name}] {question}"}
            ],
            "temperature": await self.config.temperature(),
            "max_tokens": await self.config.max_tokens(),
            "user": str(user_id)
        }
        async with self.session.post(
            "https://api.deepseek.com/v1/chat/completions",
            json=payload, headers=headers, timeout=30
        ) as r:
            data = await r.json()
            if r.status != 200:
                err = data.get("error",{}).get("message","Unknown")
                raise Exception(f"API {r.status}: {err}")
            return data["choices"][0]["message"]["content"].strip()

    async def send_text_response(self, ctx, response: str):
        if len(response) > 1900:
            pages = list(pagify(response, delims=["\n"," "], priority=True, page_length=1500))
            await menu(ctx, pages, DEFAULT_CONTROLS)
        else:
            await ctx.send(response)

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        # auto-disconnect if alone
        if member.bot:
            return
        vc = member.guild.voice_client
        if vc and len(vc.channel.members) == 1:
            await vc.disconnect()
            self.active_tts_tasks.pop(member.guild.id, None)

async def setup(bot):
    await bot.add_cog(AprilAI(bot))