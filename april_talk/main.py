import asyncio
import aiohttp
import discord
import tempfile
import os
import logging
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.utils.chat_formatting import pagify
from redbot.core.utils.menus import menu, DEFAULT_CONTROLS

# Initialize logger for this Cog
tllogger = logging.getLogger("red.aprilai")
tllogger.setLevel(logging.DEBUG)

class AprilAI(commands.Cog):
    """Unified AI assistant with text and voice via Redbot’s Audio cog"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1398462)
        self.session = aiohttp.ClientSession()
        self.active_tts_tasks = {}

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

    def cog_unload(self):
        tllogger.debug("Unloading AprilAI cog, closing session and cancelling tasks.")
        self.bot.loop.create_task(self.session.close())
        for task in self.active_tts_tasks.values():
            task.cancel()

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, input: str):
        """Interact with April AI – text or voice based on context."""
        cmd = input.strip().lower()
        if cmd == "join":
            return await self.join_voice(ctx)
        if cmd == "leave":
            return await self.april_leave(ctx)
        tllogger.debug(f"Received april command with input: {input} by {ctx.author}")
        await self.process_query(ctx, input)

    async def join_voice(self, ctx):
        """Delegate to Redbot’s Audio cog to join the channel."""
        tllogger.debug("Attempting to join voice channel.")
        if not ctx.guild:
            return await ctx.send("❌ This command only works in servers!")
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You need to be in a voice channel!")

        audio_cog = self.bot.get_cog("Audio")
        if audio_cog:
            channel = ctx.author.voice.channel
            tllogger.debug(f"Invoking summon for channel: {channel.name}")
            await ctx.invoke(self.bot.get_command("summon"))
            await ctx.send(f"🔊 Joined {channel.name} (via summon)")
        else:
            tllogger.error("Audio cog not found when attempting join.")
            await ctx.send("❌ Audio cog not found. Install Redbot Audio to use voice features.")

    async def april_leave(self, ctx):
        """Disconnect from current voice channel via direct call."""
        tllogger.debug("Attempting to disconnect from voice channel.")
        vc = ctx.guild.voice_client
        if vc:
            await vc.disconnect()
            tllogger.debug("Disconnected from voice channel.")
            await ctx.send("👋 Disconnected from voice channel.")
        else:
            tllogger.warning("No voice client to disconnect.")
            await ctx.send("❌ Not connected to any voice channel.")

    async def process_query(self, ctx, input_text):
        """Handle text vs. voice response logic."""
        use_voice = await self.config.tts_enabled() and ctx.guild and ctx.guild.voice_client
        tllogger.debug(f"process_query: use_voice={use_voice}")
        async with ctx.typing():
            try:
                resp = await self.query_deepseek(ctx.author.id, input_text, ctx)
                tllogger.debug(f"AI response length: {len(resp)}")
                if not (use_voice and not await self.config.text_response_when_voice()):
                    await self.send_text_response(ctx, resp)
                if use_voice:
                    await self.speak_response(ctx, resp)
            except Exception as e:
                tllogger.exception("Error in process_query")
                await ctx.send(f"❌ Error: {e}")

    async def speak_response(self, ctx, text: str):
        """Schedule TTS playback via Redbot Audio cog."""
        guild = ctx.guild
        tllogger.debug(f"Scheduling TTS for guild: {guild.id}")
        tts_key = await self.config.tts_key()
        if not tts_key:
            tllogger.warning("No TTS key set; skipping TTS.")
            return
        if guild.id in self.active_tts_tasks:
            self.active_tts_tasks[guild.id].cancel()
        task = asyncio.create_task(self._play_tts(ctx, tts_key, await self.config.voice_id(), text))
        self.active_tts_tasks[guild.id] = task

    async def _play_tts(self, ctx, tts_key, voice_id, text: str):
        """Fetch MP3 from ElevenLabs, save to temp file, then invoke Audio.play."""
        tllogger.debug("_play_tts started.")
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        play_cmd = self.bot.get_command("play")
        for idx, chunk in enumerate(chunks, 1):
            if not chunk.strip():
                continue
            tllogger.debug(f"TTS chunk {idx}/{len(chunks)}")
            headers = {"xi-api-key": tts_key, "Content-Type": "application/json"}
            payload = {"text": chunk, "voice_settings": {"stability":0.5,"similarity_boost":0.8}}
            try:
                async with self.session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    json=payload, headers=headers, timeout=30
                ) as resp:
                    if resp.status != 200:
                        errtxt = await resp.text()
                        tllogger.error(f"TTS API error {resp.status}: {errtxt}")
                        continue
                    data = await resp.read()
                    tllogger.debug(f"Received {len(data)} bytes from TTS API.")
            except Exception:
                tllogger.exception("Exception during TTS fetch.")
                continue

            path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tf.write(data)
                    path = tf.name
                tllogger.debug(f"Wrote TTS to temp file: {path}")
                tllogger.debug(f"Invoking play command for file: {path}")
                                # Direct playback of the temp MP3 file
                tllogger.debug(f"Invoking Audio.play for file: {path}")
                await ctx.invoke(play_cmd, song=path)
                tllogger.debug("Audio.play command invoked successfully.")
            except Exception:
                tllogger.exception("Exception during Audio.play invocation.")
            finally:
                if path:
                    try:
                        os.unlink(path)
                        tllogger.debug(f"Deleted temp file: {path}")
                    except Exception:
                        tllogger.exception("Failed to delete temp file.")

    @commands.group(name="deepseek", aliases=["ds"])
    @commands.is_owner()
    async def deepseek(self, ctx):
        """Configure DeepSeek AI settings"""
        pass

    @deepseek.command()
    async def apikey(self, ctx, key: str):
        await self.config.deepseek_key.set(key)
        await ctx.tick()

    @deepseek.command()
    async def prompt(self, ctx, *, system_prompt: str):
        await self.config.system_prompt.set(system_prompt)
        await ctx.tick()

    @deepseek.command()
    async def model(self, ctx, model_name: str):
        await self.config.model.set(model_name.lower())
        await ctx.tick()

    @deepseek.command()
    async def temperature(self, ctx, value: float):
        if 0.0 <= value <= 1.0:
            await self.config.temperature.set(value)
            await ctx.tick()
        else:
            await ctx.send("❌ Temp must be 0.0–1.0")

    @deepseek.command()
    async def tokens(self, ctx, max_tokens: int):
        if 100 <= max_tokens <= 4096:
            await self.config.max_tokens.set(max_tokens)
            await ctx.tick()
        else:
            await ctx.send("❌ Tokens must be 100–4096")

    @deepseek.command()
    async def settings(self, ctx):
        cfg = await self.config.all()
        e = discord.Embed(title="April AI Settings", color=await ctx.embed_color())
        e.add_field(name="DeepSeek Key", value=(f"...{cfg['deepseek_key'][-4:]}" if cfg['deepseek_key'] else "❌ Not set"), inline=True)
        e.add_field(name="TTS Key", value=(f"...{cfg['tts_key'][-4:]}" if cfg['tts_key'] else "❌ Not set"), inline=True)
        e.add_field(name="Voice ID", value=cfg['voice_id'], inline=True)
        e.add_field(name="TTS Enabled", value=("✅" if cfg['tts_enabled'] else "❌"), inline=True)
        e.add_field(name="Text w/Voice", value=("✅" if cfg['text_response_when_voice'] else "❌"), inline=True)
        e.add_field(name="Model", value=cfg['model'], inline=True)
        e.add_field(name="Temperature", value=str(cfg['temperature']), inline=True)
        e.add_field(name="Max Tokens", value=str(cfg['max_tokens']), inline=True)
        e.add_field(name="System Prompt", value=f"```{cfg['system_prompt']}```", inline=False)
        await ctx.send(embed=e)

    async def query_deepseek(self, user_id: int, question: str, ctx):
        key = await self.config.deepseek_key()
        if not key:
            raise Exception("DeepSeek key not set.")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": [
                {"role": "system", "content": await self.config.system_prompt()},
                {"role": "user", "content": f"[User:{ctx.author.display_name}] {question}"}
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
                err = data.get("error", {}).get("message", "Unknown")
                raise Exception(f"API {r.status}: {err}")
            return data["choices"][0]["message"]["content"].strip()

    async def send_text_response(self, ctx, response: str):
        if len(response) > 1900:
            pages = list(pagify(response, delims=["\n", " "], priority=True, page_length=1500))
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
