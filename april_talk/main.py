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
    """Unified AI assistant with text and voice via direct FFmpegPCMAudio playback"""

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
            return await self.leave_voice(ctx)
        tllogger.debug(f"Received april command: {input} from {ctx.author}")
        await self.process_query(ctx, input)

    async def join_voice(self, ctx):
        """Join the user's voice channel"""
        tllogger.debug("Attempting to join voice channel.")
        if not ctx.guild or not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a server voice channel.")
        channel = ctx.author.voice.channel
        try:
            vc = ctx.guild.voice_client
            if vc and vc.channel != channel:
                await vc.move_to(channel)
            elif not vc:
                vc = await channel.connect()
            await ctx.send(f"🔊 Joined {channel.name}")
        except Exception as e:
            tllogger.exception("Failed to join voice")
            await ctx.send(f"❌ Failed to join voice: {e}")

    async def leave_voice(self, ctx):
        """Leave the current voice channel"""
        tllogger.debug("Attempting to leave voice channel.")
        vc = ctx.guild.voice_client
        if vc:
            try:
                await vc.disconnect()
                await ctx.send("👋 Disconnected from voice.")
            except Exception as e:
                tllogger.exception("Failed to disconnect voice")
                await ctx.send(f"❌ Error disconnecting: {e}")
        else:
            await ctx.send("❌ Not connected to voice.")

    async def process_query(self, ctx, input_text):
        """Process text input and optionally speak response"""
        use_voice = await self.config.tts_enabled() and ctx.guild and ctx.guild.voice_client
        tllogger.debug(f"process_query use_voice={use_voice}")
        async with ctx.typing():
            try:
                response = await self.query_deepseek(ctx.author.id, input_text, ctx)
                if not (use_voice and not await self.config.text_response_when_voice()):
                    await self.send_text_response(ctx, response)
                if use_voice:
                    await self.speak_response(ctx, response)
            except Exception as e:
                tllogger.exception("process_query error")
                await ctx.send(f"❌ Error: {e}")

    async def speak_response(self, ctx, text: str):
        """Kick off TTS playback task"""
        tts_key = await self.config.tts_key()
        vc = ctx.guild.voice_client if ctx.guild else None
        if not tts_key or not vc:
            tllogger.warning("Skipping TTS, missing key or not in VC.")
            return
        if ctx.guild.id in self.active_tts_tasks:
            self.active_tts_tasks[ctx.guild.id].cancel()
        task = asyncio.create_task(self._play_tts(vc, tts_key, await self.config.voice_id(), text))
        self.active_tts_tasks[ctx.guild.id] = task

    async def _play_tts(self, voice_client, tts_key, voice_id, text: str):
        """Fetch MP3, save to temp file, then play via FFmpegPCMAudio"""
        tllogger.debug("_play_tts start")
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        for idx, chunk in enumerate(chunks, 1):
            if not chunk.strip():
                continue
            tllogger.debug(f"Fetching TTS chunk {idx}/{len(chunks)}")
            try:
                async with self.session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    json={"text": chunk, "voice_settings": {"stability":0.5, "similarity_boost":0.8}},
                    headers={"xi-api-key": tts_key}, timeout=30
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
            except Exception:
                tllogger.exception("TTS fetch failed")
                continue

            # Write MP3 to temp file
            path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tf.write(data)
                    path = tf.name
                tllogger.debug(f"Saved TTS to {path}")

                # Play via FFmpegPCMAudio
                source = discord.FFmpegPCMAudio(path)
                voice_client.play(source)
                tllogger.debug("Playing TTS via FFmpegPCMAudio.")
                while voice_client.is_playing():
                    await asyncio.sleep(0.1)
                tllogger.debug("TTS playback complete.")
            except Exception:
                tllogger.exception("Error during TTS playback")
            finally:
                if path:
                    try:
                        os.unlink(path)
                        tllogger.debug(f"Deleted temp file: {path}")
                    except:
                        pass

    @commands.group(name="deepseek", aliases=["ds"])
    @commands.is_owner()
    async def deepseek(self, ctx):
        """Configure DeepSeek settings"""
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
    async def temperature(self, ctx, val: float):
        if 0.0 <= val <= 1.0: await self.config.temperature.set(val); await ctx.tick()
        else: await ctx.send("❌ Temp must be between 0.0 and 1.0")

    @deepseek.command()
    async def tokens(self, ctx, num: int):
        if 100 <= num <= 4096: await self.config.max_tokens.set(num); await ctx.tick()
        else: await ctx.send("❌ Tokens must be between 100 and 4096")

    @deepseek.command()
    async def settings(self, ctx):
        cfg = await self.config.all()
        e = discord.Embed(title="AprilAI Settings", color=await ctx.embed_color())
        e.add_field(name="DeepSeek Key", value=(f"...{cfg['deepseek_key'][-4:]}" if cfg['deepseek_key'] else "❌ Not set"), inline=True)
        e.add_field(name="TTS Key", value=(f"...{cfg['tts_key'][-4:]}" if cfg['tts_key'] else "❌ Not set"), inline=True)
        e.add_field(name="Voice ID", value=cfg['voice_id'], inline=True)
        e.add_field(name="TTS Enabled", value=("✅" if cfg['tts_enabled'] else "❌"), inline=True)
        e.add_field(name="Text w/Voice", value=("✅" if cfg['text_response_when_voice'] else "❌"), inline=True)
        e.add_field(name="Model", value=cfg['model'], inline=True)
        e.add_field(name="Temp", value=str(cfg['temperature']), inline=True)
        e.add_field(name="Max Tokens", value=str(cfg['max_tokens']), inline=True)
        e.add_field(name="Prompt", value=f"```{cfg['system_prompt']}```", inline=False)
        await ctx.send(embed=e)

    async def query_deepseek(self, user_id: int, question: str, ctx):
        key = await self.config.deepseek_key()
        if not key: raise Exception("DeepSeek key missing.")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": await self.config.model(),
                   "messages":[{"role":"system","content":await self.config.system_prompt()},
                                {"role":"user","content":f"[User:{ctx.author.display_name}] {question}"}],
                   "temperature": await self.config.temperature(),
                   "max_tokens": await self.config.max_tokens(),
                   "user": str(user_id)}
        async with self.session.post(
            "https://api.deepseek.com/v1/chat/completions", json=payload,
            headers=headers, timeout=30
        ) as r:
            data = await r.json()
            if r.status != 200:
                err = data.get("error",{}).get("message","?")
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
        if member.bot: return
        vc = member.guild.voice_client
        if vc and len(vc.channel.members) == 1:
            await vc.disconnect()
            self.active_tts_tasks.pop(member.guild.id, None)

async def setup(bot):
    await bot.add_cog(AprilAI(bot))
