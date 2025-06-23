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

# Logger
tllogger = logging.getLogger("red.aprilai")
tllogger.setLevel(logging.DEBUG)

class AprilAI(commands.Cog):
    """AI assistant with text and voice via Red's audio system"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1398462)
        self.session = aiohttp.ClientSession()
        self.config.register_global(
            deepseek_key="",
            tts_key="",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="You are April...",
            tts_enabled=True,
            text_response_when_voice=True
        )

    def cog_unload(self):
        tllogger.debug("Unloading AprilAI, closing session.")
        self.bot.loop.create_task(self.session.close())

    async def get_audio_cog(self):
        return self.bot.get_cog("Audio")
    
    async def get_player_manager(self, audio_cog):
        """Get player manager safely from Audio cog"""
        if hasattr(audio_cog, "lavalink") and audio_cog.lavalink:
            return audio_cog.lavalink.player_manager
        elif hasattr(audio_cog, "_player_manager"):
            return audio_cog._player_manager
        return None

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, input: str):
        cmd = input.strip().lower()
        if cmd == "join":
            return await self.join_voice(ctx)
        if cmd == "leave":
            return await self.leave_voice(ctx)
        tllogger.debug(f"Command april: {input} by {ctx.author}")
        await self.process_query(ctx, input)

    async def join_voice(self, ctx):
        """Join voice channel using Red's audio system"""
        tllogger.debug("join_voice invoked")
        audio_cog = await self.get_audio_cog()
        if not audio_cog:
            return await ctx.send("❌ Audio cog is not loaded. Load it with `[p]load audio`")
        
        try:
            # Check user voice state
            if not ctx.author.voice or not ctx.author.voice.channel:
                return await ctx.send("❌ You must be in a voice channel.")
            
            channel = ctx.author.voice.channel
            permissions = channel.permissions_for(ctx.me)
            if not permissions.connect or not permissions.speak:
                return await ctx.send("❌ I need permissions to connect and speak!")
            
            # Get player manager
            player_manager = await self.get_player_manager(audio_cog)
            if not player_manager:
                return await ctx.send("❌ Audio system not ready. Try again in a moment.")
            
            # Get or create player
            player = player_manager.get(ctx.guild.id)
            if not player:
                # Ensure auto_connect is disabled
                await audio_cog.config.guild(ctx.guild).auto_connect.set(False)
                player = await player_manager.create(ctx.guild.id)
            
            # Connect or move
            if not player.is_connected:
                await player.connect(channel.id)
                await ctx.send(f"🔊 Joined {channel.name}")
            elif player.channel_id != channel.id:
                await player.move_to(channel.id)
                await ctx.send(f"🔊 Moved to {channel.name}")
            else:
                await ctx.send("✅ Already in your voice channel")
        except Exception as e:
            tllogger.exception("Failed join_voice")
            await ctx.send(f"❌ Join failed: {e}")

    async def leave_voice(self, ctx):
        audio_cog = await self.get_audio_cog()
        if not audio_cog:
            return await ctx.send("❌ Audio cog is not loaded. Load it with `[p]load audio`")
        
        player_manager = await self.get_player_manager(audio_cog)
        if not player_manager:
            return await ctx.send("❌ Audio system not ready.")
        
        player = player_manager.get(ctx.guild.id)
        if player and player.is_connected:
            await player.stop()
            await player.disconnect()
            await ctx.send("👋 Disconnected")
        else:
            await ctx.send("❌ Not in a voice channel.")

    async def process_query(self, ctx, input_text):
        audio_cog = await self.get_audio_cog()
        player_manager = await self.get_player_manager(audio_cog) if audio_cog else None
        player = player_manager.get(ctx.guild.id) if player_manager else None
        use_voice = await self.config.tts_enabled() and player and player.is_connected
        
        tllogger.debug(f"process_query use_voice={use_voice}")
        async with ctx.typing():
            try:
                resp = await self.query_deepseek(ctx.author.id, input_text, ctx)
                if not (use_voice and not await self.config.text_response_when_voice()):
                    await self.send_text_response(ctx, resp)
                if use_voice:
                    await self.speak_response(ctx, resp)
            except Exception as e:
                tllogger.exception("process_query error")
                await ctx.send(f"❌ Error: {e}")

    async def speak_response(self, ctx, text: str):
        tts_key = await self.config.tts_key()
        if not tts_key:
            tllogger.warning("Skipping TTS: missing API key.")
            return
        
        audio_cog = await self.get_audio_cog()
        if not audio_cog:
            return
        
        player_manager = await self.get_player_manager(audio_cog)
        if not player_manager:
            return
        
        player = player_manager.get(ctx.guild.id)
        if not player or not player.is_connected:
            tllogger.warning("Skipping TTS: Player not connected.")
            return
        
        # Split text into chunks for TTS
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        temp_files = []
        
        try:
            # Generate TTS for each chunk
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                async with self.session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{await self.config.voice_id()}",
                    json={"text": chunk, "voice_settings": {"stability":0.5,"similarity_boost":0.8}},
                    headers={"xi-api-key": tts_key}, timeout=30
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                    tf.write(data)
                    temp_files.append(tf.name)
            
            # Add tracks to player
            for path in temp_files:
                # Use Audio cog's method to get tracks
                if hasattr(audio_cog, '_audio_query_client'):
                    tracks = await audio_cog._audio_query_client.get_tracks(
                        ctx, path, False
                    )
                else:
                    # Fallback for older Red versions
                    tracks = await audio_cog.get_tracks(path)
                
                if not tracks:
                    tllogger.warning(f"No tracks found for TTS file: {path}")
                    continue
                
                # Add first track to player
                if isinstance(tracks, list):
                    track = tracks[0]
                else:
                    track = tracks["tracks"][0]
                    
                player.add(requester=ctx.author, track=track)
                tllogger.debug(f"Added TTS track to queue: {path}")
            
            # Start playing if not already
            if not player.is_playing:
                await player.play()
            
            # Schedule cleanup
            async def cleanup():
                await asyncio.sleep(60)
                for path in temp_files:
                    try:
                        os.unlink(path)
                        tllogger.debug(f"Cleaned up TTS file: {path}")
                    except:
                        pass
            self.bot.loop.create_task(cleanup())
            
        except Exception:
            tllogger.exception("TTS processing error")
            # Clean up any created files immediately on error
            for path in temp_files:
                try:
                    os.unlink(path)
                except:
                    pass

    @commands.group(name="aprilconfig", aliases=["aprilcfg"])
    @commands.is_owner()
    async def aprilconfig(self, ctx):
        """Configure AprilAI settings"""
        pass
        
    @aprilconfig.command()
    async def deepseekkey(self, ctx, key: str):
        """Set DeepSeek API key"""
        await self.config.deepseek_key.set(key)
        await ctx.tick()
    
    @aprilconfig.command()
    async def elevenlabs(self, ctx, key: str):
        """Set ElevenLabs API key"""
        await self.config.tts_key.set(key)
        await ctx.tick()
    
    @aprilconfig.command()
    async def voice(self, ctx, voice_id: str):
        """Set ElevenLabs voice ID (default: 21m00Tcm4TlvDq8ikWAM)"""
        await self.config.voice_id.set(voice_id)
        await ctx.send(f"✅ Voice ID set to `{voice_id}`")
    
    @aprilconfig.command()
    async def model(self, ctx, model_name: str):
        """Set DeepSeek model (default: deepseek-chat)"""
        await self.config.model.set(model_name.lower())
        await ctx.send(f"✅ Model set to `{model_name}`")
    
    @aprilconfig.command()
    async def prompt(self, ctx, *, system_prompt: str):
        """Set system prompt for the AI"""
        await self.config.system_prompt.set(system_prompt)
        await ctx.send("✅ System prompt updated")
    
    @aprilconfig.command()
    async def temperature(self, ctx, value: float):
        """Set AI temperature (0.0-1.0)"""
        if 0.0 <= value <= 1.0:
            await self.config.temperature.set(value)
            await ctx.send(f"✅ Temperature set to `{value}`")
        else:
            await ctx.send("❌ Value must be between 0.0 and 1.0")
    
    @aprilconfig.command()
    async def tokens(self, ctx, num: int):
        """Set max response tokens (100-4096)"""
        if 100 <= num <= 4096:
            await self.config.max_tokens.set(num)
            await ctx.send(f"✅ Max tokens set to `{num}`")
        else:
            await ctx.send("❌ Value must be between 100 and 4096")
    
    @aprilconfig.command()
    async def tts(self, ctx, enabled: bool):
        """Enable/disable TTS functionality"""
        await self.config.tts_enabled.set(enabled)
        status = "enabled" if enabled else "disabled"
        await ctx.send(f"✅ TTS {status}")
    
    @aprilconfig.command()
    async def textresponse(self, ctx, enabled: bool):
        """Enable/disable text responses when using voice"""
        await self.config.text_response_when_voice.set(enabled)
        status = "shown" if enabled else "hidden"
        await ctx.send(f"✅ Text responses will be {status} when using voice")
    
    @aprilconfig.command()
    async def settings(self, ctx):
        """Show current configuration"""
        cfg = await self.config.all()
        e = discord.Embed(title="AprilAI Configuration", color=await ctx.embed_color())
        
        # Security: show partial keys
        deepseek_key = cfg['deepseek_key']
        tts_key = cfg['tts_key']
        
        e.add_field(name="DeepSeek Key", value=f"`...{deepseek_key[-4:]}`" if deepseek_key else "❌ Not set", inline=False)
        e.add_field(name="ElevenLabs Key", value=f"`...{tts_key[-4:]}`" if tts_key else "❌ Not set", inline=False)
        e.add_field(name="Voice ID", value=f"`{cfg['voice_id']}`", inline=True)
        e.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        e.add_field(name="Temperature", value=f"`{cfg['temperature']}`", inline=True)
        e.add_field(name="Max Tokens", value=f"`{cfg['max_tokens']}`", inline=True)
        e.add_field(name="TTS Enabled", value="✅" if cfg['tts_enabled'] else "❌", inline=True)
        e.add_field(name="Text with Voice", value="✅" if cfg['text_response_when_voice'] else "❌", inline=True)
        
        prompt_preview = cfg['system_prompt'][:200] + ("..." if len(cfg['system_prompt']) > 200 else "")
        e.add_field(name="System Prompt", value=f"```{prompt_preview}```", inline=False)
        
        await ctx.send(embed=e)

    async def query_deepseek(self, user_id: int, question: str, ctx):
        key = await self.config.deepseek_key()
        if not key: 
            raise Exception("DeepSeek API key not set. Use `[p]aprilconfig deepseekkey <key>`")
            
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": [
                {"role": "system", "content": await self.config.system_prompt()},
                {"role": "user", "content": question}
            ],
            "temperature": await self.config.temperature(),
            "max_tokens": await self.config.max_tokens(),
            "user": str(user_id)
        }
        
        try:
            async with self.session.post(
                "https://api.deepseek.com/v1/chat/completions", 
                json=payload,
                headers=headers, 
                timeout=30
            ) as r:
                if r.status != 200:
                    error_data = await r.json()
                    err_msg = error_data.get("error", {}).get("message", f"HTTP Error {r.status}")
                    raise Exception(f"API Error: {err_msg}")
                
                data = await r.json()
                return data["choices"][0]["message"]["content"].strip()
        except asyncio.TimeoutError:
            raise Exception("API request timed out")
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")

    async def send_text_response(self, ctx, resp: str):
        if len(resp) > 1900:
            pages = list(pagify(resp, delims=["\n"," "], page_length=1500))
            await menu(ctx, pages, DEFAULT_CONTROLS)
        else:
            await ctx.send(resp)

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        if member.bot: 
            return
            
        audio_cog = await self.get_audio_cog()
        if not audio_cog:
            return
            
        player_manager = await self.get_player_manager(audio_cog)
        if not player_manager:
            return
            
        player = player_manager.get(member.guild.id)
        if player and player.is_connected:
            # Check if only bot remains in voice
            voice_channel = self.bot.get_channel(player.channel_id)
            if voice_channel and len(voice_channel.members) == 1:
                await player.disconnect()
                tllogger.debug(f"Left voice in {member.guild} (empty channel)")

async def setup(bot):
    await bot.add_cog(AprilAI(bot))