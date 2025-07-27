import asyncio
import aiohttp
import discord
import lavalink
import os
import logging
import random
import time
import tempfile
import base64
from collections import deque
from pathlib import Path
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import pagify
from redbot.core.utils.menus import menu, DEFAULT_CONTROLS

# Logger
tllogger = logging.getLogger("red.aprilai")
tllogger.setLevel(logging.DEBUG)

class AprilAI(commands.Cog):
    """AI assistant with text and voice via Lavalink"""

    def __init__(self, bot: Red):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1398462)
        self.session = aiohttp.ClientSession()
        # Per-channel conversation history {channel_id: deque}
        self.history = {}
        # Track TTS files for cleanup
        self.tts_files = set()
        self._unloading = False
        
        # Create TTS directory using cog-specific path
        self.tts_dir = Path(cog_data_path(self)) / "tts"
        self.tts_dir.mkdir(exist_ok=True, parents=True)
        
        self.config.register_global(
            deepseek_key="",
            tts_key="",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="You are April, a helpful AI assistant.",
            tts_enabled=True,
            text_response_when_voice=True,
            max_history=5,  # Default 5 exchanges
        )

    def cog_unload(self):
        self._unloading = True
        tllogger.debug("Unloading AprilAI, closing session and cleaning up TTS files.")
        
        self.bot.loop.create_task(self.session.close())
        
        # Clean up any remaining TTS files
        for path in list(self.tts_files):
            try:
                if os.path.exists(path):
                    os.unlink(path)
                tllogger.debug(f"Cleaned up TTS file on unload: {path}")
            except Exception as e:
                tllogger.error(f"Error cleaning up TTS file {path}: {e}")
            finally:
                self.tts_files.discard(path)

    def get_player(self, guild_id: int):
        """Get or create player for a guild using standard lavalink"""
        return lavalink.get_player(guild_id)

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, input: str):
        cmd = input.strip().lower()
        if cmd == "join":
            return await self.join_voice(ctx)
        if cmd == "leave":
            return await self.leave_voice(ctx)
        if cmd == "clearhistory":
            return await self.clear_history(ctx)
        tllogger.debug(f"Command april: {input} by {ctx.author}")
        await self.process_query(ctx, input)

    @april.command(name="clearhistory")
    async def clear_history(self, ctx):
        """Clear conversation history for this channel"""
        channel_id = ctx.channel.id
        if channel_id in self.history:
            self.history[channel_id].clear()
        await ctx.send("✅ Conversation history cleared for this channel.")

    async def join_voice(self, ctx):
        """Join voice channel"""
        tllogger.info(f"[AprilAI] join_voice invoked by {ctx.author}")
        
        # Check if user is in voice
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a voice channel.")
        
        channel = ctx.author.voice.channel
        permissions = channel.permissions_for(ctx.me)
        if not permissions.connect or not permissions.speak:
            return await ctx.send("❌ I need permissions to connect and speak!")
        
        try:
            # Get player using standard lavalink method
            player = self.get_player(ctx.guild.id)
            
            # Connect to voice channel
            if not player.is_connected:
                await player.connect(channel.id)
                await ctx.send(f"🔊 Joined {channel.name}")
            elif player.channel_id != channel.id:
                await player.move_to(channel.id)
                await ctx.send(f"🔊 Moved to {channel.name}")
            else:
                await ctx.send("✅ Already in your voice channel")
                
        except Exception as e:
            tllogger.exception("[AprilAI] Failed join_voice")
            await ctx.send(f"❌ Join failed: {e}")

    async def leave_voice(self, ctx):
        """Leave voice channel"""
        try:
            player = self.get_player(ctx.guild.id)
            if player.is_connected:
                await player.stop()
                await player.disconnect()
                await ctx.send("👋 Disconnected")
            else:
                await ctx.send("❌ Not in a voice channel.")
        except Exception as e:
            tllogger.error(f"Disconnect failed: {e}")
            await ctx.send(f"❌ Disconnect failed: {e}")

    async def process_query(self, ctx, input_text):
        # Check if we should use voice
        use_voice = False
        if await self.config.tts_enabled():
            try:
                player = self.get_player(ctx.guild.id)
                use_voice = player.is_connected
            except:
                use_voice = False
        
        tllogger.debug(f"process_query use_voice={use_voice}")
        async with ctx.typing():
            try:
                # Get or create history for this channel
                channel_id = ctx.channel.id
                if channel_id not in self.history:
                    max_history = await self.config.max_history()
                    self.history[channel_id] = deque(maxlen=max_history * 2)  # 2 messages per exchange
                
                # Build message history
                messages = [
                    {"role": "system", "content": await self.config.system_prompt()}
                ]
                messages.extend(self.history[channel_id])
                messages.append({"role": "user", "content": input_text})
                
                resp = await self.query_deepseek(ctx.author.id, messages)
                
                # Update history with new exchange
                self.history[channel_id].append({"role": "user", "content": input_text})
                self.history[channel_id].append({"role": "assistant", "content": resp})
                
                # Send text response unless disabled when using voice
                if not (use_voice and not await self.config.text_response_when_voice()):
                    await self.send_text_response(ctx, resp)
                
                # Send voice response if enabled and connected
                if use_voice:
                    await self.speak_response(ctx, resp)
                    
            except Exception as e:
                tllogger.exception("process_query error")
                await ctx.send(f"❌ Error: {e}")

    async def speak_response(self, ctx, text: str):
        """Generate TTS and play through Lavalink - simplified version"""
        tts_key = await self.config.tts_key()
        if not tts_key:
            tllogger.warning("Skipping TTS: missing API key.")
            return
        
        try:
            # Get player
            player = self.get_player(ctx.guild.id)
            if not player.is_connected:
                tllogger.warning("Skipping TTS: Player not connected.")
                return
            
            # Clean text for TTS
            clean_text = self.clean_text_for_tts(text)
            if not clean_text.strip():
                return
            
            # Generate TTS audio
            audio_data = await self.generate_tts_audio(clean_text, tts_key)
            if not audio_data:
                tllogger.error("Failed to generate TTS audio")
                return
            
            # Create data URI directly for Lavalink
            encoded_data = base64.b64encode(audio_data).decode('utf-8')
            data_uri = f"data:audio/mpeg;base64,{encoded_data}"
            
            tllogger.debug(f"Created data URI with {len(audio_data)} bytes of audio data")
            
            # Get tracks from Lavalink
            try:
                results = await lavalink.get_tracks(data_uri)
                if results and results.tracks:
                    track = results.tracks[0]
                    player.add(requester=ctx.author.id, track=track)
                    tllogger.debug("Successfully added TTS track to queue")
                    
                    # Start playing if not already playing
                    if not player.is_playing:
                        await player.play()
                        tllogger.debug("Started playing TTS")
                else:
                    tllogger.error(f"No tracks found for data URI (length: {len(data_uri)})")
                    
            except Exception as e:
                tllogger.error(f"Failed to load track from data URI: {e}")
                
        except Exception as e:
            tllogger.exception("TTS processing error")

    def clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS output"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # *italic*
        text = re.sub(r'`(.*?)`', r'\1', text)        # `code`
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # code blocks
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean up extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit length to avoid API issues
        if len(text) > 1000:
            text = text[:1000].rsplit(' ', 1)[0] + "..."
        
        return text

    async def generate_tts_audio(self, text: str, api_key: str) -> bytes:
        """Generate TTS audio using ElevenLabs API"""
        try:
            voice_id = await self.config.voice_id()
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            payload = {
                "text": text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8
                }
            }
            
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json"
            }
            
            tllogger.debug(f"Requesting TTS for text: {text[:100]}...")
            
            async with self.session.post(url, json=payload, headers=headers, timeout=30) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    tllogger.debug(f"Successfully generated {len(audio_data)} bytes of TTS audio")
                    return audio_data
                else:
                    error_text = await resp.text()
                    tllogger.error(f"TTS API error {resp.status}: {error_text}")
                    return None
                    
        except Exception as e:
            tllogger.error(f"TTS generation failed: {e}")
            return None

    @commands.group(name="aprilconfig", aliases=["aprilcfg"])
    @commands.is_owner()
    async def aprilconfig(self, ctx):
        """Configure AprilAI settings"""
        if ctx.invoked_subcommand is None:
            await self.show_settings(ctx)

    @aprilconfig.command()
    async def deepseekkey(self, ctx, key: str):
        """Set DeepSeek API key"""
        await self.config.deepseek_key.set(key)
        await ctx.tick()
        try:
            await ctx.message.delete()
        except:
            pass

    @aprilconfig.command()
    async def elevenlabs(self, ctx, key: str):
        """Set ElevenLabs API key"""
        await self.config.tts_key.set(key)
        await ctx.tick()
        try:
            await ctx.message.delete()
        except:
            pass

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
    async def maxhistory(self, ctx, num: int):
        """Set max conversation history exchanges (1-20)"""
        if 1 <= num <= 20:
            await self.config.max_history.set(num)
            # Update existing history maxlen
            for channel_id in self.history:
                self.history[channel_id] = deque(self.history[channel_id], maxlen=num*2)
            await ctx.send(f"✅ Max history set to `{num}` exchanges")
        else:
            await ctx.send("❌ Value must be between 1 and 20")

    @aprilconfig.command(name="debug")
    async def debug_lavalink(self, ctx):
        """Debug Lavalink connection status"""
        embed = discord.Embed(title="AprilAI Debug Information", color=0xff0000)
        
        try:
            # Test getting a player
            player = self.get_player(ctx.guild.id)
            embed.add_field(name="Player Access", value="✅ Success", inline=True)
            embed.add_field(name="Player Connected", value="✅ Yes" if player.is_connected else "❌ No", inline=True)
            
            if player.is_connected:
                channel = self.bot.get_channel(player.channel_id)
                channel_name = channel.name if channel else "Unknown"
                embed.add_field(name="Connected Channel", value=channel_name, inline=True)
                
        except Exception as e:
            embed.add_field(name="Player Access", value=f"❌ Error: {e}", inline=False)
        
        # Check if lavalink module is available
        try:
            import lavalink
            embed.add_field(name="Lavalink Module", value="✅ Available", inline=True)
        except ImportError:
            embed.add_field(name="Lavalink Module", value="❌ Not available", inline=True)
        
        # Test TTS generation
        tts_key = await self.config.tts_key()
        if tts_key:
            embed.add_field(name="ElevenLabs Key", value="✅ Set", inline=True)
            try:
                test_audio = await self.generate_tts_audio("Test", tts_key)
                if test_audio:
                    embed.add_field(name="TTS Generation", value=f"✅ Success ({len(test_audio)} bytes)", inline=True)
                else:
                    embed.add_field(name="TTS Generation", value="❌ Failed", inline=True)
            except Exception as e:
                embed.add_field(name="TTS Generation", value=f"❌ Error: {str(e)[:50]}", inline=True)
        else:
            embed.add_field(name="ElevenLabs Key", value="❌ Not set", inline=True)
        
        await ctx.send(embed=embed)

    @aprilconfig.command(name="settings")
    async def show_settings(self, ctx):
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
        e.add_field(name="Max History", value=f"`{cfg['max_history']} exchanges`", inline=True)
        e.add_field(name="TTS Enabled", value="✅" if cfg['tts_enabled'] else "❌", inline=True)
        e.add_field(name="Text with Voice", value="✅" if cfg['text_response_when_voice'] else "❌", inline=True)
        
        prompt_preview = cfg['system_prompt'][:200] + ("..." if len(cfg['system_prompt']) > 200 else "")
        e.add_field(name="System Prompt", value=f"```{prompt_preview}```", inline=False)
        
        await ctx.send(embed=e)

    async def query_deepseek(self, user_id: int, messages: list):
        """Query DeepSeek API"""
        key = await self.config.deepseek_key()
        if not key: 
            raise Exception("DeepSeek API key not set. Use `[p]aprilconfig deepseekkey <key>`")
            
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": await self.config.model(),
            "messages": messages,
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
        """Send text response, paginated if necessary"""
        if len(resp) > 1900:
            pages = list(pagify(resp, delims=["\n", " "], page_length=1500))
            await menu(ctx, pages, DEFAULT_CONTROLS)
        else:
            await ctx.send(resp)

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        """Handle voice state updates"""
        if member.bot:
            return
            
        try:
            player = self.get_player(member.guild.id)
            if player.is_connected:
                # Check if only bot remains in voice
                voice_channel = self.bot.get_channel(player.channel_id)
                if voice_channel:
                    human_members = [m for m in voice_channel.members if not m.bot]
                    if len(human_members) == 0:
                        await player.disconnect()
                        tllogger.debug(f"Left voice in {member.guild} (empty channel)")
        except Exception as e:
            tllogger.error(f"Error handling voice state update: {e}")

async def setup(bot):
    """Set up the cog"""
    await bot.add_cog(AprilAI(bot))