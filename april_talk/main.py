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
import re
from collections import deque
from pathlib import Path
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import pagify
from redbot.core.utils.menus import menu, DEFAULT_CONTROLS
from typing import Optional, List, Dict

# Logger
tllogger = logging.getLogger("red.aprilai")
tllogger.setLevel(logging.DEBUG)

# Emotion GIFs repository
EMOTION_GIFS = {
    "happy": [
        "https://media.giphy.com/media/XbxZ41fWLeRECPsGIJ/giphy.gif",
        "https://media.giphy.com/media/l0HlMG1EX2H38cZeE/giphy.gif",
        "https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif"
    ],
    "thinking": [
        "https://media.giphy.com/media/d3mlE7uhX8KFgEmY/giphy.gif",
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",
        "https://media.giphy.com/media/l0HlUNj5BRuYDLxFm/giphy.gif"
    ],
    "confused": [
        "https://media.giphy.com/media/3o7btPCcdNniyf0ArS/giphy.gif",
        "https://media.giphy.com/media/xT0xeJpnrWC4XWblEk/giphy.gif",
        "https://media.giphy.com/media/3oEjI5VtIhHvK37WYo/giphy.gif"
    ],
    "excited": [
        "https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif",
        "https://media.giphy.com/media/l0HlMURBbyUqF0XQI/giphy.gif",
        "https://media.giphy.com/media/3rgXBOmTlzyFCURutG/giphy.gif"
    ],
    "sad": [
        "https://media.giphy.com/media/OPU6wzx8JrHna/giphy.gif",
        "https://media.giphy.com/media/l1AsyjZ8XLd1V7pUk/giphy.gif",
        "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif"
    ]
}

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
        # Streaming message cache
        self.streaming_messages = {}
        
        # Create TTS directory using cog-specific path
        self.tts_dir = Path(cog_data_path(self)) / "tts"
        self.tts_dir.mkdir(exist_ok=True, parents=True)
        
        self.config.register_global(
            deepseek_key="",
            anthropic_key="",  # For smert mode
            tts_key="",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=2048,
            system_prompt="You are April, a helpful AI assistant.",
            smert_prompt="You are April in 'smert' mode - an incredibly intelligent, witty, and creative AI assistant with deep knowledge across all domains.",
            tts_enabled=True,
            text_response_when_voice=True,
            max_history=5,  # Default 5 exchanges
            use_gifs=True,
            max_message_length=1800,  # For splitting long messages
        )
        
        self.config.register_user(
            smert_mode=False,
            custom_anthropic_key="",
            custom_smert_prompt=""
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

    async def detect_emotion(self, text: str) -> Optional[str]:
        """Detect emotion from text for GIF selection"""
        text_lower = text.lower()
        
        # Simple keyword-based emotion detection
        if any(word in text_lower for word in ["happy", "great", "awesome", "excellent", "wonderful", "amazing"]):
            return "happy"
        elif any(word in text_lower for word in ["think", "consider", "ponder", "wonder", "hmm"]):
            return "thinking"
        elif any(word in text_lower for word in ["confused", "don't understand", "what", "huh", "unclear"]):
            return "confused"
        elif any(word in text_lower for word in ["excited", "can't wait", "awesome!", "wow", "amazing!"]):
            return "excited"
        elif any(word in text_lower for word in ["sad", "sorry", "unfortunately", "regret"]):
            return "sad"
        
        return None

    def get_player(self, guild_id: int):
        """Get player for a guild using lavalink"""
        try:
            # Use the proper lavalink player access method
            player = lavalink.get_player(guild_id)
            tllogger.debug(f"Got player for guild {guild_id}: {player}")
            return player
        except Exception as e:
            tllogger.error(f"Failed to get player for guild {guild_id}: {e}")
            return None

    def is_player_connected(self, player) -> bool:
        """Check if player is connected"""
        if not player:
            return False
        
        try:
            # Check if player is connected to a voice channel
            return player.is_connected
        except Exception as e:
            tllogger.error(f"Error checking player connection: {e}")
            return False

    def get_player_channel_id(self, player):
        """Get the channel ID the player is connected to"""
        if not player:
            return None
            
        try:
            return player.channel_id
        except Exception as e:
            tllogger.error(f"Error getting player channel ID: {e}")
            return None

    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, input: str):
        """Main April command"""
        cmd = input.strip().lower()
        
        # Handle special commands
        if cmd == "join":
            return await self.join_voice(ctx)
        if cmd == "leave":
            return await self.leave_voice(ctx)
        if cmd == "clearhistory":
            return await self.clear_history(ctx)
        if cmd == "smert":
            return await self.toggle_smert_mode(ctx)
            
        tllogger.debug(f"Command april: {input} by {ctx.author}")
        await self.process_query(ctx, input)

    @april.command(name="smert")
    async def toggle_smert_mode(self, ctx):
        """Toggle smert mode (requires Anthropic API key)"""
        user_config = self.config.user(ctx.author)
        current_mode = await user_config.smert_mode()
        
        if not current_mode:
            # Check if user has custom key or global key exists
            custom_key = await user_config.custom_anthropic_key()
            global_key = await self.config.anthropic_key()
            
            if not custom_key and not global_key:
                return await ctx.send(
                    "❌ Smert mode requires an Anthropic API key. "
                    "Set one with `[p]aprilconfig anthropickey <key>` or "
                    "`[p]apriluser setkey <key>`"
                )
            
            await user_config.smert_mode.set(True)
            await ctx.send("🧠 Smert mode activated! I'm now using Claude for enhanced intelligence.")
        else:
            await user_config.smert_mode.set(False)
            await ctx.send("💡 Smert mode deactivated. Back to regular mode.")

    @april.command(name="clearhistory")
    async def clear_history(self, ctx):
        """Clear conversation history for this channel"""
        channel_id = ctx.channel.id
        if channel_id in self.history:
            self.history[channel_id].clear()
        await ctx.send("✅ Conversation history cleared for this channel.")

    @april.command(name="join")
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
            # Check if lavalink is available
            if not hasattr(lavalink, 'get_player'):
                return await ctx.send("❌ Lavalink is not initialized. Please load the Audio cog first with `[p]load audio`")
            
            # Try to get existing player first
            player = self.get_player(ctx.guild.id)
            
            if player and self.is_player_connected(player):
                current_channel_id = self.get_player_channel_id(player)
                if current_channel_id == channel.id:
                    return await ctx.send(f"✅ Already connected to {channel.name}")
                else:
                    # Disconnect from current channel
                    await player.disconnect()
            
            # Connect to new channel
            await lavalink.connect(channel)
            player = self.get_player(ctx.guild.id)
            
            if player and self.is_player_connected(player):
                await ctx.send(f"🔊 Joined {channel.name}")
            else:
                await ctx.send("❌ Failed to connect to voice channel")
                
        except Exception as e:
            tllogger.exception("[AprilAI] Failed join_voice")
            await ctx.send(f"❌ Join failed: {e}")

    @april.command(name="leave")
    async def leave_voice(self, ctx):
        """Leave voice channel"""
        try:
            player = self.get_player(ctx.guild.id)
            if player and self.is_player_connected(player):
                await player.stop()
                await player.disconnect()
                await ctx.send("👋 Disconnected from voice")
            else:
                await ctx.send("❌ Not connected to a voice channel.")
        except Exception as e:
            tllogger.error(f"Disconnect failed: {e}")
            await ctx.send(f"❌ Disconnect failed: {e}")

    async def process_query(self, ctx, input_text):
        # Check if we should use voice
        use_voice = False
        if await self.config.tts_enabled():
            try:
                player = self.get_player(ctx.guild.id)
                use_voice = player and self.is_player_connected(player)
                tllogger.debug(f"TTS check - player: {player}, connected: {use_voice}")
            except:
                use_voice = False
        
        # Check if user is in smert mode
        user_config = self.config.user(ctx.author)
        smert_mode = await user_config.smert_mode()
        
        tllogger.debug(f"process_query use_voice={use_voice}, smert_mode={smert_mode}")
        
        # Start typing indicator
        async with ctx.typing():
            try:
                # Get or create history for this channel
                channel_id = ctx.channel.id
                if channel_id not in self.history:
                    max_history = await self.config.max_history()
                    self.history[channel_id] = deque(maxlen=max_history * 2)  # 2 messages per exchange
                
                # Build message history
                if smert_mode:
                    custom_prompt = await user_config.custom_smert_prompt()
                    system_prompt = custom_prompt or await self.config.smert_prompt()
                else:
                    system_prompt = await self.config.system_prompt()
                    
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.history[channel_id])
                messages.append({"role": "user", "content": input_text})
                
                # Get response from appropriate API
                if smert_mode:
                    resp = await self.query_anthropic(ctx.author.id, messages)
                else:
                    resp = await self.query_deepseek(ctx.author.id, messages)
                
                # Update history with new exchange
                self.history[channel_id].append({"role": "user", "content": input_text})
                self.history[channel_id].append({"role": "assistant", "content": resp})
                
                # Start both text and voice simultaneously for natural conversation feel
                tasks = []
                
                # Always send text response (unless specifically disabled for voice)
                if not (use_voice and not await self.config.text_response_when_voice()):
                    tasks.append(asyncio.create_task(self.send_streamed_response(ctx, resp)))
                
                # Send voice response if enabled and connected
                if use_voice:
                    tasks.append(asyncio.create_task(self.speak_response(ctx, resp)))
                
                # Wait for both to complete
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except Exception as e:
                tllogger.exception("process_query error")
                await ctx.send(f"❌ Error: {e}")

    async def send_streamed_response(self, ctx, resp: str):
        """Send response with streaming effect and handle long messages"""
        max_length = await self.config.max_message_length()
        use_gifs = await self.config.use_gifs()
        
        # Detect emotion and potentially add GIF
        emotion = await self.detect_emotion(resp) if use_gifs else None
        gif_url = None
        if emotion and random.random() < 0.3:  # 30% chance to include GIF
            gif_url = random.choice(EMOTION_GIFS[emotion])
        
        # Split long messages
        if len(resp) > max_length:
            # Create initial "thinking" message
            embed = discord.Embed(
                description="💭 *Thinking...*",
                color=await ctx.embed_color()
            )
            thinking_msg = await ctx.send(embed=embed)
            
            # Split response into chunks
            chunks = []
            words = resp.split(' ')
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > max_length:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Send first chunk by editing thinking message
            first_chunk = chunks[0]
            if gif_url and len(chunks) == 1:
                await thinking_msg.edit(content=first_chunk + f"\n{gif_url}", embed=None)
            else:
                await thinking_msg.edit(content=first_chunk, embed=None)
            
            # Send remaining chunks
            for i, chunk in enumerate(chunks[1:], 1):
                await asyncio.sleep(0.5)  # Small delay between messages
                if gif_url and i == len(chunks) - 1:
                    await ctx.send(chunk + f"\n{gif_url}")
                else:
                    await ctx.send(chunk)
        else:
            # Single message with streaming effect
            embed = discord.Embed(
                description="💭 *Thinking...*",
                color=await ctx.embed_color()
            )
            msg = await ctx.send(embed=embed)
            
            # Simulate streaming by updating message in chunks
            chunk_size = 50
            for i in range(0, len(resp), chunk_size):
                chunk = resp[:i + chunk_size]
                if i + chunk_size >= len(resp) and gif_url:
                    await msg.edit(content=chunk + f"\n{gif_url}", embed=None)
                else:
                    await msg.edit(content=chunk + "▌", embed=None)
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            # Final message without cursor
            if gif_url:
                await msg.edit(content=resp + f"\n{gif_url}", embed=None)
            else:
                await msg.edit(content=resp, embed=None)

    async def speak_response(self, ctx, text: str):
        """Generate TTS and play via Lavalink"""
        tts_key = await self.config.tts_key()
        if not tts_key:
            tllogger.warning("Skipping TTS: missing API key.")
            return

        try:
            # Verify voice connection
            player = self.get_player(ctx.guild.id)
            if not self.is_player_connected(player):
                tllogger.warning("Skipping TTS: Player not connected.")
                return

            # Clean and generate TTS
            clean_text = self.clean_text_for_tts(text)
            if not clean_text.strip():
                return

            audio_data = await self.generate_tts_audio(clean_text, tts_key)
            if not audio_data:
                tllogger.error("Failed to generate TTS audio")
                return

            # Save to temporary file
            filename = f"tts_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
            filepath = self.tts_dir / filename

            with open(filepath, 'wb') as f:
                f.write(audio_data)

            tllogger.debug(f"TTS audio saved: {filepath}")
            self.tts_files.add(str(filepath))

            # Create Lavalink track and play
            track = await lavalink.decode_track(f"file://{filepath}")
            if track:
                player.add(requester=ctx.author.id, track=track)
                if not player.is_playing:
                    await player.play()
                    
                # Schedule cleanup after the track duration + buffer
                track_duration = track.duration / 1000  # Convert to seconds
                cleanup_delay = max(track_duration + 5, 30)  # At least 30 seconds
                asyncio.create_task(self._cleanup_tts_file(str(filepath), cleanup_delay))
            else:
                tllogger.error("Failed to decode TTS track")
                self._cleanup_tts_file_sync(str(filepath))

        except Exception as e:
            tllogger.exception("TTS playback failed")

    async def _cleanup_tts_file(self, path: str, delay: float):
        """Clean up TTS file after delay"""
        await asyncio.sleep(delay)
        self._cleanup_tts_file_sync(path)

    def _cleanup_tts_file_sync(self, path: str):
        """Synchronously clean up TTS file"""
        try:
            if os.path.exists(path):
                os.unlink(path)
            self.tts_files.discard(path)
            tllogger.debug(f"Cleaned up TTS file: {path}")
        except Exception as e:
            tllogger.error(f"Error cleaning up TTS file {path}: {e}")

    def clean_text_for_tts(self, text: str) -> str:
        """Clean text for better TTS output"""
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
    async def anthropickey(self, ctx, key: str):
        """Set Anthropic API key for smert mode"""
        await self.config.anthropic_key.set(key)
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
    async def smertprompt(self, ctx, *, prompt: str):
        """Set default smert mode prompt"""
        await self.config.smert_prompt.set(prompt)
        await ctx.send("✅ Smert mode prompt updated")

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

    @aprilconfig.command()
    async def gifs(self, ctx, enabled: bool):
        """Enable/disable emotion GIFs"""
        await self.config.use_gifs.set(enabled)
        status = "enabled" if enabled else "disabled"
        await ctx.send(f"✅ Emotion GIFs {status}")

    @aprilconfig.command()
    async def messagelength(self, ctx, length: int):
        """Set max message length before splitting (500-2000)"""
        if 500 <= length <= 2000:
            await self.config.max_message_length.set(length)
            await ctx.send(f"✅ Max message length set to `{length}`")
        else:
            await ctx.send("❌ Value must be between 500 and 2000")

    @aprilconfig.command(name="debug")
    async def debug_lavalink(self, ctx):
        """Debug Lavalink connection status"""
        embed = discord.Embed(title="AprilAI Debug Information", color=0xff0000)
        
        try:
            # Check lavalink initialization
            if hasattr(lavalink, 'get_player'):
                embed.add_field(name="Lavalink Status", value="✅ Initialized", inline=True)
                
                # Test getting a player with better error handling
                try:
                    player = self.get_player(ctx.guild.id)
                    if player is not None:
                        embed.add_field(name="Player Access", value="✅ Success", inline=True)
                        
                        is_connected = self.is_player_connected(player)
                        embed.add_field(name="Player Connected", value="✅ Yes" if is_connected else "❌ No", inline=True)
                        
                        if is_connected:
                            try:
                                channel_id = self.get_player_channel_id(player)
                                if channel_id:
                                    channel = self.bot.get_channel(channel_id)
                                    channel_name = channel.name if channel else f"Unknown ({channel_id})"
                                    embed.add_field(name="Connected Channel", value=channel_name, inline=True)
                                else:
                                    embed.add_field(name="Connected Channel", value="❌ No channel ID", inline=True)
                            except Exception as e:
                                embed.add_field(name="Connected Channel", value=f"❌ Error: {e}", inline=True)
                    else:
                        embed.add_field(name="Player Access", value="❌ Player is None", inline=True)
                        embed.add_field(name="Note", value="Try using `[p]april join` first", inline=True)
                except Exception as e:
                    embed.add_field(name="Player Access", value=f"❌ Exception: {e}", inline=True)
            else:
                embed.add_field(name="Lavalink Status", value="❌ Not initialized", inline=True)
                embed.add_field(name="Solution", value="Load the Audio cog first with `[p]load audio`", inline=False)
                
        except Exception as e:
            embed.add_field(name="Debug Error", value=f"❌ Error: {e}", inline=False)
        
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
        
        # Add guild info for debugging
        embed.add_field(name="Guild ID", value=f"`{ctx.guild.id}`", inline=True)
        embed.add_field(name="Bot Voice State", value="✅ Connected" if ctx.guild.voice_client else "❌ Not connected", inline=True)
        
        await ctx.send(embed=embed)

    @aprilconfig.command(name="settings")
    async def show_settings(self, ctx):
        """Show current configuration"""
        cfg = await self.config.all()
        e = discord.Embed(title="AprilAI Configuration", color=await ctx.embed_color())
        
        # Security: show partial keys
        deepseek_key = cfg['deepseek_key']
        anthropic_key = cfg['anthropic_key']
        tts_key = cfg['tts_key']
        
        e.add_field(name="DeepSeek Key", value=f"`...{deepseek_key[-4:]}`" if deepseek_key else "❌ Not set", inline=False)
        e.add_field(name="Anthropic Key", value=f"`...{anthropic_key[-4:]}`" if anthropic_key else "❌ Not set", inline=False)
        e.add_field(name="ElevenLabs Key", value=f"`...{tts_key[-4:]}`" if tts_key else "❌ Not set", inline=False)
        e.add_field(name="Voice ID", value=f"`{cfg['voice_id']}`", inline=True)
        e.add_field(name="Model", value=f"`{cfg['model']}`", inline=True)
        e.add_field(name="Temperature", value=f"`{cfg['temperature']}`", inline=True)
        e.add_field(name="Max Tokens", value=f"`{cfg['max_tokens']}`", inline=True)
        e.add_field(name="Max History", value=f"`{cfg['max_history']} exchanges`", inline=True)
        e.add_field(name="Max Message Length", value=f"`{cfg['max_message_length']}`", inline=True)
        e.add_field(name="TTS Enabled", value="✅" if cfg['tts_enabled'] else "❌", inline=True)
        e.add_field(name="Text with Voice", value="✅" if cfg['text_response_when_voice'] else "❌", inline=True)
        e.add_field(name="Emotion GIFs", value="✅" if cfg['use_gifs'] else "❌", inline=True)
        
        prompt_preview = cfg['system_prompt'][:200] + ("..." if len(cfg['system_prompt']) > 200 else "")
        e.add_field(name="System Prompt", value=f"```{prompt_preview}```", inline=False)
        
        smert_preview = cfg['smert_prompt'][:200] + ("..." if len(cfg['smert_prompt']) > 200 else "")
        e.add_field(name="Smert Prompt", value=f"```{smert_preview}```", inline=False)
        
        await ctx.send(embed=e)

    @commands.group(name="apriluser")
    async def apriluser(self, ctx):
        """User-specific AprilAI settings"""
        if ctx.invoked_subcommand is None:
            await self.show_user_settings(ctx)

    @apriluser.command(name="setkey")
    async def set_user_key(self, ctx, key: str):
        """Set your personal Anthropic API key for smert mode"""
        await self.config.user(ctx.author).custom_anthropic_key.set(key)
        await ctx.tick()
        try:
            await ctx.message.delete()
        except:
            pass

    @apriluser.command(name="setprompt")
    async def set_user_prompt(self, ctx, *, prompt: str):
        """Set your personal smert mode prompt"""
        await self.config.user(ctx.author).custom_smert_prompt.set(prompt)
        await ctx.send("✅ Your personal smert prompt has been set")

    @apriluser.command(name="settings")
    async def show_user_settings(self, ctx):
        """Show your personal settings"""
        user_cfg = await self.config.user(ctx.author).all()
        
        e = discord.Embed(
            title=f"AprilAI Settings for {ctx.author.display_name}",
            color=await ctx.embed_color()
        )
        
        e.add_field(name="Smert Mode", value="✅ Active" if user_cfg['smert_mode'] else "❌ Inactive", inline=True)
        
        custom_key = user_cfg['custom_anthropic_key']
        e.add_field(
            name="Personal Anthropic Key", 
            value=f"`...{custom_key[-4:]}`" if custom_key else "❌ Not set", 
            inline=True
        )
        
        if user_cfg['custom_smert_prompt']:
            prompt_preview = user_cfg['custom_smert_prompt'][:200] + ("..." if len(user_cfg['custom_smert_prompt']) > 200 else "")
            e.add_field(name="Personal Smert Prompt", value=f"```{prompt_preview}```", inline=False)
        else:
            e.add_field(name="Personal Smert Prompt", value="Using default smert prompt", inline=False)
        
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
                timeout=60  # Increased timeout for large responses
            ) as r:
                if r.status != 200:
                    error_data = await r.json()
                    err_msg = error_data.get("error", {}).get("message", f"HTTP Error {r.status}")
                    raise Exception(f"API Error: {err_msg}")
                
                data = await r.json()
                return data["choices"][0]["message"]["content"].strip()
        except asyncio.TimeoutError:
            raise Exception("API request timed out - try reducing max_tokens")
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")

    async def query_anthropic(self, user_id: int, messages: list):
        """Query Anthropic Claude API for smert mode"""
        # Check for user's custom key first
        user = self.bot.get_user(user_id)
        if user:
            custom_key = await self.config.user(user).custom_anthropic_key()
            if custom_key:
                key = custom_key
            else:
                key = await self.config.anthropic_key()
        else:
            key = await self.config.anthropic_key()
            
        if not key:
            raise Exception("Anthropic API key not set. Use `[p]aprilconfig anthropickey <key>` or `[p]apriluser setkey <key>`")
        
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Extract system message and user/assistant messages
        system_content = None
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                conversation_messages.append(msg)
        
        # Build the payload with proper format for latest Anthropic API
        payload = {
            "model": "claude-3-5-sonnet-20241022",  # Latest stable Claude model
            "messages": conversation_messages,
            "max_tokens": await self.config.max_tokens(),
            "temperature": await self.config.temperature()
        }
        
        # Add system prompt if it exists
        if system_content:
            payload["system"] = system_content
        
        try:
            async with self.session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=60
            ) as r:
                if r.status != 200:
                    error_text = await r.text()
                    raise Exception(f"Anthropic API Error {r.status}: {error_text}")
                
                data = await r.json()
                return data["content"][0]["text"].strip()
        except asyncio.TimeoutError:
            raise Exception("API request timed out - try reducing max_tokens")
        except Exception as e:
            raise Exception(f"API Error: {str(e)}")

    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        """Handle voice state updates"""
        if member.bot:
            return
            
        try:
            player = self.get_player(member.guild.id)
            if player and self.is_player_connected(player):
                # Check if only bot remains in voice
                channel_id = self.get_player_channel_id(player)
                if channel_id:
                    voice_channel = self.bot.get_channel(channel_id)
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
