import asyncio
import aiohttp
import discord
import os
import logging
import random
import time
import tempfile
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
        self._lavalink_ready = False
        
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
            tts_server_url=""  # For HTTP server hosting TTS files
        )
        
        # Initialize Lavalink connection
        self._init_task = bot.loop.create_task(self.initialize_lavalink())

    async def initialize_lavalink(self):
        """Wait for Audio cog and Lavalink to be ready"""
        max_retries = 30
        retry_delay = 2
        
        for attempt in range(1, max_retries + 1):
            try:
                tllogger.info(f"[AprilAI] Checking Lavalink readiness (attempt {attempt}/{max_retries})")
                
                # Get Audio cog
                audio_cog = self.bot.get_cog("Audio")
                if not audio_cog:
                    tllogger.info("[AprilAI] Audio cog not loaded yet. Waiting...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Check if Lavalink attribute exists
                if not hasattr(audio_cog, "lavalink"):
                    tllogger.info("[AprilAI] Audio cog has no lavalink attribute yet. Waiting...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                lavalink = audio_cog.lavalink
                if not lavalink:
                    tllogger.info("[AprilAI] Lavalink client is None. Waiting...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Check node manager
                if not hasattr(lavalink, 'node_manager') or not lavalink.node_manager:
                    tllogger.info("[AprilAI] Node manager not available yet. Waiting...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Check if we have any nodes at all
                try:
                    nodes = getattr(lavalink.node_manager, 'nodes', [])
                    if not nodes:
                        tllogger.info("[AprilAI] No nodes configured yet. Waiting...")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    # Check if any node is available
                    available_nodes = [n for n in nodes if hasattr(n, 'is_available') and n.is_available()]
                    if not available_nodes:
                        tllogger.info(f"[AprilAI] No available nodes yet ({len(nodes)} total nodes). Waiting...")
                        await asyncio.sleep(retry_delay)
                        continue
                    
                    tllogger.info(f"[AprilAI] Found {len(available_nodes)} available nodes out of {len(nodes)} total")
                    
                except Exception as e:
                    tllogger.info(f"[AprilAI] Error checking nodes: {e}. Waiting...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                # Test creating a player to ensure everything works
                try:
                    test_guild_id = 123456789  # Dummy guild ID for testing
                    if hasattr(lavalink, 'player_manager') and lavalink.player_manager:
                        # Just check if player manager is functional
                        tllogger.info("[AprilAI] Player manager is available")
                    else:
                        tllogger.info("[AprilAI] Player manager not available yet. Waiting...")
                        await asyncio.sleep(retry_delay)
                        continue
                        
                except Exception as e:
                    tllogger.info(f"[AprilAI] Player manager test failed: {e}. Waiting...")
                    await asyncio.sleep(retry_delay)
                    continue
                
                self._lavalink_ready = True
                tllogger.info("[AprilAI] Lavalink is ready!")
                return
                
            except Exception as e:
                tllogger.error(f"[AprilAI] Lavalink initialization check failed (attempt {attempt}): {e}")
            
            await asyncio.sleep(retry_delay)
        
        tllogger.error("[AprilAI] Failed to detect Lavalink initialization after multiple attempts")
        tllogger.error("[AprilAI] TTS functionality will be disabled until Lavalink is ready")

    def cog_unload(self):
        self._unloading = True
        tllogger.debug("Unloading AprilAI, closing session and cleaning up TTS files.")
        
        # Cancel initialization task
        if hasattr(self, '_init_task') and not self._init_task.done():
            self._init_task.cancel()
        
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

    def get_audio_cog(self):
        """Get the Audio cog instance"""
        return self.bot.get_cog("Audio")
    
    def get_lavalink(self):
        """Get Lavalink client from Audio cog"""
        audio_cog = self.get_audio_cog()
        return getattr(audio_cog, "lavalink", None) if audio_cog else None

    def is_lavalink_ready(self):
        """Check if Lavalink is ready for use"""
        try:
            audio_cog = self.get_audio_cog()
            if not audio_cog:
                tllogger.debug("[AprilAI] Audio cog not found")
                return False
                
            if not hasattr(audio_cog, 'lavalink') or not audio_cog.lavalink:
                tllogger.debug("[AprilAI] Lavalink client not available")
                return False
                
            lavalink = audio_cog.lavalink
            if not hasattr(lavalink, 'node_manager') or not lavalink.node_manager:
                tllogger.debug("[AprilAI] Node manager not available")
                return False
            
            # Check if we have any available nodes
            try:
                nodes = getattr(lavalink.node_manager, 'nodes', [])
                if not nodes:
                    tllogger.debug("[AprilAI] No nodes configured")
                    return False
                    
                available_nodes = [n for n in nodes if hasattr(n, 'is_available') and n.is_available()]
                if not available_nodes:
                    tllogger.debug(f"[AprilAI] No available nodes (have {len(nodes)} total)")
                    return False
                    
                # Check player manager
                if not hasattr(lavalink, 'player_manager') or not lavalink.player_manager:
                    tllogger.debug("[AprilAI] Player manager not available")
                    return False
                    
                return True
                
            except Exception as e:
                tllogger.debug(f"[AprilAI] Error checking node availability: {e}")
                return False
                
        except Exception as e:
            tllogger.debug(f"[AprilAI] Error in lavalink readiness check: {e}")
            return False

    async def get_player(self, guild_id: int):
        """Get or create player for a guild"""
        lavalink = self.get_lavalink()
        if not lavalink or not lavalink.player_manager:
            tllogger.warning("Lavalink player manager not available")
            return None
            
        try:
            player = lavalink.player_manager.get(guild_id)
            if not player:
                player = lavalink.player_manager.create(guild_id)
                tllogger.debug(f"Created new player for guild {guild_id}")
            return player
        except Exception as e:
            tllogger.error(f"Player creation/retrieval failed: {e}")
            return None

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
        
        # Check if Audio cog exists first
        audio_cog = self.get_audio_cog()
        if not audio_cog:
            return await ctx.send("❌ Audio cog is not loaded. Please load it with `[p]load audio`")
        
        # Check if Lavalink is ready
        if not self.is_lavalink_ready():
            # Provide more specific error information
            lavalink = self.get_lavalink()
            if not lavalink:
                return await ctx.send("❌ Lavalink client not initialized. Please ensure Audio cog is properly configured.")
            elif not hasattr(lavalink, 'node_manager') or not lavalink.node_manager:
                return await ctx.send("❌ Lavalink node manager not available. Please check Audio cog configuration.")
            else:
                try:
                    nodes = getattr(lavalink.node_manager, 'nodes', [])
                    if not nodes:
                        return await ctx.send("❌ No Lavalink nodes configured. Please configure Audio cog with `[p]audioset`")
                    else:
                        available = [n for n in nodes if hasattr(n, 'is_available') and n.is_available()]
                        return await ctx.send(f"❌ No available Lavalink nodes ({len(available)}/{len(nodes)} ready). Please check your Lavalink server.")
                except:
                    return await ctx.send("❌ Error checking Lavalink status. Please restart the Audio cog.")
        
        # Check user voice state
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a voice channel.")
        
        channel = ctx.author.voice.channel
        permissions = channel.permissions_for(ctx.me)
        if not permissions.connect or not permissions.speak:
            return await ctx.send("❌ I need permissions to connect and speak!")
        
        # Get player
        player = await self.get_player(ctx.guild.id)
        if not player:
            return await ctx.send("❌ Failed to create player. Please try again.")
        
        try:
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
            tllogger.exception("[AprilAI] Failed join_voice")
            await ctx.send(f"❌ Join failed: {e}")

    async def leave_voice(self, ctx):
        """Leave voice channel"""
        if not self.is_lavalink_ready():
            return await ctx.send("❌ Audio system not ready.")
        
        player = await self.get_player(ctx.guild.id)
        if player and player.is_connected:
            try:
                await player.stop()
                await player.disconnect()
                await ctx.send("👋 Disconnected")
            except Exception as e:
                tllogger.error(f"Disconnect failed: {e}")
                await ctx.send(f"❌ Disconnect failed: {e}")
        else:
            await ctx.send("❌ Not in a voice channel.")

    async def process_query(self, ctx, input_text):
        # Check if we should use voice
        use_voice = False
        if self.is_lavalink_ready() and await self.config.tts_enabled():
            player = await self.get_player(ctx.guild.id)
            use_voice = player and player.is_connected
        
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
        """Generate TTS and play through Lavalink"""
        tts_key = await self.config.tts_key()
        if not tts_key:
            tllogger.warning("Skipping TTS: missing API key.")
            return
        
        # Get player
        player = await self.get_player(ctx.guild.id)
        if not player or not player.is_connected:
            tllogger.warning("Skipping TTS: Player not connected.")
            return
        
        # Clean text for TTS (remove markdown formatting, etc.)
        clean_text = self.clean_text_for_tts(text)
        if not clean_text.strip():
            return
        
        # Split text into manageable chunks
        chunks = self.split_text_for_tts(clean_text, max_length=1000)
        
        try:
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                # Generate TTS audio
                audio_data = await self.generate_tts_audio(chunk, tts_key)
                if not audio_data:
                    continue
                
                # Create temporary file
                temp_file = await self.create_temp_audio_file(audio_data)
                if not temp_file:
                    continue
                
                # Create a simple HTTP-accessible URL or use data URI
                track_uri = await self.create_playable_uri(temp_file, audio_data)
                if not track_uri:
                    continue
                
                # Get tracks from Lavalink
                try:
                    result = await self.get_lavalink().get_tracks(track_uri)
                    if result and result.tracks:
                        track = result.tracks[0]
                        player.add(requester=ctx.author.id, track=track)
                        tllogger.debug(f"Added TTS track {i+1}/{len(chunks)} to queue")
                    else:
                        tllogger.warning(f"No tracks found for URI: {track_uri}")
                except Exception as e:
                    tllogger.error(f"Failed to load track: {e}")
            
            # Start playing if not already playing
            if not player.is_playing and player.queue:
                await player.play()
                
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
        
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def split_text_for_tts(self, text: str, max_length: int = 1000) -> list:
        """Split text into chunks suitable for TTS"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence + '. ') <= max_length:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

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
            
            async with self.session.post(url, json=payload, headers=headers, timeout=30) as resp:
                if resp.status == 200:
                    return await resp.read()
                else:
                    error_text = await resp.text()
                    tllogger.error(f"TTS API error {resp.status}: {error_text}")
                    return None
                    
        except Exception as e:
            tllogger.error(f"TTS generation failed: {e}")
            return None

    async def create_temp_audio_file(self, audio_data: bytes) -> str:
        """Create temporary audio file"""
        try:
            # Create unique filename
            filename = f"tts_{int(time.time())}_{random.randint(0, 10000)}.mp3"
            filepath = self.tts_dir / filename
            
            # Write audio data
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            self.tts_files.add(str(filepath))
            return str(filepath)
            
        except Exception as e:
            tllogger.error(f"Failed to create temp audio file: {e}")
            return None

    async def create_playable_uri(self, filepath: str, audio_data: bytes) -> str:
        """Create a URI that Lavalink can play"""
        # Option 1: Try data URI (may not work with all Lavalink setups)
        try:
            import base64
            encoded_data = base64.b64encode(audio_data).decode('utf-8')
            data_uri = f"data:audio/mpeg;base64,{encoded_data}"
            return data_uri
        except:
            pass
        
        # Option 2: If you have a web server configured, serve the file
        server_url = await self.config.tts_server_url()
        if server_url:
            filename = os.path.basename(filepath)
            return f"{server_url.rstrip('/')}/{filename}"
        
        # Option 3: Try file URI as fallback (unlikely to work)
        return f"file://{filepath}"

    @commands.Cog.listener()
    async def on_red_audio_track_end(self, guild, track, reason):
        """Clean up TTS files after playback"""
        if self._unloading:
            return
        
        # Clean up based on track info
        if hasattr(track, 'uri') and track.uri:
            # Handle data URIs or file URIs
            if track.uri.startswith("file://"):
                file_path = track.uri[7:]  # Remove file:// prefix
                await self.cleanup_tts_file(file_path)
            elif "tts_" in str(track.uri):
                # Try to find matching file for cleanup
                timestamp = int(time.time())
                for filepath in list(self.tts_files):
                    try:
                        # Clean up files older than 5 minutes
                        file_age = timestamp - os.path.getmtime(filepath)
                        if file_age > 300:  # 5 minutes
                            await self.cleanup_tts_file(filepath)
                    except:
                        pass

    async def cleanup_tts_file(self, filepath: str):
        """Clean up a single TTS file"""
        if filepath in self.tts_files:
            try:
                if os.path.exists(filepath):
                    os.unlink(filepath)
                self.tts_files.discard(filepath)
                tllogger.debug(f"Cleaned up TTS file: {filepath}")
            except Exception as e:
                tllogger.error(f"Error cleaning up TTS file {filepath}: {e}")

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
    async def ttsserver(self, ctx, url: str = None):
        """Set TTS server URL for serving audio files to Lavalink"""
        if url is None:
            current = await self.config.tts_server_url()
            await ctx.send(f"Current TTS server URL: `{current or 'Not set'}`")
        else:
            await self.config.tts_server_url.set(url)
            await ctx.send(f"✅ TTS server URL set to `{url}`")

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
        
        # Check Audio cog
        audio_cog = self.get_audio_cog()
        embed.add_field(name="Audio Cog", value="✅ Loaded" if audio_cog else "❌ Not loaded", inline=True)
        
        if not audio_cog:
            embed.add_field(name="Fix", value="Load with `[p]load audio`", inline=False)
            return await ctx.send(embed=embed)
        
        # Check Lavalink client
        has_lavalink = hasattr(audio_cog, 'lavalink')
        lavalink = getattr(audio_cog, 'lavalink', None)
        embed.add_field(name="Lavalink Client", value="✅ Available" if has_lavalink and lavalink else "❌ Not available", inline=True)
        
        if not (has_lavalink and lavalink):
            embed.add_field(name="Fix", value="Configure Audio cog with `[p]audioset`", inline=False)
            return await ctx.send(embed=embed)
        
        # Check node manager
        has_node_manager = hasattr(lavalink, 'node_manager') and lavalink.node_manager
        embed.add_field(name="Node Manager", value="✅ Available" if has_node_manager else "❌ Not available", inline=True)
        
        if not has_node_manager:
            embed.add_field(name="Fix", value="Check Lavalink server connection", inline=False)
            return await ctx.send(embed=embed)
        
        # Check nodes
        try:
            nodes = getattr(lavalink.node_manager, 'nodes', [])
            embed.add_field(name="Total Nodes", value=str(len(nodes)), inline=True)
            
            if nodes:
                available_nodes = []
                for i, node in enumerate(nodes):
                    try:
                        is_available = hasattr(node, 'is_available') and node.is_available()
                        status = "✅" if is_available else "❌"
                        node_info = f"Node {i+1}: {status}"
                        if hasattr(node, 'host') and hasattr(node, 'port'):
                            node_info += f" ({node.host}:{node.port})"
                        embed.add_field(name=f"Node {i+1}", value=node_info, inline=True)
                        if is_available:
                            available_nodes.append(node)
                    except Exception as e:
                        embed.add_field(name=f"Node {i+1}", value=f"❌ Error: {e}", inline=True)
                
                embed.add_field(name="Available Nodes", value=str(len(available_nodes)), inline=True)
            else:
                embed.add_field(name="Fix", value="No nodes configured. Check `[p]audioset`", inline=False)
                
        except Exception as e:
            embed.add_field(name="Node Check Error", value=str(e), inline=False)
        
        # Check player manager
        has_player_manager = hasattr(lavalink, 'player_manager') and lavalink.player_manager
        embed.add_field(name="Player Manager", value="✅ Available" if has_player_manager else "❌ Not available", inline=True)
        
        # Overall status
        overall_ready = self.is_lavalink_ready()
        embed.add_field(name="Overall Status", value="✅ Ready" if overall_ready else "❌ Not Ready", inline=True)
        embed.add_field(name="Init Status", value="✅ Complete" if self._lavalink_ready else "❌ In Progress", inline=True)
        
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
        e.add_field(name="TTS Server URL", value=f"`{cfg['tts_server_url']}`" if cfg['tts_server_url'] else "❌ Not set", inline=False)
        
        # Lavalink status
        status_emoji = "✅" if self.is_lavalink_ready() else "❌"
        ready_status = "Ready" if self._lavalink_ready else "Not Ready"
        e.add_field(name="Lavalink Status", value=f"{status_emoji} {ready_status}", inline=True)
        
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
        if member.bot or not self.is_lavalink_ready():
            return
            
        player = await self.get_player(member.guild.id)
        if player and player.is_connected:
            # Check if only bot remains in voice
            try:
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