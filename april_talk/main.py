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
from io import BytesIO
from redbot.core import commands, Config
from redbot.core.bot import Red
from redbot.core.data_manager import cog_data_path
from redbot.core.utils.chat_formatting import pagify
from redbot.core.utils.menus import menu, DEFAULT_CONTROLS
from typing import Optional, List, Dict

# Logger
tllogger = logging.getLogger("red.aprilai")
tllogger.setLevel(logging.DEBUG)

# Default style suffix for images
STYLE_SUFFIX = ", in a futuristic neo-cyberpunk aesthetic"

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
            system_prompt="You are April, a helpful AI assistant for Discord. "
                          "Default to useful answers. Be concise and kind.",
            smert_prompt="You are April in 'smert' mode - an incredibly intelligent, witty, and creative AI assistant with deep knowledge across all domains.",
            tts_enabled=True,
            text_response_when_voice=True,
            max_history=5,  # Default 5 exchanges
            use_gifs=True,
            max_message_length=1800,  # For splitting long messages
            openai_key="",  # for Draw image generation
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

    # -----------------------
    # Prompt helpers
    # -----------------------
    def style_prompt(self, prompt: str) -> str:
        """Bias every image to neo-cyberpunk/futuristic unless the user explicitly sets a style."""
        p = prompt.strip()
        if any(k in p.lower() for k in ["cyberpunk", "synthwave", "futuristic", "sci-fi", "science fiction", "blade runner"]):
            return p
        return p + STYLE_SUFFIX

    def maybe_extract_draw_prompt(self, text: str) -> Optional[str]:
        """Extracts a single <draw>...</draw> block; returns inner prompt or None."""
        match = re.search(r"<draw>(.*?)</draw>", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        raw = match.group(1).strip()
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw or None

    # -----------------------
    # Lavalink helpers
    # -----------------------
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
            if hasattr(lavalink, 'get_player'):
                player = lavalink.get_player(guild_id)
                tllogger.debug(f"Got player for guild {guild_id}: {player}")
                return player
            else:
                tllogger.error("lavalink.get_player not available")
                return None
        except Exception as e:
            tllogger.error(f"Failed to get player for guild {guild_id}: {e}")
            return None

    def is_player_connected(self, player) -> bool:
        """Check if player is connected"""
        if not player:
            return False
        
        try:
            if hasattr(player, 'is_connected') and callable(player.is_connected):
                return player.is_connected()
            elif hasattr(player, 'is_connected'):
                return player.is_connected
            elif hasattr(player, 'channel_id'):
                return bool(player.channel_id)
            elif hasattr(player, 'channel'):
                return bool(player.channel)
            else:
                return False
        except Exception as e:
            tllogger.error(f"Error checking player connection: {e}")
            return False

    def get_player_channel_id(self, player):
        """Get the channel ID the player is connected to"""
        if not player:
            return None
            
        try:
            if hasattr(player, 'channel_id'):
                return player.channel_id
            elif hasattr(player, 'channel') and player.channel:
                return player.channel.id
            else:
                return None
        except Exception as e:
            tllogger.error(f"Error getting player channel ID: {e}")
            return None

    # -----------------------
    # Commands
    # -----------------------
    @commands.group(name="april", invoke_without_command=True)
    @commands.cooldown(1, 15, commands.BucketType.user)
    async def april(self, ctx, *, input: str):
        """Main April command"""
        cmd = input.strip()

        # special trigger (case-insensitive, tolerant)
        if cmd.lower().startswith("i know what you're thinking"):
            return await self._draw_what_you_think(ctx)

        # Handle special short text commands
        low = cmd.lower()
        if low == "join":
            return await self.join_voice(ctx)
        if low == "leave":
            return await self.leave_voice(ctx)
        if low == "clearhistory":
            return await self.clear_history(ctx)
        if low == "smert":
            return await self.toggle_smert_mode(ctx)
            
        tllogger.debug(f"Command april: {input} by {ctx.author}")
        await self.process_query(ctx, input)

    @april.command(name="draw")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def april_draw(self, ctx: commands.Context, *, prompt: str):
        """
        Draw an image with OpenAI (group command).
        Usage:
          .april draw a neon fox in the rain
          .april draw me a neon fox in the rain
        """
        p = prompt.strip()
        if p.lower().startswith("me "):
            p = p[3:].strip()

        styled_prompt = self.style_prompt(p)

        try:
            async with ctx.typing():
                png_bytes = await self.generate_openai_image_png(styled_prompt, size="1024x1024")
                file = discord.File(BytesIO(png_bytes), filename="april_draw.png")
                await ctx.send(content=f"**Prompt:** {styled_prompt}", file=file)
        except Exception as e:
            tllogger.exception("April group draw failed")
            await ctx.send(
                f"⚠️ I couldn't draw that: `{e}`\n"
                "Set the key with `[p]aprilconfig openaikey <key>` if needed."
            )

    @april.command(name="smert")
    async def toggle_smert_mode(self, ctx):
        """Toggle smert mode (requires Anthropic API key)"""
        user_config = self.config.user(ctx.author)
        current_mode = await user_config.smert_mode()
        
        if not current_mode:
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
        
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("❌ You must be in a voice channel.")
        
        channel = ctx.author.voice.channel
        permissions = channel.permissions_for(ctx.me)
        if not permissions.connect or not permissions.speak:
            return await ctx.send("❌ I need permissions to connect and speak!")
        
        try:
            if not hasattr(lavalink, 'get_player'):
                return await ctx.send("❌ Lavalink is not initialized. Please load the Audio cog first with `[p]load audio`")
            
            player = self.get_player(ctx.guild.id)
            
            if player and self.is_player_connected(player):
                current_channel_id = self.get_player_channel_id(player)
                if current_channel_id == channel.id:
                    return await ctx.send(f"✅ Already connected to {channel.name}")
                else:
                    await player.disconnect()
            
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

    # -----------------------
    # Core flow
    # -----------------------
    async def process_query(self, ctx, input_text):
        # Should we use voice?
        use_voice = False
        if await self.config.tts_enabled():
            try:
                player = self.get_player(ctx.guild.id)
                use_voice = player and self.is_player_connected(player)
                tllogger.debug(f"TTS check - player: {player}, connected: {use_voice}")
            except:
                use_voice = False
        
        # Smert mode?
        user_config = self.config.user(ctx.author)
        smert_mode = await user_config.smert_mode()
        
        tllogger.debug(f"process_query use_voice={use_voice}, smert_mode={smert_mode}")
        
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
                
                # Detect <draw>...</draw> and remove it from the displayed text
                draw_prompt = self.maybe_extract_draw_prompt(resp)
                clean_resp = re.sub(r"<draw>.*?</draw>", "", resp, flags=re.IGNORECASE | re.DOTALL).strip()

                # Update history with the clean text
                self.history[channel_id].append({"role": "user", "content": input_text})
                self.history[channel_id].append({"role": "assistant", "content": clean_resp})
                
                tasks = []

                # Send text response (unless you mute it when voice-only)
                if not (use_voice and not await self.config.text_response_when_voice()):
                    tasks.append(asyncio.create_task(self.send_streamed_response(ctx, clean_resp)))
                
                # Voice TTS if connected
                if use_voice:
                    tasks.append(asyncio.create_task(self.speak_response(ctx, clean_resp)))

                # If April suggested a draw, render image inline
                if draw_prompt:
                    styled = self.style_prompt(draw_prompt)
                    async def _draw_and_send():
                        try:
                            png = await self.generate_openai_image_png(styled, size="1024x1024")
                            file = discord.File(BytesIO(png), filename="april_draw.png")
                            await ctx.send(content=f"**Inline image suggestion:** {styled}", file=file)
                        except Exception as e:
                            tllogger.exception("Inline draw failed")
                            await ctx.send(f"⚠️ Couldn't render the suggested image: `{e}`")
                    tasks.append(asyncio.create_task(_draw_and_send()))
                
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
        if emotion and random.random() < 0.3:
            gif_url = random.choice(EMOTION_GIFS[emotion])
        
        # Split long messages
        if len(resp) > max_length:
            embed = discord.Embed(
                description="💭 *Thinking...*",
                color=await ctx.embed_color()
            )
            thinking_msg = await ctx.send(embed=embed)
            
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
            
            first_chunk = chunks[0]
            if gif_url and len(chunks) == 1:
                await thinking_msg.edit(content=first_chunk + f"\n{gif_url}", embed=None)
            else:
                await thinking_msg.edit(content=first_chunk, embed=None)
            
            for i, chunk in enumerate(chunks[1:], 1):
                await asyncio.sleep(0.5)
                if gif_url and i == len(chunks) - 1:
                    await ctx.send(chunk + f"\n{gif_url}")
                else:
                    await ctx.send(chunk)
        else:
            embed = discord.Embed(
                description="💭 *Thinking...*",
                color=await ctx.embed_color()
            )
            msg = await ctx.send(embed=embed)
            
            chunk_size = 50
            for i in range(0, len(resp), chunk_size):
                chunk = resp[:i + chunk_size]
                if i + chunk_size >= len(resp) and gif_url:
                    await msg.edit(content=chunk + f"\n{gif_url}", embed=None)
                else:
                    await msg.edit(content=chunk + "▌", embed=None)
                await asyncio.sleep(0.05)
            
            if gif_url:
                await msg.edit(content=resp + f"\n{gif_url}", embed=None)
            else:
                await msg.edit(content=resp, embed=None)

    async def speak_response(self, ctx, text: str):
        """Generate TTS, play via local command, and clean up"""
        tts_key = await self.config.tts_key()
        if not tts_key:
            tllogger.warning("Skipping TTS: missing API key.")
            return

        try:
            player = self.get_player(ctx.guild.id)
            if not self.is_player_connected(player):
                tllogger.warning("Skipping TTS: Player not connected.")
                return

            clean_text = self.clean_text_for_tts(text)
            if not clean_text.strip():
                return

            audio_data = await self.generate_tts_audio(clean_text, tts_key)
            if not audio_data:
                tllogger.error("Failed to generate TTS audio")
                return

            localtrack_dir = cog_data_path(self).parent / "Audio" / "localtracks" / "april_tts"
            localtrack_dir.mkdir(parents=True, exist_ok=True)

            filename = f"tts_{int(time.time())}_{random.randint(1000, 9999)}.mp3"
            filepath = localtrack_dir / filename

            with open(filepath, 'wb') as f:
                f.write(audio_data)

            tllogger.debug(f"TTS audio saved: {filepath}")

            audio_cog = self.bot.get_cog("Audio")
            if audio_cog:
                play_command = audio_cog.command_play
                await play_command(ctx, query=f"localtracks/april_tts/{filename}")
            else:
                tllogger.error("Audio cog not found")
                return

            async def delayed_delete():
                await asyncio.sleep(30)
                try:
                    if filepath.exists():
                        filepath.unlink()
                        tllogger.debug(f"TTS file deleted: {filepath.name}")
                except Exception as e:
                    tllogger.error(f"Failed to delete TTS file: {filepath.name} — {e}")

            asyncio.create_task(delayed_delete())

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
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 1000:
            text = text[:1000].rsplit(' ', 1)[0] + "..."
        return text

    # -----------------------
    # TTS (ElevenLabs)
    # -----------------------
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

    # -----------------------
    # OpenAI IMAGE GENERATION
    # -----------------------
    async def generate_openai_image_png(self, prompt: str, size: str = "1024x1024") -> bytes:
        """
        Generate an image with OpenAI Images API and return PNG bytes.
        Uses model 'gpt-image-1' and returns the first image.
        """
        key = await self.config.openai_key()
        if not key:
            raise RuntimeError("OpenAI API key not set. Use `[p]aprilconfig openaikey <key>`.")

        url = "https://api.openai.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "size": size,
        }

        tllogger.debug(f"OpenAI image request for prompt: {prompt[:120]}...")
        async with self.session.post(url, json=payload, headers=headers, timeout=90) as r:
            if r.status != 200:
                try:
                    err = await r.json()
                except:
                    err = {"error": {"message": await r.text()}}
                message = err.get("error", {}).get("message", f"HTTP {r.status}")
                raise RuntimeError(f"OpenAI Images API error: {message}")

            data = await r.json()
            b64 = data["data"][0]["b64_json"]
            return base64.b64decode(b64)

    # -----------------------
    # DeepSeek helpers
    # -----------------------
    async def deepseek_image_prompt(self, messages: list) -> str:
        """
        Ask DeepSeek to produce a single-line image prompt (no commentary).
        Reuses your existing DeepSeek chat API.
        """
        sys = {
            "role": "system",
            "content": (
                "You write exactly ONE single-line image prompt suitable for a text-to-image model. "
                "No commentary, no markdown, no quotes. Keep it concise and vivid. "
                "Default to a futuristic neo-cyberpunk aesthetic unless the user requests otherwise."
            )
        }
        crafted = await self.query_deepseek(
            user_id=0,
            messages=[sys] + messages[-10:]
        )
        crafted = re.sub(r"\s+", " ", crafted).strip()
        return crafted

    # -----------------------
    # Special flow
    # -----------------------
    async def _draw_what_you_think(self, ctx: commands.Context):
        """
        Path: DeepSeek chat for reply -> DeepSeek image prompt -> OpenAI image -> post text + image.
        Uses recent channel history for context.
        """
        async with ctx.typing():
            try:
                channel_id = ctx.channel.id
                max_history = await self.config.max_history()
                if channel_id not in self.history:
                    self.history[channel_id] = deque(maxlen=max_history * 2)

                system_prompt = await self.config.system_prompt()
                messages = [{"role": "system", "content": system_prompt}]
                messages.extend(self.history[channel_id])
                messages.append({"role": "user", "content": "User: I know what you're thinking, draw me it!"})

                # 1) Chat reply (DeepSeek)
                chat_resp = await self.query_deepseek(ctx.author.id, messages)

                # 2) DeepSeek crafts a single-line image prompt
                img_prompt = await self.deepseek_image_prompt(messages + [{"role": "assistant", "content": chat_resp}])

                # 3) OpenAI renders the image
                styled = self.style_prompt(img_prompt)
                png = await self.generate_openai_image_png(styled, size="1024x1024")

                # 4) Post both
                file = discord.File(BytesIO(png), filename="april_draw.png")
                await ctx.send(content=f"{chat_resp}\n\n**Image:** {styled}", file=file)

                # Update history
                self.history[channel_id].append({"role": "user", "content": "I know what you're thinking, draw me it!"})
                self.history[channel_id].append({"role": "assistant", "content": chat_resp + f"\n[Image: {styled}]"})

            except Exception as e:
                tllogger.exception("draw-what-you-think flow failed")
                await ctx.send(f"⚠️ Couldn't complete the draw-what-you-think flow: `{e}`")

    # -----------------------
    # Top-level DRAW command (kept)
    # -----------------------
    @commands.command(name="draw")
    @commands.cooldown(1, 10, commands.BucketType.user)
    async def draw_command(self, ctx: commands.Context, *, prompt: str):
        """Generate an image from a prompt. Usage: [p]draw a neon fox in the rain"""
        p = prompt.strip()
        styled_prompt = self.style_prompt(p)
        try:
            async with ctx.typing():
                png_bytes = await self.generate_openai_image_png(styled_prompt, size="1024x1024")
                file = discord.File(BytesIO(png_bytes), filename="april_draw.png")
                await ctx.send(content=f"**Prompt:** {styled_prompt}", file=file)
        except Exception as e:
            tllogger.exception("Draw command failed")
            await ctx.send(
                f"⚠️ I couldn't draw that: `{e}`\n"
                "Set the API key with `[p]aprilconfig openaikey <key>` if needed."
            )

    # -----------------------
    # Config commands
    # -----------------------
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
    async def openaikey(self, ctx, key: str):
        """Set OpenAI API key for image generation."""
        await self.config.openai_key.set(key)
        await ctx.tick()
        try:
            await ctx.message.delete()
        except:
            pass

    @commands.command()
    async def play_tts(self, ctx, filename: str):
        """Play a TTS file from the april_tts folder."""
        audio_cog = self.bot.get_cog("Audio")
        if not audio_cog:
            await ctx.send("Audio cog not loaded.")
            return
        query = f"april_tts/{filename}"
        await ctx.invoke(audio_cog.play, query=query)

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
            if hasattr(lavalink, 'get_player'):
                embed.add_field(name="Lavalink Status", value="✅ Initialized", inline=True)
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
        
        embed.add_field(name="Guild ID", value=f"`{ctx.guild.id}`", inline=True)
        embed.add_field(name="Bot Voice State", value="✅ Connected" if ctx.guild.voice_client else "❌ Not connected", inline=True)
        
        await ctx.send(embed=embed)

    @aprilconfig.command(name="settings")
    async def show_settings(self, ctx):
        """Show current configuration"""
        cfg = await self.config.all()
        e = discord.Embed(title="AprilAI Configuration", color=await ctx.embed_color())
        
        deepseek_key = cfg['deepseek_key']
        anthropic_key = cfg['anthropic_key']
        tts_key = cfg['tts_key']
        openai_key = cfg.get('openai_key', '')
        
        e.add_field(name="DeepSeek Key", value=f"`...{deepseek_key[-4:]}`" if deepseek_key else "❌ Not set", inline=False)
        e.add_field(name="Anthropic Key", value=f"`...{anthropic_key[-4:]}`" if anthropic_key else "❌ Not set", inline=False)
        e.add_field(name="ElevenLabs Key", value=f"`...{tts_key[-4:]}`" if tts_key else "❌ Not set", inline=False)
        e.add_field(name="OpenAI Key", value=f"`...{openai_key[-4:]}`" if openai_key else "❌ Not set", inline=False)
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

    # -----------------------
    # DeepSeek / Anthropic
    # -----------------------
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
                timeout=60
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
        
        system_content = None
        conversation_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                conversation_messages.append(msg)
        
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": conversation_messages,
            "max_tokens": await self.config.max_tokens(),
            "temperature": await self.config.temperature()
        }
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

    # -----------------------
    # Voice events
    # -----------------------
    @commands.Cog.listener()
    async def on_voice_state_update(self, member, before, after):
        """Handle voice state updates"""
        if member.bot:
            return
            
        try:
            player = self.get_player(member.guild.id)
            if player and self.is_player_connected(player):
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
