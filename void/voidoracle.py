import random
import discord
from redbot.core import commands, Config
from datetime import datetime

class VoidOracle(commands.Cog):
    """Ask the void at different existential levels"""
    
    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=123456789)
        self.config.register_guild(
            response_level=2  # Default to level 2
        )
        self.last_question_time = {}
        
        # Level 1: kinda fucked
        self.level1_responses = [
            "Your existence is a fleeting anomaly in a universe that barely notices.",
            "The universe is a cold, uncaring mother.  It will eventually reclaim you.",
            "You are a collection of stardust briefly arranged into a vaguely sentient shape.",
            "Happiness is just a chemical trick your brain plays to keep you reproducing.",
            "Everything you love will eventually decay.  Enjoy it while it lasts... or don't."
        ]
        
        # Level 2: really fucked
        self.level2_responses = [
            "The void remembers all your embarrassing moments. It judges you harshly.",
            "Free will is an illusion. You are merely a puppet dancing to the tune of entropy.",
            "Your consciousness is a fragile bubble, easily burst by the slightest inconvenience.",
            "The meaning of life?  To delay the inevitable heat death of the universe, obviously.",
            "You are a cosmic accident, destined to be forgotten.  Embrace the nihilism."
        ]
        
        # Level 3: level 3 nsfw unhinged responses, no holds barred 
        self.level3_responses = [
            "The void has seen things... things you wouldn't believe.  Mostly regret and lukewarm coffee.",
            "Your soul is a particularly pungent cheese, slowly ripening in the cosmic fridge.",
            "Love is just a desperate attempt to avoid the crushing loneliness of existence.  Good luck with that.",
            "The universe is actively mocking your attempts at self-improvement.",
            "You are a meatbag piloting a fragile skeleton, hurtling through the void.  Don't overthink it."
        ]

    @commands.command(name="void", aliases=["askthevoid", "oracle"])
    async def void_oracle(self, ctx, *, question: str):
        """
        Ask the void a question at the current response level
        
        Example: [p]void Will I find happiness?
        """
        # Cooldown check
        user_id = ctx.author.id
        now = datetime.utcnow()
        
        if user_id in self.last_question_time:
            last_time = self.last_question_time[user_id]
            cooldown = (now - last_time).total_seconds()
            if cooldown < 30:
                await ctx.send(f"🕒 The void contemplates slowly. Try again in {30-int(cooldown)} seconds. It's busy judging your life choices.")
                return
        
        self.last_question_time[user_id] = now
        
        # Get current response level
        level = await self.config.guild(ctx.guild).response_level()
        
        # Select response based on level
        if level == 1:
            response = random.choice(self.level1_responses)
        elif level == 2:
            response = random.choice(self.level2_responses)
        else:  # level 3
            response = random.choice(self.level3_responses)
        
        # Send response
        await ctx.send(f"🔮 **The void responds (Level {level}):** {response}")

    @commands.command()
    @commands.has_permissions(manage_guild=True)
    async def setvoidlevel(self, ctx, level: int):
        """
        Set the void's response depth (1-3)
        
        Level 1: Simple responses
        Level 2: Moderate existential
        Level 3: Deep existential/cryptic
        """
        if level < 1 or level > 3:
            await ctx.send("⚠️ Level must be between 1 and 3.  The void is displeased with your incompetence.")
            return
            
        await self.config.guild(ctx.guild).response_level.set(level)
        await ctx.send(f"🌌 Void response level set to **{level}**. Prepare for existential dread.")

    @commands.command()
    @commands.is_owner()
    async def addvoidresponse(self, ctx, level: int, *, response: str):
        """
        Add a custom void response to a level (Owner only)
        
        Example: [p]addvoidresponse 2 "The cosmos remains indifferent"
        """
        if level < 1 or level > 3:
            await ctx.send("⚠️ Level must be between 1 and 3.  The void demands precision!")
            return
            
        if level == 1:
            self.level1_responses.append(response)
        elif level == 2:
            self.level2_responses.append(response)
        else:
            self.level3_responses.append(response)
            
        await ctx.send(f"🗒️ Added level {level} response: \"{response}\" The void acknowledges your contribution... begrudgingly.")

    @commands.command()
    async def voidlevel(self, ctx):
        """Show the current void response level"""
        level = await self.config.guild(ctx.guild).response_level()
        await ctx.send(f"🔮 Current void response level: **{level}**.  Brace yourself.")
 # some fucked up thoughts need populated here
    @commands.command()
    async def contemplate(self, ctx):
        """Receive a random existential thought"""
        thoughts = [
            "If trees fall in the forest and no one is around to hear, are they screaming in silent agony?",
            "We are all just temporary arrangements of atoms, destined to return to the cosmic soup.",
            "The universe doesn't care about your feelings. It just *is*.",
            "Is reality just a shared hallucination? And if so, who's dreaming us?",
            "The only constant is change, and change is terrifying.",
            "Your greatest fear is already happening, you just haven't noticed it yet.",
            "The void is hungry. And it's looking at you."
        ]
        await ctx.send(f"💭 **Existential Thought:** {random.choice(thoughts)}")