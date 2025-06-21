import random
from redbot.core import commands

class Magic8Ball(commands.Cog):
    """Ask the magic 8-ball a question!"""

    def __init__(self, bot):
        self.bot = bot
        self.responses = [
            "It is certain.", "It is decidedly so.", "Without a doubt.",
            "Yes - definitely.", "You may rely on it.", "As I see it, yes.",
            "Most likely.", "Outlook good.", "Yes.", "Signs point to yes.",
            "Reply hazy, try again.", "Ask again later.", "Better not tell you now.",
            "Cannot predict now.", "Concentrate and ask again.", "Don't count on it.",
            "My reply is no.", "My sources say no.", "Outlook not so good.",
            "Very doubtful."
        ]

    @commands.command(name="8ball", aliases=["eightball"])
    async def magic8ball(self, ctx, *, question: str):
        """
        Ask the magic 8-ball a question
        
        Example: [p]8ball Will I become rich tomorrow?
        """
        # Select random response
        answer = random.choice(self.responses)
        # Format and send response
        await ctx.send(f"🎱 **Question:** {question}\n**Answer:** {answer}")