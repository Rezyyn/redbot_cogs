from .reactionstats import ReactionStats

async def setup(bot):
    await bot.add_cog(ReactionStats(bot))