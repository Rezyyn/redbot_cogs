from .lokibackfill import LokiBackfill

async def setup(bot):
    await bot.add_cog(LokiBackfill(bot))
