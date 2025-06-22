from .voidoracle import VoidOracle

async def setup(bot):
    await bot.add_cog(VoidOracle(bot))