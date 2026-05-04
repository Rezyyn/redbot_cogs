from .craftyserver import CraftyServer

async def setup(bot):
    await bot.add_cog(CraftyServer(bot))
