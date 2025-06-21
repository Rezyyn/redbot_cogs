from .lokilogger import LokiLogger

async def setup(bot):
    cog = LokiLogger(bot)
    await bot.add_cog(cog)