from .stream_controller import StreamController


async def setup(bot):
    cog = StreamController(bot)
    await bot.add_cog(cog)
