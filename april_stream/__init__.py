from .stream_controller import StreamController


async def setup(bot):
    await bot.add_cog(StreamController(bot))