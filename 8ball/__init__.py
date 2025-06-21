from .magic8ball import Magic8Ball

async def setup(bot):
    await bot.add_cog(Magic8Ball(bot))