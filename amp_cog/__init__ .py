from .ampservers import AMPServers


async def setup(bot):
    await bot.add_cog(AMPServers(bot))
