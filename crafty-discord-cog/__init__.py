from .craftyserver import CraftyServer

__red_end_user_data_statement__ = "This cog does not store persistent user data."

async def setup(bot):
    await bot.add_cog(CraftyServer(bot))
