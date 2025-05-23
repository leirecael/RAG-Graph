import asyncio

from presentation.gui import start_interface

async def main():
    await start_interface()

if __name__ == "__main__":
    asyncio.run(main())