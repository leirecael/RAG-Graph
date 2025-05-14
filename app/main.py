import sys
import os
import asyncio

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from presentation.gui import start_interface

async def main():
    await start_interface()

if __name__ == "__main__":
    asyncio.run(main())