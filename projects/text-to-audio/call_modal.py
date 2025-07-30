import httpx
import soundfile as sf
import asyncio
import os
from dotenv import load_dotenv
# Make sure to set your .env variables!

load_dotenv()

prompt = "A soft calming chord that glistens over time"
async def do_inference():
    headers = {
    "Authorization": f"Bearer {os.environ["INFERENCE_SECRET"]}",
    }
    async with httpx.AsyncClient(timeout=360.0) as client:
            response = await client.post(os.environ["MODAL_API"],json={"prompt": prompt}, headers=headers)
            if response.status_code == 200:
                with open("out.wav", "wb") as f:
                    f.write(response.content)
            else:
                print("Error:", response.status_code, response.text)

asyncio.run(do_inference())