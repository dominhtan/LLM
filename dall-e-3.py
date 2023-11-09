#openai ver: 1.1

import os
from openai import OpenAI
from pathlib import Path
import io
import time


client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="###",
)
# DATA_DIR = Path("D:\Lam_Viec\Python\Reports").parent / "AI-Image"
# DATA_DIR.mkdir(exist_ok=True)
PROMPT = "A shop in an e-commerce platform with targeting customers aged 24-30 with logo Tenten.vn/ai. The photo should reflect a pink and black color palette for books products. No text in photo"
start_time = time.time()
response = client.images.generate(
    model="dall-e-3",
    prompt=PROMPT,
    n=1,
    size="1024x1024",
    quality="standard",
)

image_url = response.data[0].url
end_time = time.time()
response_time = end_time - start_time
print(f"Thời gian phản hồi: {response_time:.2f} giây")
print(image_url)
