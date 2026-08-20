"""Connectivity smoke test: confirm the OpenAI-compatible endpoint in .env responds.

    python test.py
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

response = client.chat.completions.create(
    model=os.getenv("OPENAI_MODEL", "gemma-4-E4B-it"),
    messages=[{"role": "user", "content": "Briefly introduce the semiconductor industry in Taiwan."}],
    max_tokens=512,
)

print(response.choices[0].message.content)
