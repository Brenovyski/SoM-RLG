import anthropic
import os
import base64
import requests
from io import BytesIO
from dotenv import load_dotenv
import gc
import PIL.Image

load_dotenv()
api_key = os.environ["ANTHROPIC_API_KEY"]

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=api_key,
)
message = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude, explain the meaning of life."},
    ]
)
message_content = message.content[0].text
print(message_content)