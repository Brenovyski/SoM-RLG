import os
import base64
import requests
import json
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

def encode_image_from_file(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_pil(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def request_gpt4v(prompt, model_name, pil_image=None, files=None):
    """
    Send a request to the GPT-4 vision model.
    Parameters:
      - prompt: The user prompt as a string.
      - model_name: The model name string (e.g., "chatgpt-4o-latest").
      - pil_image: Optional PIL image object to be included as the main image.
      - files: Optional list of file paths (images, JSON, text) to be appended.
    """

    # Start building the user content array from the prompt
    user_content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    # If we have a PIL image, encode it and add as image_url
    if pil_image is not None:
        encoded_image = encode_image_from_pil(pil_image)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })

    # Process additional files if any
    if files is not None:
        for fpath in files:
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                # Additional image
                img_b64 = encode_image_from_file(fpath)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
            elif ext == ".json":
                # JSON file content as text
                with open(fpath, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    json_str = json.dumps(data, ensure_ascii=False, indent=2)
                    user_content.append({
                        "type": "text",
                        "text": f"Attached JSON file content:\n{json_str}"
                    })
            else:
                # Treat as text file
                with open(fpath, "r", encoding="utf-8", errors="replace") as txt_file:
                    text_data = txt_file.read()
                    user_content.append({
                        "type": "text",
                        "text": f"Attached file content ({os.path.basename(fpath)}):\n{text_data}"
                    })

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": [
                ]
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_tokens": 800
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    res = response.json()['choices'][0]['message']['content']
    return res
