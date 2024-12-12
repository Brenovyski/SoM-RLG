import os
import json
from io import BytesIO
import PIL.Image
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Get Google API Key from environment variable
api_key = os.environ["GOOGLE_API_KEY"]

def load_image_from_file(image_path):
    return PIL.Image.open(image_path)

def prepare_gemini_inputs(message, pil_image=None, image_path=None, files=None):
    final_message = message
    main_image = pil_image

    if main_image is None and image_path is not None:
        main_image = load_image_from_file(image_path)

    attachments = []

    if files is not None:
        for fpath in files:
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fpath)[1].lower()
            if ext in [".jpg", ".jpeg", ".png"]:
                img = load_image_from_file(fpath)
                attachments.append(img)
                final_message += f"\n[Attached image file: {os.path.basename(fpath)}]"
            elif ext == ".json":
                with open(fpath, "r", encoding="utf-8") as json_file:
                    data = json.load(json_file)
                    json_str = json.dumps(data, ensure_ascii=False, indent=2)
                    final_message += f"\n[Attached JSON file content:]\n{json_str}"
            else:
                with open(fpath, "r", encoding="utf-8", errors="replace") as txt_file:
                    text_data = txt_file.read()
                    final_message += f"\n[Attached file content ({os.path.basename(fpath)}):]\n{text_data}"

    return final_message, main_image, attachments

def request_genai_model(model_name, message, pil_image=None, image_path=None, files=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    final_message, main_image, attachments = prepare_gemini_inputs(
        message, pil_image=pil_image, image_path=image_path, files=files
    )

    inputs = [final_message]
    if main_image is not None:
        inputs.append(main_image)
    if attachments:
        inputs.extend(attachments)

    response = model.generate_content(inputs)

    return response.text