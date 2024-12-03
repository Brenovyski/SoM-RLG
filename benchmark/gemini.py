import google.generativeai as genai
import os
import base64
import requests
from io import BytesIO
from dotenv import load_dotenv
import gc
import PIL.Image

load_dotenv()

# Get OpenAI API Key from environment variable
api_key = os.environ["GOOGLE_API_KEY"]
question = 'I want to drink water, what is the best way to do it?'
instructions = 'For any marks mentioned in your answer, please highlight them with [].'
message = question + instructions
somimage = PIL.Image.open("imagesom.jpg")
image = PIL.Image.open("image.jpg")

#Fast and versatile performance across a diverse variety of tasks
def request_genai15flash(message, image=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([message, image])
    print(response.text)

#High volume and lower intelligence tasks
def request_genai15flash8B(message, image=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content([message, image])
    print(response.text)

#Complex reasoning tasks requiring more intelligence
def request_genai15pro(message, image=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content([message, image])
    print(response.text)

#Improved coding, reasoning, and vision capabilities
def request_genexperimental1121(message, image=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-exp-1121")
    response = model.generate_content([message, image])
    print(response.text)

def benchmark ():

    print("Testing Gemini models")
    print('gemini 1.5 flash')
    request_genai15flash(message, image)
    print('gemini 1.5 flash 8b')
    request_genai15flash8B(message, image)
    print('gemini 1.5 pro')
    request_genai15pro(message, image)
    print('Gemini experimental 1121')
    request_genexperimental1121(message, image)
    
    print("Testing Gemini models + SoM")
    print('gemini 1.5 flash + SoM')
    request_genai15flash(message, somimage)
    print('gemini 1.5 flash 8b + SoM')
    request_genai15flash8B(message, somimage)
    print('gemini 1.5 pro + SoM')
    request_genai15pro(message, somimage)
    print('Gemini experimental 1121 + SoM')
    request_genexperimental1121(message, somimage)