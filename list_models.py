import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=api_key)
    print("Listing models:")
    for model in genai.list_models():
        print(f"- {model.name} (supported: {model.supported_generation_methods})")
except Exception as e:
    import traceback
    traceback.print_exc()
