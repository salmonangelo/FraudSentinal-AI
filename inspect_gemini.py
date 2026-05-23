import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=api_key)
    # Using a model that is confirmed to be available and supported
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("Sending test request to Gemini API (gemini-2.5-flash)...")
    response = model.generate_content("Hello! This is a test. Please reply in one sentence.")
    print("Response text:", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
