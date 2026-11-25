import os
from google import genai

client = genai.Client()
response = client.models.generate_content(
    model=os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite"),
    contents="Reply with the word OK"
)
print(response.text)