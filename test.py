import time
from google import genai
from google.genai import errors

client = genai.Client()

for attempt in range(5):
    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="Say hello in one sentence."
        )
        print(response.text)
        break
    except errors.ServerError:
        print(f"Busy, retrying ({attempt + 1}/5)...")
        time.sleep(10)
else:
    print("Still busy after 5 tries. Try a different model.")