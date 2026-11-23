from google import genai

client = genai.Client()
candidates = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

for name in candidates:
    try:
        reply = client.models.generate_content(
            model=name, contents="Reply with the word OK"
        ).text
        print(f"WORKS   {name}: {reply.strip()}")
    except Exception as err:
        print(f"FAILS   {name}: {str(err)[:90]}")