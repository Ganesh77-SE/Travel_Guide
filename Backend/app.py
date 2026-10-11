import base64
import os
import tempfile
import traceback

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai

load_dotenv()

app = Flask(__name__)
CORS(app)

MURF_API_KEY = os.getenv("MURF_API_KEY")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Change this if you want to try another Gemini model
GEMINI_MODEL = "gemini-3.8-flash"

PROMPTS = {
    "Summary": """
You are a professional tourist guide.
Provide a high-level overview of "{place}" in {language}.

Focus on:
- The historical significance
- Why the place is famous
- Key architectural or cultural highlights

Keep the explanation concise, engaging, and easy to follow.
Avoid excessive details and dates.
Limit the response to around 200 words.

Respond ONLY in {language}.
""",
    "Detailed": """
You are a professional tourist guide.
Provide a detailed and immersive explanation of "{place}" in {language}.

Cover:
- Historical background and timeline
- Architectural design and unique features
- Cultural importance and notable events
- Interesting facts and visitor insights

Explain concepts clearly and in a storytelling manner.
Include relevant details and examples to create a rich experience.
Limit the response to around 400 words.

Respond ONLY in {language}.
""",
}


@app.route("/ping")
def ping():
    return jsonify({"status": "backend connected"})


def generate_description(place, answer_type, language):
    if answer_type not in PROMPTS:
        raise ValueError(
            f"answerType must be 'Summary' or 'Detailed', but got '{answer_type}'"
        )

    prompt = PROMPTS[answer_type].format(place=place, language=language)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


def generate_speech(text, voice_id, locale):
    url = "https://global.api.murf.ai/v1/speech/stream"
    headers = {
        "api-key": MURF_API_KEY,
        "Content-Type": "application/json",
    }
    data = {
        "voice_id": voice_id,
        "text": text,
        "locale": locale,
        "model": "FALCON",
        "format": "MP3",
        "sampleRate": 24000,
        "channelType": "MONO",
    }

    response = requests.post(url, headers=headers, json=data, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Murf error {response.status_code}: {response.text}")

    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    with temp_audio as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    print("Audio streaming completed")
    return temp_audio.name


@app.route("/generate-audio-guide", methods=["POST"])
def generate_audio_guide():
    audio_path = None
    try:
        data = request.json
        print("Received:", data)

        place = data["place"]
        answer_type = data["answerType"]
        language = data["language"]
        voice_id = data["voiceId"]
        locale = data["locale"]

        text_description = generate_description(place, answer_type, language)
        audio_path = generate_speech(text_description, voice_id, locale)

        with open(audio_path, "rb") as f:
            encoded_audio = base64.b64encode(f.read()).decode("utf-8")

        return jsonify(
            {
                "description": text_description,
                "audioBase64": encoded_audio,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    app.run(debug=True)