# Navis - Voice Assistant with Memory + Object Detection

import os
import sys
import time
import json
import subprocess
import requests
import speech_recognition as sr
from dotenv import load_dotenv

# Load OpenRouter API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# User data file path
USER_DATA_FILE = "/home/robo/NAVIS/Assistant/user_data.json"

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Corrupted user_data.json. Resetting.")
            os.remove(USER_DATA_FILE)
    return {}

user_data = load_user_data()

# Speak using espeak
def speak(text):
    print(f"🗣️ {text}")
    os.system(f'espeak -s140 -p70 "{text}"')

# Listen from microphone
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            speak("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("❌ Could not request results from Google Speech Recognition.")
            speak("Speech recognition service is unavailable.")
            return None

# Run object detection for 25 seconds
def run_object_detection():
    speak("Starting object detection for 25 seconds.")
    process = subprocess.Popen([sys.executable, "/home/robo/raspi_arduino_testing/Voice_Assistant/Test/Object_Detection/detection.py"])
    time.sleep(25)
    process.terminate()
    try:
        process.wait(timeout=2)
        speak("Object detection completed.")
    except subprocess.TimeoutExpired:
        process.kill()
        speak("Object detection was forcibly stopped.")

# Ask AI assistant
def ask_jarvis(prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://yourapp.com",
        "X-Title": "Navis-assistant"
    }

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {
                "role": "system",
                "content": f"You are Navis, a helpful and intelligent assistant. The user's name is {user_data.get('name', 'unknown')}. Reply in less than 30 words."
            },
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        reply = response.json()['choices'][0]['message']['content']
        return reply.strip()
    else:
        return f"Error: {response.status_code} - {response.text}"

# === Main Loop ===
if __name__ == "__main__":
    print("🔊 Navis with Voice is online. Say 'exit' to quit.\n")
    speak("Navis is online. Say something.")

    while True:
        user_input = listen()
        if user_input:
            command = user_input.lower()

            if command in ["exit", "quit", "stop"]:
                speak("Goodbye!")
                break

            elif "object detection" in command or "detect objects" in command:
                run_object_detection()
                continue

            elif any(x in command for x in ["what is my", "when is my", "tell me my"]):
                matched = False
                for key in user_data:
                    if key.lower() in command:
                        speak(f"Your {key} is {user_data[key]}.")
                        matched = True
                        break
                if not matched:
                    speak("I couldn't find that information in my memory.")
                continue

            # Default: ask AI
            reply = ask_jarvis(user_input)
            print(f"Navis: {reply}\n")
            speak(reply)
