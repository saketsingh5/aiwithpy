import asyncio
from dotenv import load_dotenv
load_dotenv()

import speech_recognition as sr
from openai import OpenAI
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer


client = OpenAI()
async_client = AsyncOpenAI()

async def tts(speech: str):
   async with async_client.audio.speech.with_streaming_response.create(
         model="gpt-4o-mini-tts",
         voice="coral",
         input=speech,
         instructions="Speak in a cheerful manner with delight positive and happy tone.",
         response_format="pcm"
   ) as response:
       await LocalAudioPlayer().play(response)

def main():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 1


        SYSTEM_PROMPT = f"""  
            You are an expert voice agent. You are given the transcript of what user has said using voice.
            You need to output as if you are an voice agent and whatever you speak will be converted back to audio using AI and played back to user.
         """

        messages= [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        while True:
         print("Speak something...")
         audio = r.listen(source)

         print("Processing audio... (STT)")

         
         stt = r.recognize_google(audio)
         print("You said:", stt)
         

         messages.append({"role": "user", "content": stt})
         response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages
         )

         print("AI response:", response.choices[0].message.content);
         asyncio.run(tts(speech=response.choices[0].message.content))

main()

