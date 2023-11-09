#openai ver: 1.1
import io
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="###",
)

speech_file_path = Path("D:\Lam_Viec\Python\Reports").parent / "speech_3.mp3"
response = client.audio.speech.create(
  model="tts-1-hd",
  voice="nova", 
  input="""###"""
)

response.stream_to_file(speech_file_path)
