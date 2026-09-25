from tenacity import wait_chain
import yt_dlp
from pydub import AudioSegment
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, 'downloads')

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "restrictfilenames": True,
        "js_runtimes": {"node": {}},
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        raw_filename = ydl.prepare_filename(info)
        filename = os.path.splitext(raw_filename)[0] + ".wav"
        if os.path.exists(filename):
            print(f"Found existing downloaded file: {filename}")
            return filename
        info = ydl.extract_info(url, download=True)

    return filename

def convert_to_wav(input_path: str)-> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    if os.path.exists(output_path):
        print(f"Found existing converted file: {output_path}")
        return output_path
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path

def chunk_audio(wav_path: str, chunk_mins : int = 10) -> list:
    # Check if chunks already exist on disk
    existing = []
    i = 0
    while True:
        c_path = f"{wav_path}_chunk_{i}.wav"
        if os.path.exists(c_path):
            existing.append(c_path)
            i += 1
        else:
            break
    if existing:
        print(f"Reusing {len(existing)} existing audio chunk(s) from disk.")
        return existing

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_mins * 60 * 1000

    chunks = []

    for i,start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start: start+chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks

def process_input(source:str) -> list:
    if source.startswith("https://") or source.startswith("http://"):
        print("Detected Youtube URL. Downloading Audio...")
        downloaded_audio = download_youtube_audio(source)
        print("Converting to 16kHz mono wav...")
        wav_path = convert_to_wav(downloaded_audio)
    else:
        print("Detected local file. Converting to wav...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks