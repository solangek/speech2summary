import argparse
import os
import tempfile
import threading
from datetime import datetime

from google import genai
from groq import Groq
import numpy as np
import sounddevice as sd
import soundfile as sf

def parse_args():
    parser = argparse.ArgumentParser(description="Record, transcribe, and summarize speech")
    parser.add_argument("--output", default="~/summaries/", help="Output directory (default: ~/summaries/)")
    parser.add_argument("--groq-model", default="whisper-large-v3-turbo", help="Groq transcription model (default: whisper-large-v3-turbo)")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini summarization model (default: gemini-2.5-flash)")
    return parser.parse_args()


def record_audio(samplerate=16000):
    audio_chunks = []
    recording_flag = threading.Event()

    def callback(indata, frames, time, status):
        if recording_flag.is_set():
            audio_chunks.append(indata.copy())

    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        input("Press Enter to start recording...")
        recording_flag.set()
        print("Recording... Press Enter to stop.")
        input()
        recording_flag.clear()

    if not audio_chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(audio_chunks, axis=0)


def transcribe(audio, model):
    print(f"Transcribing with Groq ({model})...")
    client = Groq()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, 16000)
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=("audio.wav", f),
                model=model,
            )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return result.text.strip()


def summarize(transcript, model):
    print(f"Summarizing with Gemini ({model})...")
    client = genai.Client()

    prompt = f"""You are a meeting/lecture assistant. Analyze the following transcript and provide a structured summary.
Respond in the same language as the transcript.

Format your response EXACTLY as:

## Key Points
- [point 1]
- [point 2]

## Action Items
- [item 1] (or "None identified" if there are none)

## Decisions Made
- [decision 1] (or "None identified" if there are none)

Transcript:
{transcript}"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text


def save_transcript(transcript, output_dir):
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"summary_{timestamp}.txt")

    with open(filename, "w") as f:
        f.write(f"Recording Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## TRANSCRIPT\n\n")
        f.write(transcript)
        f.write("\n")

    return filename


def append_summary(filename, transcript, summary):
    with open(filename, "r") as f:
        header = f.readline()  # "Recording Date: ..." line
    with open(filename, "w") as f:
        f.write(header)
        f.write("\n## SUMMARY\n\n")
        f.write(summary)
        f.write("\n\n---\n\n")
        f.write("## TRANSCRIPT\n\n")
        f.write(transcript)
        f.write("\n")


def main():
    args = parse_args()

    audio = record_audio()

    if audio.size == 0:
        print("No audio recorded.")
        return

    transcript = transcribe(audio, args.groq_model)

    if not transcript:
        print("No speech detected.")
        return

    print("\n--- TRANSCRIPT ---")
    print(transcript)

    filename = save_transcript(transcript, args.output)
    print(f"\nTranscript saved to: {filename}")

    summary = summarize(transcript, args.gemini_model)

    print("\n--- SUMMARY ---")
    print(summary)

    append_summary(filename, transcript, summary)
    print(f"Summary appended to: {filename}")


if __name__ == "__main__":
    main()
