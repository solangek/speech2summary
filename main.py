import argparse
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone

from google import genai
from groq import Groq, RateLimitError
import numpy as np
import sounddevice as sd
import soundfile as sf

DEFAULT_PROMPT = """You are a meeting/lecture assistant. Analyze the following transcript and provide a structured summary.
Respond in the same language as the transcript.

Format your response EXACTLY as:

## Key Points
- [point 1]
- [point 2]

## Action Items
- [item 1] (or "None identified" if there are none)

## Decisions Made
- [decision 1] (or "None identified" if there are none)"""

COMPRESS_THRESHOLD_BYTES = 5_000_000
SESSION_ID = uuid.uuid4().hex[:12]

_usage_logger = logging.getLogger("speech2summary.usage")
if not _usage_logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _usage_logger.addHandler(_handler)
    _usage_logger.setLevel(logging.INFO)
    _usage_logger.propagate = False


def log_event(event: str, **fields) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "session_id": SESSION_ID,
        **fields,
    }
    _usage_logger.info(json.dumps(payload, ensure_ascii=False, default=str))


def parse_args():
    parser = argparse.ArgumentParser(description="Record, transcribe, and summarize speech")
    parser.add_argument("--output", default="~/summaries/", help="Output directory (default: ~/summaries/)")
    parser.add_argument("--groq-model", default="whisper-large-v3-turbo", help="Groq transcription model")
    parser.add_argument("--gemini-model", default="gemini-flash-latest", help="Gemini summarization model")
    parser.add_argument("--no-compress", action="store_true", help="Skip ffmpeg Opus compression before transcription")
    parser.add_argument("--log-file", help="Append JSON usage events to this file in addition to stderr")
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


def audio_to_wav_bytes(audio, samplerate=16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format="WAV")
    return buf.getvalue()


def compress_audio(wav_bytes: bytes) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-i", "pipe:0",
            "-ac", "1", "-ar", "16000",
            "-c:a", "libopus", "-b:a", "16k",
            "-f", "ogg", "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=True,
    )
    return result.stdout


def transcribe_with_retry(audio_bytes: bytes, filename: str, model: str, max_attempts: int = 4) -> str:
    client = Groq()
    for attempt in range(max_attempts):
        try:
            result = client.audio.transcriptions.create(
                file=(filename, audio_bytes),
                model=model,
            )
            return result.text.strip()
        except RateLimitError as e:
            if attempt == max_attempts - 1:
                raise
            retry_after = 2 ** attempt
            try:
                retry_after = int(e.response.headers.get("retry-after", retry_after))
            except (AttributeError, ValueError, TypeError):
                pass
            print(f"Rate limited, retrying in {retry_after}s…", file=sys.stderr)
            time.sleep(retry_after)
    return ""


def transcribe(audio, model, compress: bool = True) -> str:
    print(f"Transcribing with Groq ({model})...")
    audio_bytes = audio_to_wav_bytes(audio)
    filename = "audio.wav"
    original_size = len(audio_bytes)

    if compress and original_size >= COMPRESS_THRESHOLD_BYTES:
        t0 = time.monotonic()
        try:
            audio_bytes = compress_audio(audio_bytes)
            filename = "audio.ogg"
            log_event(
                "compress_done",
                before_bytes=original_size,
                after_bytes=len(audio_bytes),
                duration_s=round(time.monotonic() - t0, 2),
            )
            print(f"Compressed {original_size / 1_000_000:.1f} Mo → {len(audio_bytes) / 1_000_000:.2f} Mo")
        except FileNotFoundError:
            log_event("compress_skipped", reason="ffmpeg_not_found")
            print("ffmpeg not found — sending uncompressed audio.", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            log_event("compress_failed", error=e.stderr.decode(errors="ignore")[:200])
            print("Compression failed — sending uncompressed audio.", file=sys.stderr)

    t0 = time.monotonic()
    try:
        text = transcribe_with_retry(audio_bytes, filename, model)
    except Exception as e:
        log_event("transcribe_failed", model=model, bytes=len(audio_bytes), error=str(e)[:300])
        raise
    log_event(
        "transcribe_done",
        model=model,
        bytes=len(audio_bytes),
        duration_s=round(time.monotonic() - t0, 2),
        transcript_chars=len(text),
    )
    return text


def summarize(transcript, model):
    print(f"Summarizing with Gemini ({model})...")
    client = genai.Client()
    prompt = f"{os.environ.get('SUMMARIZATION_PROMPT', DEFAULT_PROMPT)}\n\nTranscript:\n{transcript}"

    t0 = time.monotonic()
    try:
        response = client.models.generate_content(model=model, contents=prompt)
    except Exception as e:
        log_event("summarize_failed", model=model, error=str(e)[:300])
        raise
    log_event(
        "summarize_done",
        model=model,
        transcript_chars=len(transcript),
        summary_chars=len(response.text or ""),
        duration_s=round(time.monotonic() - t0, 2),
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

    if args.log_file:
        file_handler = logging.FileHandler(os.path.expanduser(args.log_file))
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        _usage_logger.addHandler(file_handler)

    log_event("run_started", groq_model=args.groq_model, gemini_model=args.gemini_model, compress=not args.no_compress)

    audio = record_audio()

    if audio.size == 0:
        print("No audio recorded.")
        log_event("no_audio")
        return

    transcript = transcribe(audio, args.groq_model, compress=not args.no_compress)

    if not transcript:
        print("No speech detected.")
        log_event("transcribe_empty")
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
    log_event("run_done", filename=filename)


if __name__ == "__main__":
    main()
