# github/mirbyte
# v0.1

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_TRANSFER_CONCURRENCY"] = "4" # Parallel downloads, idk if it works or not
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import re
import sys
import json
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import yt_dlp
from colorama import Fore, Style, init
from tqdm import tqdm
import faster_whisper
import time
import psutil
import gc
import warnings

# ────────── console colours & config ──────────
init()
# no need for symlink ffs
warnings.filterwarnings("ignore", message=".*symlinks.*")

class Config:
    """Configuration class for transcription settings"""
    def __init__(self):
        self.max_memory_usage = 0.8 # 80% of available RAM
        self.temp_dir = tempfile.gettempdir()
        self.audio_formats = ["wav", "mp3", "m4a", "flac"]
        self.max_audio_length = 14400 # 4 hours in seconds
        self.chunk_size_mb = 100 # For large files

        # ============ Model cache and download settings ============
        self.model_cache_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(self.model_cache_dir, exist_ok=True)
        

    def get_optimal_settings(self) -> Dict[str, Any]:
        """Determine optimal settings based on system resources"""
        total_ram_gb = psutil.virtual_memory().total / (1024**3)

        if total_ram_gb >= 16:
            return {
                "beam_size": 5,
                "best_of": 5,
                "compute_type": "float16" if self._has_gpu() else "int8",
                "batch_size": 8
            }
        elif total_ram_gb >= 8:
            return {
                "beam_size": 3,
                "best_of": 3,
                "compute_type": "int8",
                "batch_size": 4
            }
        else:
            return {
                "beam_size": 1,
                "best_of": 1,
                "compute_type": "int8",
                "batch_size": 2
            }

    def _has_gpu(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

def c(msg, colour):  # short colour helper
    return f"{colour}{msg}{Style.RESET_ALL}"

def ts(sec: float) -> str:
    h, sec = divmod(int(sec), 3600)
    m, s = divmod(sec, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"\\/|?*]', '', name)
    return re.sub(r'\s+', ' ', name).strip()[:100] or "transcript"

def get_file_hash(filepath: str) -> str:
    """Generate hash for resume capability"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_audio_file(filepath: str) -> Tuple[bool, str, float]:
    """Validate audio file and return info"""
    try:
        import librosa
        y, sr = librosa.load(filepath, sr=None, duration=1) # Load just 1 second for validation
        # Get full duration without loading entire file
        duration = librosa.get_duration(path=filepath)

        if duration <= 0:
            return False, "Audio file appears to be empty", 0
        if duration > 14400:  # 4 hours
            return False, f"Audio too long ({duration/3600:.1f}h). Max 4 hours.", duration
        if sr < 8000:
            return False, f"Sample rate too low ({sr}Hz). Minimum 8kHz required.", duration

        return True, "Valid audio file", duration
    except Exception as e:
        return False, f"Audio validation failed: {str(e)}", 0

class YouTubeTranscriber:
    def __init__(self):
        self.config = Config()
        self.tmp_wav: Optional[str] = None
        self.model: Optional[faster_whisper.WhisperModel] = None
        self.audio_duration: float = 0
        self.session_dir: str = ""
        self.resume_data: Dict[str, Any] = {}
        self.transcription_settings: Dict[str, Any] = {}

    # ───────────────── UI ─────────────────
    @staticmethod
    def header():
        width = 65
        line = "=" * width
        ram_gb = psutil.virtual_memory().total / (1024**3)
    
        title = "YouTube Video Transcriber (Whisper v3)"
        ram_info = f"System RAM: {ram_gb:.1f}GB"
    
        print(c(f"\n{line}", Fore.CYAN))
        print(c(title.center(width), Fore.WHITE))
        print(c(ram_info.center(width), Fore.WHITE))
        print(c(f"{line}\n", Fore.CYAN))

        
        ###

    def ask_url(self) -> str:
        while True:
            url = input("Enter YouTube URL: ").strip()
            if not url:
                print(c("URL cannot be empty.", Fore.RED))
                continue

            # Basic URL validation
            if not (url.startswith(("http://", "https://")) and
                    any(domain in url for domain in ["youtu.be", "youtube.com", "m.youtube.com"])):
                print(c("Please enter a valid YouTube URL.", Fore.RED))
                continue

            # Test URL accessibility
            try:
                with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                    info = ydl.extract_info(url, download=False)
                    duration = info.get("duration", 0)

                    if duration > self.config.max_audio_length:
                        if input(c(f"Video is {duration/3600:.1f}h long. Continue? (y/n): ", Fore.YELLOW)).lower() != "y":
                            continue

                    print(c(f"✓ Valid video: {info.get('title', 'Unknown')} ({ts(duration)})", Fore.GREEN))
                    return url
            except Exception as e:
                print(c(f"Cannot access video: {str(e)}", Fore.RED))

    @staticmethod
    def ask_lang() -> str:
        print("\nChoose transcription language:")
        print(" 1. Finnish")
        print(" 2. English")
        print(" 3. Auto-detect")
        print(" 4. Other language (specify)")

        # All languages now use the same model
        mapping = {"1": "fi", "2": "en", "3": None}

        while True:
            ch = input("\nChoice (1-4): ").strip()

            if ch in mapping:
                lang = mapping[ch]
                return lang if lang else "auto"
            elif ch == "4":
                print("\nSupported languages include:")
                print("de (German), fr (French), es (Spanish), it (Italian),")
                print("pt (Portuguese), ru (Russian), ja (Japanese), ko (Korean),")
                print("zh (Chinese), ar (Arabic), hi (Hindi), and many more...")

                lang = input("\nEnter language code (e.g., 'de', 'fr', 'es'): ").strip().lower()

                if len(lang) == 2 and lang.isalpha():
                    return lang

                print(c("Please enter a valid 2-letter language code.", Fore.RED))
            else:
                print(c("Enter 1, 2, 3, or 4.", Fore.RED))

    def ask_output_format(self) -> str:
        print("\nChoose output format:")
        print(" 1. Text with timestamps")
        print(" 2. Plain text only")
        print(" 3. JSON format")
        print(" 4. SRT subtitles")

        mapping = {"1": "timestamps", "2": "plain", "3": "json", "4": "srt"}

        while True:
            ch = input("\nChoice (1-4): ").strip()

            if ch in mapping:
                return mapping[ch]

            print(c("Enter 1, 2, 3, or 4.", Fore.RED))

    # ───────────── session management ─────────────
    def create_session(self, url: str) -> str:
        """Create session directory for resume capability"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        session_name = f"session_{url_hash}_{timestamp}"
        self.session_dir = os.path.join(self.config.temp_dir, session_name)
        os.makedirs(self.session_dir, exist_ok=True)
        return self.session_dir

    def save_resume_data(self, data: Dict[str, Any]):
        """Save progress for resume capability"""
        resume_file = os.path.join(self.session_dir, "resume.json")
        try:
            with open(resume_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Non-critical if resume save fails

    # ───────────── download audio ─────────────
    def download_audio(self, url: str) -> str:
        print("\nDownloading audio...")

        # ============ SponsorBlock configuration ============
        opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(self.session_dir, "audio.%(ext)s"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            # SponsorBlock settings
            "extractor_args": {
                "youtube": {
                    "skip": ["sponsor"]
                }
            },
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }
        # ========================================================

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Show download progress
                def progress_hook(d):
                    if d['status'] == 'downloading':
                        if 'total_bytes' in d:
                            pct = d['downloaded_bytes'] / d['total_bytes'] * 100
                            print(f"\rDownloading: {pct:.1f}%", end='', flush=True)

                opts['progress_hooks'] = [progress_hook]
                info = ydl.extract_info(url, download=True)
                print()  # New line after progress

                self.tmp_wav = os.path.join(self.session_dir, "audio.wav")

                # Validate downloaded audio
                is_valid, msg, duration = validate_audio_file(self.tmp_wav)

                if not is_valid:
                    raise Exception(f"Audio validation failed: {msg}")

                self.audio_duration = duration

                # Save resume data
                self.save_resume_data({
                    "url": url,
                    "title": info.get("title", "YouTube Audio"),
                    "duration": duration,
                    "audio_file": self.tmp_wav,
                    "stage": "downloaded"
                })

                print(c(f"✓ Audio downloaded ({ts(duration)})", Fore.GREEN))
                return info.get("title", "YouTube Audio")

        except Exception as e:
            print(c(f"Error downloading audio: {e}", Fore.RED))
            raise

    # ───────────── model loader (simplified - single model) ─────────────
    def load_model(self):
        """Load Whisper large-v3 model for all languages"""
        model_name = "large-v3"

        # Get optimal settings based on system
        settings = self.config.get_optimal_settings()
        device = "cuda" if self.config._has_gpu() else "cpu"

        print(f"\nLoading Whisper large-v3 model (universal)...")
        print(f"Device: {device} | Compute: {settings['compute_type']} | Beam size: {settings['beam_size']}")
        print(f"Model cache: {self.config.model_cache_dir}")

        try:
            # Free up memory before loading model
            if self.model:
                del self.model
            gc.collect()

            # ============ Pass custom cache_dir to WhisperModel ============
            self.model = faster_whisper.WhisperModel(
                model_name,
                device=device,
                compute_type=settings['compute_type'],
                cpu_threads=min(4, psutil.cpu_count()),
                download_root=self.config.model_cache_dir
            )
            # ===================================================================

            # Store settings for transcription
            self.transcription_settings = settings

            print(c("✓ Model loaded successfully", Fore.GREEN))

        except Exception as e:
            print(c(f"Error loading model: {e}", Fore.RED))

            # Fallback to base model
            print(c("Trying base model...", Fore.YELLOW))

            try:
                self.model = faster_whisper.WhisperModel(
                    "base",
                    device="cpu",
                    compute_type="int8",
                    download_root=self.config.model_cache_dir # Custom model cache
                )

                print(c("✓ Fallback model loaded", Fore.GREEN))
            except Exception as e2:
                raise Exception(f"Both models failed: {e}, {e2}")

    # ───────────── transcription ─────────────
    def transcribe(self, wav: str, choice: str):
        language_param = None if choice == "auto" else choice

        if choice == "auto":
            print(c("\nLanguage will be auto-detected.", Fore.CYAN))
        else:
            lang_names = {
                "fi": "Finnish", "en": "English", "de": "German",
                "fr": "French", "es": "Spanish", "it": "Italian",
                "pt": "Portuguese", "ru": "Russian", "ja": "Japanese"
            }

            lang_display = lang_names.get(choice, choice.upper())
            print(c(f"\nTranscribing in {lang_display}...", Fore.CYAN))

        # Monitor memory usage
        initial_memory = psutil.virtual_memory().percent

        try:
            seg_iter, info = self.model.transcribe(
                wav,
                language=language_param,
                task="transcribe",
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=300,
                    max_speech_duration_s=30,
                ),
                condition_on_previous_text=False,
                beam_size=self.transcription_settings.get("beam_size", 1),
                best_of=self.transcription_settings.get("best_of", 1),
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )

            print("Processing audio segments...")

            segments = []
            start_time = time.time()
            last_end_time = 0

            # Enhanced progress tracking
            with tqdm(desc="Processing",
                      bar_format="{desc}: {elapsed} | {postfix}",
                      unit="seg") as pbar:

                for segment in seg_iter:
                    segments.append(segment)
                    last_end_time = segment.end

                    # Memory monitoring
                    current_memory = psutil.virtual_memory().percent

                    if current_memory > 90:
                        print(c(f"\nWarning: High memory usage ({current_memory:.1f}%)", Fore.YELLOW))

                    # Update progress
                    if self.audio_duration > 0:
                        progress_pct = min((segment.end / self.audio_duration) * 100, 100)
                        elapsed = time.time() - start_time
                        eta = (elapsed / progress_pct * 100 - elapsed) if progress_pct > 0 else 0

                        pbar.set_postfix_str(
                            f"Progress: {progress_pct:.1f}% | "
                            f"Segments: {len(segments)} | "
                            f"ETA: {ts(eta) if eta > 0 else 'calculating...'}"
                        )
                    else:
                        pbar.set_postfix_str(f"Segments: {len(segments)}")

                    pbar.update(1)

                    # Periodic progress save
                    if len(segments) % 50 == 0:
                        self.save_resume_data({
                            "stage": "transcribing",
                            "segments_processed": len(segments),
                            "last_timestamp": segment.end
                        })

            if not segments:
                print(c("No speech detected in audio.", Fore.RED))
                return segments, info

            # Final validation
            total_transcribed_time = max(s.end for s in segments) if segments else 0
            coverage = (total_transcribed_time / self.audio_duration * 100) if self.audio_duration > 0 else 0

            print(c(f"✓ Transcription complete!", Fore.GREEN))
            print(f" Segments: {len(segments)}")
            print(f" Coverage: {coverage:.1f}% of audio")
            print(f" Language: {info.language if info else 'Unknown'}")

            return segments, info

        except Exception as e:
            print(c(f"Error during transcription: {e}", Fore.RED))
            raise

    # ───────────── save transcript ─────────────
    def save_transcript(self, segs: List, info: Any, title: str, format_type: str):
        """Save transcript in requested format"""
        base_name = safe_name(title)

        if format_type == "json":
            self._save_json(segs, info, title, base_name)
        elif format_type == "srt":
            self._save_srt(segs, base_name)
        elif format_type == "plain":
            self._save_plain(segs, base_name)
        else:  # timestamps
            self._save_timestamps(segs, info, title, base_name)

    def _save_timestamps(self, segs, info, title, base_name):
        fname = f"{base_name}.txt"
        self._check_overwrite(fname)

        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write("YouTube Transcription\n" + "=" * 60 + "\n")
                f.write(f"Title : {title}\n")
                f.write(f"Language : {info.language if info else 'Unknown'}\n")
                f.write(f"Duration : {ts(self.audio_duration)}\n")
                f.write(f"Segments : {len(segs)}\n")
                f.write(f"Model : Whisper large-v3\n")
                f.write("=" * 60 + "\n\nTIMESTAMPS\n" + "-" * 40 + "\n")

                for s in segs:
                    f.write(f"[{ts(s.start)} → {ts(s.end)}] {s.text.strip()}\n")

                f.write("\n" + "=" * 60 + "\nFULL TEXT\n" + "-" * 40 + "\n")
                f.write(" ".join(s.text.strip() for s in segs))

            print(c(f"✓ Transcript saved → {fname}", Fore.GREEN))

        except Exception as e:
            print(c(f"Error saving transcript: {e}", Fore.RED))
            raise

    def _save_json(self, segs, info, title, base_name):
        fname = f"{base_name}.json"
        self._check_overwrite(fname)

        data = {
            "title": title,
            "language": info.language if info else "unknown",
            "duration": self.audio_duration,
            "model": "whisper-large-v3",
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text.strip()
                } for s in segs
            ],
            "full_text": " ".join(s.text.strip() for s in segs)
        }

        try:
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(c(f"✓ JSON transcript saved → {fname}", Fore.GREEN))

        except Exception as e:
            print(c(f"Error saving JSON: {e}", Fore.RED))
            raise

    def _save_srt(self, segs, base_name):
        fname = f"{base_name}.srt"
        self._check_overwrite(fname)

        try:
            with open(fname, "w", encoding="utf-8") as f:
                for i, s in enumerate(segs, 1):
                    start_time = self._seconds_to_srt_time(s.start)
                    end_time = self._seconds_to_srt_time(s.end)
                    f.write(f"{i}\n{start_time} --> {end_time}\n{s.text.strip()}\n\n")

            print(c(f"✓ SRT subtitles saved → {fname}", Fore.GREEN))

        except Exception as e:
            print(c(f"Error saving SRT: {e}", Fore.RED))
            raise

    def _save_plain(self, segs, base_name):
        fname = f"{base_name}_plain.txt"
        self._check_overwrite(fname)

        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(" ".join(s.text.strip() for s in segs))

            print(c(f"✓ Plain text saved → {fname}", Fore.GREEN))

        except Exception as e:
            print(c(f"Error saving plain text: {e}", Fore.RED))
            raise

    @staticmethod
    def _seconds_to_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def _check_overwrite(self, fname):
        if os.path.exists(fname):
            if input(c(f"File '{fname}' exists. Overwrite? (y/n): ", Fore.YELLOW)).lower() != "y":
                print("Save cancelled.")
                sys.exit(0)

    # ───────────── cleanup ─────────────
    def cleanup(self, keep_session=False):
        """Enhanced cleanup with option to keep session data"""
        try:
            if self.model:
                del self.model
            self.model = None
            gc.collect()

            if not keep_session and self.session_dir and os.path.exists(self.session_dir):
                shutil.rmtree(self.session_dir, ignore_errors=True)

            print(c("✓ Temporary files cleaned up", Fore.GREEN))

        except Exception as e:
            print(c(f"Warning: Cleanup issues: {e}", Fore.YELLOW))

    # ───────────── main loop ─────────────
    def run(self):
        self.header()

        try:
            url = self.ask_url()
            lang = self.ask_lang()
            output_format = self.ask_output_format()

            # Create session for this run
            self.create_session(url)

            title = self.download_audio(url)

            self.load_model()  # Single model for all languages

            segs, info = self.transcribe(self.tmp_wav, lang)

            if segs:
                self.save_transcript(segs, info, title, output_format)
                print(c(f"\n✓ All done! Processed {len(segs)} segments.", Fore.GREEN))
            else:
                print(c("Nothing to save – no speech detected.", Fore.RED))

        except KeyboardInterrupt:
            print(c("\n\nInterrupted by user.", Fore.YELLOW))
            if input("Keep session data for resume? (y/n): ").lower() == "y":
                print(c(f"Session saved in: {self.session_dir}", Fore.CYAN))
                return

        except Exception as e:
            print(c(f"\nUnexpected error: {e}", Fore.RED))

        finally:
            self.cleanup()

if __name__ == "__main__":
    YouTubeTranscriber().run()
