# YouTube Whisper Transcriber

A powerful Python tool for transcribing YouTube videos to text using OpenAI's Whisper large-v3 model. Features universal language support, multiple output formats and SponsorBlock integration. *Create YouTube video summaries completely for free!*

[Video Demo](https://www.youtube.com/watch?v=VQk_CA9RRCo)


this script still lacks proper testing but i thought ill just put it out there for now


## Requirements

### System Requirements

- **Python 3.8+**
- **Minimum 4GB of disk space**
- **Minimum 4GB RAM** (8GB+ recommended)
- **GPU support** optional but recommended (CUDA for NVIDIA GPUs)


### Dependencies

```bash
pip install yt-dlp faster-whisper colorama tqdm psutil librosa torch
```

*Update yt-dlp regularly!*

```bash
pip install --upgrade yt-dlp
```

## Usage

### Interactive Mode

Simply run the script and follow the prompts:

```bash
python yt_whisper_transcriber.py
```

### Supported Languages

- **Auto-detect** - Automatically identifies the spoken language (not recommended)
- **Predefined options** - Finnish (fi), English (en)
- **Custom languages** - German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt), Russian (ru), Japanese (ja), Korean (ko), Chinese (zh), and many more


## Output Formats

### 1. Text with Timestamps (Default)

Includes metadata header and full text:

```
YouTube Transcription
============================================================
Title : Video Title Here
Language : en
Duration : 00:12:34
Segments : 145
Model : Whisper large-v3
============================================================

TIMESTAMPS
----------------------------------------
[00:00:15 → 00:00:18] Welcome to this tutorial video
[00:00:18 → 00:00:22] Today we'll be learning about Python

============================================================
FULL TEXT
----------------------------------------
Welcome to this tutorial video. Today we'll be learning about Python...
```


### 2. Plain Text

```
Welcome to this tutorial video. Today we'll be learning about Python...
```


### 3. JSON Format

```json
{
  "title": "Video Title",
  "language": "en",
  "duration": 754.5,
  "model": "whisper-large-v3",
  "segments": [
    {
      "start": 15.0,
      "end": 18.0,
      "text": "Welcome to this tutorial video"
    },
    {
      "start": 18.0,
      "end": 22.0,
      "text": "Today we'll be learning about Python"
    }
  ],
  "full_text": "Welcome to this tutorial video. Today we'll be learning about Python..."
}
```


### 4. SRT Subtitles

```
1
00:00:15,000 --> 00:00:18,000
Welcome to this tutorial video

2
00:00:18,000 --> 00:00:22,000
Today we'll be learning about Python
```


## System Optimization

The tool automatically optimizes settings based on your system resources:


| RAM Available | Beam Size | Compute Type | Batch Size | GPU Support |
| :-- | :-- | :-- | :-- | :-- |
| 16GB+ | 5 | float16/int8* | 8 | Yes (CUDA) |
| 8GB-16GB | 3 | int8 | 4 | CPU only |
| <8GB | 1 | int8 | 2 | CPU only |

*Uses float16 if GPU available, int8 otherwise

## Advanced Configuration

### Model Cache

Models are automatically downloaded to a `models/` directory in the script location. This allows offline use after initial download.

### VAD (Voice Activity Detection)

The transcriber uses enhanced VAD to skip silence:

- **Min silence duration**: 500ms (skips short pauses)
- **Speech padding**: 300ms (includes natural speech boundaries)
- **Max speech duration**: 30 seconds per segment


### SponsorBlock Integration

Automatically skips known sponsor segments during YouTube download. https://github.com/ajayyy/SponsorBlock


## Troubleshooting

### Common Issues

**"Cannot access video"**

- Verify the YouTube URL is correct and accessible
- Check your internet connection
- Some videos may be region-restricted or age-gated

**"Audio validation failed"**

- The video may not contain audio or has insufficient audio duration
- Audio quality might be too low (<8kHz sample rate)
- Maximum audio length is 4 hours
- Try a different video

**"High memory usage warning"**

- Close other applications to free up RAM
- The tool automatically adapts beam size and batch size for your available memory
- Consider using a system with more RAM for faster processing

**"Model loading failed"**

- Ensure you have at least 2GB free disk space for the Whisper large-v3 model (~3GB)
- Check internet connection for initial model download
- The tool automatically falls back to the base model if large-v3 fails
- Clear the `models/` directory if corrupt and restart

**"No speech detected"**

- The video may contain only music or background noise
- Try adjusting the VAD sensitivity (currently optimized for standard speech)


### Performance Tips

- **Use GPU** - Install CUDA-compatible PyTorch for 5-10x faster processing
- **Specify language** - Explicitly choosing language is ~20% faster than auto-detection
- **Close background apps** - Frees RAM for larger batch sizes and beam search
- **Use timestamps format** - Fastest format to save (less processing than JSON)


## Technical Details

- **Model**: OpenAI Whisper large-v3 (universal multilingual model)
- **Backend**: faster-whisper (optimized C++ implementation)
- **Audio Extraction**: yt-dlp with SponsorBlock support
- **Audio Processing**: Librosa for validation and sample rate detection
- **Voice Activity Detection**: faster-whisper built-in VAD with custom parameters
- **Session Persistence**: JSON-based resume files in system temp directory
- **Memory Management**: Real-time psutil monitoring with dynamic resource allocation



***

**Note**: This tool is for educational and personal use. Please respect YouTube's terms of service and copyright laws when using this tool.



