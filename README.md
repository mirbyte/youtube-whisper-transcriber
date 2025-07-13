
# YouTube Whisper Transcriber

A powerful Python tool for transcribing YouTube videos to text using OpenAI's Whisper model. Features automatic language detection, multiple output formats, and intelligent system resource management.

## Features

- **Universal Language Support** - Auto-detect or specify from dozens of languages
- **Multiple Output Formats** - Text with timestamps, plain text, JSON, and SRT subtitles
- **Smart Resource Management** - Automatically optimizes settings based on available RAM and CPU
- **Resume Capability** - Session management allows resuming interrupted transcriptions
- **Progress Tracking** - Real-time progress monitoring with ETA calculations
- **Memory Monitoring** - Prevents system overload with intelligent memory usage tracking
- **Audio Validation** - Comprehensive audio file validation and quality checks

## Requirements

### System Requirements
- **Python 3.8+**
- **Minimum 4GB RAM** (8GB+ recommended)
- **GPU support** optional but recommended for faster processing

### Dependencies
```

pip install yt-dlp faster-whisper colorama tqdm psutil librosa torch

```

## Installation

1. Clone the repository:
```

git clone https://github.com/yourusername/youtube-whisper-transcriber.git
cd youtube-whisper-transcriber

```

2. Install dependencies:
```

pip install -r requirements.txt

```

3. Run the transcriber:
```

python yt_transcriber.py

```

## Usage

### Interactive Mode
Simply run the script and follow the prompts:
```

python yt_transcriber.py

```

### Step-by-Step Process
1. **Enter YouTube URL** - Paste any YouTube video URL
2. **Select Language** - Choose from Finnish, English, auto-detect, or specify custom
3. **Choose Output Format** - Select your preferred output format
4. **Wait for Processing** - The tool handles download, transcription, and saving automatically

### Supported Languages
- **Auto-detect** - Automatically identifies the spoken language
- **Predefined options** - Finnish, English
- **Custom languages** - German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt), Russian (ru), Japanese (ja), Korean (ko), Chinese (zh), Arabic (ar), Hindi (hi), and many more

## Output Formats

### 1. Text with Timestamps
```

[00:00:15 → 00:00:18] Welcome to this tutorial video
[00:00:18 → 00:00:22] Today we'll be learning about Python

```

### 2. Plain Text
```

Welcome to this tutorial video. Today we'll be learning about Python...

```

### 3. JSON Format
```

{
"title": "Video Title",
"language": "en",
"duration": 1234.5,
"segments": [
{
"start": 15.0,
"end": 18.0,
"text": "Welcome to this tutorial video"
}
]
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

The tool automatically optimizes settings based on your system:

| RAM Available | Beam Size | Compute Type | Batch Size |
|---------------|-----------|--------------|------------|
| 16GB+         | 5         | float16/int8 | 8          |
| 8GB-16GB      | 3         | int8         | 4          |
| <8GB          | 1         | int8         | 2          |

## Troubleshooting

### Common Issues

**"Cannot access video"**
- Verify the YouTube URL is correct and accessible
- Check your internet connection
- Some videos may be region-restricted

**"Audio validation failed"**
- The video may not contain audio
- Audio quality might be too low (<8kHz sample rate)
- Try a different video

**"High memory usage warning"**
- Close other applications to free up RAM
- The tool will automatically use lower quality settings on systems with limited memory

**"Model loading failed"**
- Ensure you have sufficient disk space for the Whisper model
- Check internet connection for initial model download
- The tool will automatically fallback to a smaller model if needed

### Performance Tips
- **Use GPU** - Install CUDA-compatible PyTorch for faster processing
- **Choose appropriate language** - Specifying language is faster than auto-detection

## Technical Details

- **Model**: OpenAI Whisper large-v3 (universal language model)
- **Audio Processing**: Automatic VAD (Voice Activity Detection)
- **Session Management**: Automatic cleanup with resume capability
- **Memory Management**: Dynamic resource allocation based on system capabilities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for the Whisper model
- faster-whisper for the optimized implementation
- yt-dlp for YouTube audio extraction

---

**Note**: This tool is for educational and personal use. Please respect YouTube's terms of service and copyright laws when using this tool.
