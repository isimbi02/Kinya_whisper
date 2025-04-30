<<<<<<< HEAD
# Kinyarwanda Speech Q&A System

This system processes Kinyarwanda speech input to answer common questions in Kinyarwanda. It uses KinyaWhisper for speech recognition and XTTS v2 for text-to-speech synthesis.

## Features

- Speech recognition for Kinyarwanda audio
- Question matching with a predefined knowledge base
- Text-to-speech response in Kinyarwanda
- User-friendly web interface with Gradio
- Support for both audio file uploads and microphone recording

## System Requirements

- Python 3.8 or higher
- 8GB+ RAM recommended
- CUDA-compatible GPU recommended for faster processing (but not required)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/isimbi02/Kinya_whisper.git
cd small_whisper
```

### 2. Create and Activate Virtual Environment

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

The system will automatically download required models when first run:
- KinyaWhisper ASR model
- XTTS v2 text-to-speech model

## Running the Application

Run the application with the following command:

```bash
python whisper.py
```

Once started, the application will be accessible in your web browser at:
```
http://localhost:7860
```

## Using the System

1. **Upload Audio Tab**
   - Upload an audio file containing a question in Kinyarwanda
   - Click "Process Audio" to analyze the question and get an answer

2. **Record Audio Tab**
   - Record your question in Kinyarwanda using your microphone
   - Click "Process Recording" to analyze the question and get an answer

3. **Review Results**
   - The system will show:
     - Transcription of your question
     - Which question in the database it matched to
     - Text answer
     - Audio response in Kinyarwanda

## Best working test Examples

For testing the system, try asking the following questions in Kinyarwanda:

- "Mwiriwe neza" (How are you?)
- "Wiga hehe" (Which school do you go to?)
- "Ese urankunda" (Do you love me?)


## Directory Structure

- `audio_samples/` - Stores user audio samples
- `transcriptions/` - Stores transcriptions of audio files
- `answers/` - Stores text and audio answers

## Customizing Voice

To use a custom voice for responses:
1. Add a reference voice file named `ref_voice.wav` to the root directory
2. The system will automatically use this voice for synthesizing responses

## Troubleshooting

1. **TTS Issues**
   - If you encounter TTS errors, the system will fall back to text-only mode
   - Check TTS installation and dependencies

2. **Audio Input Problems**
   - Ensure your microphone is properly set up
   - Try using higher quality audio recordings

3. **Model Download Failures**
   - Ensure you have a stable internet connection
   - Try running the application again

## Extending the System

To add more question-answer pairs:
1. Edit the `qa_pairs` dictionary in the `KinyarwandaQASystem` class
2. Add your new pairs in the format: `"Question in Kinyarwanda": "Answer in Kinyarwanda"`


## Acknowledgments

- KinyaWhisper model by [benax-rw](https://huggingface.co/benax-rw/KinyaWhisper)
- XTTS v2 for text-to-speech synthesis
=======
# Kinya_whisper
>>>>>>> bfd5ed16f5bc9c7c474b4c28df2fe155169d9cea
