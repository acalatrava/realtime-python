# OpenAI Realtime API Voice Assistant

This Python project implements a voice assistant using OpenAI's new Realtime API. It features a client-side Voice Activity Detection (VAD) system to optimize token usage and reduce costs.

## Features

- Utilizes OpenAI's Realtime API for real-time conversation
- Implements client-side Voice Activity Detection (VAD)
- Supports both text and audio modalities
- Provides real-time audio input and output
- Calculates and displays token usage and associated costs
- Allow to stop the assistant when start talking again

## Sample video

https://github.com/user-attachments/assets/c40001bc-198b-4f5b-b98a-49b8984601ed

## Why Client-side VAD?

This implementation uses Voice Activity Detection (VAD) on the client side, which offers several advantages:

1. **Cost Efficiency**: By only sending audio to OpenAI when speech is detected, you significantly reduce the number of tokens processed, lowering your API usage costs.

2. **Reduced Latency**: Client-side VAD allows for quicker response times as it doesn't rely on server-side processing to determine when speech has ended.

3. **Bandwidth Optimization**: Only relevant audio data is transmitted, reducing bandwidth usage.

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in the `.env` file
4. Run the application:
   ```
   python start.py
   ```

## Configuration

You can customize various settings in the `start.py` file, including:

- Silence threshold for VAD
- Minimum silence duration

You can also modify the `prompt.txt` file to change the prompt of the assistant.

## Usage

After starting the application, speak into your microphone. The system will detect your voice, process your speech, and provide both text and audio responses from the AI assistant.

## Note

This project is designed for educational and experimental purposes. Make sure you keep your API key private and do not expose it to the public.

## License

[MIT License](LICENSE)
