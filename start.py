import os
import json
import asyncio
import sounddevice as sd
import numpy as np
import base64
from dotenv import load_dotenv
import traceback
import sys
import aiohttp
from aiohttp import WSMsgType
import queue
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv(override=True)

# Audio settings
CHANNELS = 1
RATE = 24000
CHUNK = 1024
SILENCE_THRESHOLD = -30  # in dB
MIN_SILENCE_DURATION = 0.5  # in seconds
VOLUME = 1.0

# WebSocket URL
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"

# Global variables
is_recording = False
is_model_talking = False
audio_output_stream = None
ws = None
audio_output_queue = queue.Queue()
loop = None
executor = ThreadPoolExecutor(max_workers=1)
total_cost = 0
silence_counter = 0
current_recording = []
audio_output_buffer = np.array([], dtype=np.float32)
audio_event_queue = asyncio.Queue()
audio_event_task = None

# Utility functions


def debug_log(message):
    if os.getenv("DEBUG") and os.getenv("DEBUG").lower() != "false":
        print(f"DEBUG: {message}", file=sys.stderr)


def error_log(message):
    print(f"ERROR: {message}", file=sys.stderr)


def select_input_device():
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]

    print("Available input devices:")
    for i, device in enumerate(input_devices):
        print(f"{i}: {device['name']}")

    while True:
        try:
            selection = int(input(f"Select input device (0-{len(input_devices)-1}): "))
            if 0 <= selection < len(input_devices):
                return input_devices[selection]['index']
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def calculate_db(audio_chunk, dtype='int16'):
    if dtype == 'int16':
        float_data = audio_chunk.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(float_data**2))
        epsilon = 1e-10
        db = 20 * np.log10(max(rms, epsilon))
    else:
        rms = np.sqrt(np.mean(audio_chunk**4))
        db = 20 * np.log10(rms) if rms > 0 else -np.inf
    return db

# Audio callbacks


def audio_input_callback(indata, frames, time, status):
    global is_recording, silence_counter, current_recording, loop, audio_output_buffer, audio_output_stream, audio_output_queue

    if status:
        debug_log(f"Audio callback status: {status}")

    try:
        db_level = calculate_db(indata)

        if db_level > SILENCE_THRESHOLD:
            if not is_recording:
                debug_log("Speech detected, starting recording...")
                is_recording = True
                current_recording = [indata.copy()]

                if not audio_output_queue.empty():
                    while not audio_output_queue.empty():
                        audio_output_queue.get_nowait()

                    if is_model_talking:
                        loop.call_soon_threadsafe(audio_event_queue.put_nowait, {"type": "response.cancel"})
            else:
                current_recording.append(indata.copy())
            silence_counter = 0

            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(indata).decode('utf-8')
            }
            loop.call_soon_threadsafe(audio_event_queue.put_nowait, audio_event)

        else:
            if is_recording:
                silence_counter += 1
                if silence_counter >= int(MIN_SILENCE_DURATION * RATE / CHUNK):
                    debug_log("Silence detected, stopping recording...")
                    is_recording = False
                    silence_counter = 0

                    loop.call_soon_threadsafe(audio_event_queue.put_nowait, {"type": "input_audio_buffer.commit"})
                    loop.call_soon_threadsafe(audio_event_queue.put_nowait, {"type": "response.create"})

                    current_recording = []
                else:
                    current_recording.append(indata.copy())
                    audio_event = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(indata).decode('utf-8')
                    }
                    loop.call_soon_threadsafe(audio_event_queue.put_nowait, audio_event)

    except Exception as e:
        debug_log(f"Error in audio callback: {e}")
        debug_log(traceback.format_exc())


def audio_output_callback(outdata, frames, time, status):
    global VOLUME, audio_output_buffer
    if status:
        debug_log(f"Audio output callback status: {status}")
    try:
        if len(audio_output_buffer) < len(outdata):
            while len(audio_output_buffer) < len(outdata):
                if audio_output_queue.empty():
                    break
                chunk = audio_output_queue.get_nowait()
                audio_output_buffer = np.concatenate((audio_output_buffer, chunk.flatten()))

        if len(audio_output_buffer) > 0:
            samples_to_play = min(len(audio_output_buffer), len(outdata))
            outdata[:samples_to_play, 0] = audio_output_buffer[:samples_to_play] * VOLUME
            audio_output_buffer = audio_output_buffer[samples_to_play:]

            if samples_to_play < len(outdata):
                outdata[samples_to_play:, 0] = 0
        else:
            outdata[:] = 0

        if len(audio_output_buffer) > 0:
            debug_log(
                f"Played {len(outdata)} samples, buffer size: {len(audio_output_buffer)}, queue size: {audio_output_queue.qsize()}")
    except Exception as e:
        debug_log(f"Error in audio output callback: {e}")
        debug_log(traceback.format_exc())
        outdata[:] = 0

# WebSocket and event handling


async def send_audio_event(audio_event):
    global ws
    if ws and not ws.closed:
        debug_log(f"ws -> {audio_event['type']}")
        await ws.send_str(json.dumps(audio_event))


async def handle_server_events():
    global is_model_talking, ws, audio_output_stream, total_cost

    while True:
        try:
            message = await asyncio.wait_for(ws.receive(), timeout=30.0)

            if message.type == aiohttp.WSMsgType.TEXT:
                event = json.loads(message.data)

                debug_log("ws <- " + event["type"])

                if event["type"] == "response.audio.delta":
                    handle_audio_delta(event)
                elif event["type"] == "response.done":
                    handle_response_done(event)
                elif event["type"] == "response.text.delta":
                    print(event["delta"], end="", flush=True)
                elif event["type"] == "response.audio_transcript.delta":
                    print(f"{event['delta']}", end="", flush=True)
                elif event["type"] == "error":
                    print(f"Error: {event['error']['message']}")
                elif event["type"] == "session.created":
                    debug_log(f"Session created: {event['session']}")
                elif event["type"] == "session.updated":
                    debug_log(f"Session updated: {event['session']}")
                    loop.call_soon_threadsafe(audio_event_queue.put_nowait, {"type": "response.create"})
                    print("Connected!\n")
                elif event["type"] == "conversation.item.input_audio_transcription.completed":
                    print(f"> {event['transcript']}\n", end="", flush=True)
                elif event["type"] == "conversation.item.input_audio_transcription.falied":
                    print(f"> ??? {event['error']['message']}", flush=True)

                else:
                    debug_log(f"Unhandled event type: {event['type']}")

            elif message.type == aiohttp.WSMsgType.CLOSED:
                print("WebSocket connection closed")
                break
            elif message.type == aiohttp.WSMsgType.ERROR:
                print(f"WebSocket error: {message.data}")
                break

        except asyncio.TimeoutError:
            debug_log("No message received from server in 30 seconds")
        except aiohttp.ClientConnectionError:
            error_log("WebSocket connection closed")
            break
        except Exception as e:
            error_log(f"Error in handle_server_events: {e}")
            debug_log(traceback.format_exc())


def handle_audio_delta(event):
    global is_model_talking, audio_output_stream
    if not is_model_talking:
        is_model_talking = True
        debug_log("Model is talking...")
        if audio_output_stream is None or not audio_output_stream.active:
            audio_output_stream = sd.OutputStream(
                samplerate=RATE, channels=CHANNELS, callback=audio_output_callback,
                blocksize=CHUNK)
            audio_output_stream.start()
    debug_log("Received audio chunk")
    audio_data = base64.b64decode(event["delta"])
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    audio_array = audio_array.astype(np.float32) / 32768.0
    audio_array = audio_array.reshape(-1, 1)
    debug_log(
        f"Audio chunk size: {audio_array.shape}, min: {audio_array.min()}, max: {audio_array.max()}")

    audio_output_queue.put(audio_array)
    debug_log(f"Added chunk to queue, new size: {audio_output_queue.qsize()}")


def handle_response_done(event):
    global is_model_talking, total_cost
    is_model_talking = False
    debug_log("Model finished talking")

    input_tokens = event["response"]["usage"]["input_tokens"]
    output_tokens = event["response"]["usage"]["output_tokens"]

    price = (input_tokens * 100 + output_tokens * 200) / 1000000
    total_cost += price

    print(
        f"\n - Usage: input_tokens: {input_tokens}, output_tokens: {output_tokens}, price: ${price:.2f}, total_cost: ${total_cost:.2f}\n\n")


async def process_audio_events():
    while True:
        event = await audio_event_queue.get()
        await send_audio_event(event)
        audio_event_queue.task_done()

# Main function and setup


async def main():
    global is_recording, is_model_talking, ws, audio_output_stream, loop, audio_event_task

    loop = asyncio.get_running_loop()

    input_device = select_input_device()
    print(f"Selected input device: {sd.query_devices(input_device)['name']}")

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "OpenAI-Beta": "realtime=v1"
    }

    try:
        while True:
            try:
                debug_log("Attempting to connect to WebSocket")
                print("Connecting...")
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(WS_URL, headers=headers) as websocket:
                        ws = websocket
                        debug_log("WebSocket connected")

                        await setup_session()

                        server_task = asyncio.create_task(handle_server_events())
                        audio_event_task = asyncio.create_task(process_audio_events())

                        debug_log("Starting audio stream")
                        with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=audio_input_callback, blocksize=CHUNK, device=input_device, dtype='int16'):
                            debug_log("Audio stream started")
                            await asyncio.gather(server_task, audio_event_task)

                await wait_for_audio_completion()

            except aiohttp.ClientResponseError as e:
                debug_log(f"HTTP error during WebSocket connection: {e.status} {e.message}")
                await asyncio.sleep(1)
            except aiohttp.WSServerHandshakeError as e:
                debug_log(f"WebSocket handshake error: {e.status} {e.message}")
                await asyncio.sleep(1)
            except aiohttp.ClientConnectionError as e:
                debug_log(f"Connection error: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                debug_log(f"An error occurred in main loop: {e}")
                debug_log(traceback.format_exc())
                await asyncio.sleep(1)

    finally:
        cleanup()


async def setup_session():
    global ws
    debug_log("ws -> session.update")

    with open("prompt.txt", "r") as f:
        system_prompt = f.read().strip()

    transcription_model = None
    if os.getenv("TRANSCRIBE").lower() == "true":
        transcription_model = {
            "model": "whisper-1"
        }

    await ws.send_str(json.dumps({
        "type": "session.update",
        "session": {
            "instructions": system_prompt,
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": None,
            "input_audio_transcription": transcription_model,
        }
    }))
    debug_log("Session update sent")


async def wait_for_audio_completion():
    global audio_output_stream, audio_output_queue, audio_output_buffer
    while not audio_output_queue.empty() or len(audio_output_buffer) > 0 or (audio_output_stream and audio_output_stream.active):
        await asyncio.sleep(0.1)
    if audio_output_stream and audio_output_stream.active:
        audio_output_stream.stop()
        audio_output_stream.close()
        audio_output_stream = None
    debug_log("Audio playback completed")


def cleanup():
    global audio_event_task, audio_output_stream
    if audio_event_task:
        audio_event_task.cancel()
    executor.shutdown(wait=True)
    if audio_output_stream is not None and audio_output_stream.active:
        audio_output_stream.stop()
        audio_output_stream.close()
    while not audio_output_queue.empty():
        try:
            audio_output_queue.get_nowait()
        except queue.Empty:
            break


if __name__ == "__main__":
    try:
        debug_log("Starting main function")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        error_log(f"Unhandled exception: {e}")
        error_log(traceback.format_exc())
