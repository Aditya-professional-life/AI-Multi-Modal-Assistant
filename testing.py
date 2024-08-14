import os
import time
import re
from PIL import ImageGrab, Image
import cv2
import pyperclip
import pyttsx3
import speech_recognition as sr
from faster_whisper import WhisperModel
from groq import Groq
import google.generativeai as genai

wake_word = "bro"
groq_client = Groq(api_key="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")

genai.configure(api_key="AIzaSyAL8p-6eRcdNC4PE5qRZAjLgsv8b51M-ag")
web_cam = cv2.VideoCapture(0)

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previously generated images. Just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your response clear and concise, avoiding any verbosity.'
)
convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_setting = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    }
]

# Initialize Generative Model
model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_setting)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device="cpu",
    compute_type="int8",
    cpu_threads=num_cores//2,
    num_workers=num_cores//2
)

def wav_to_text(audio_path):
    segments,_ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

r = sr.Recognizer()
source = sr.Microphone()

def call_back(recognizer, audio):
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path,'wb') as f:
        f.write(audio.get_wav_data())
    prompt_text = wav_to_text(prompt_audio_path)
    clean_prompt = extract_prompt(prompt_text,wake_word)
    
    vision_context = None

    if clean_prompt:
        print(f"USER: {clean_prompt}")
        call = function_call(clean_prompt)
        if 'take screenshot' in call:
            print("Taking screenshot")
            take_screenshot()
            vision_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')
        elif 'capture webcam' in call:
            print("Capturing WebCam")
            web_cam_capture()
            vision_context = vision_prompt(prompt=clean_prompt, photo_path='webcam.jpg')
        elif 'extract clipboard' in call:
            print("Extracting clipboard text.")
            paste = get_clipboard_text()
            clean_prompt = f'{clean_prompt} \n\n CLIPBOARD CONTENT: {paste}'
            vision_context = None

        else:
            vision_context = None

        response = groq_prompt(clean_prompt, vision_context)
        print(f"ASSISTANT: {response}")
        speak(response)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print("\nSay", wake_word, "followed with your prompt.\n")
    stop_listening = r.listen_in_background(source, call_back)

    while True:
        cont = input("Do you want to continue listening for voice commands? (yes/no): ").strip().lower()
        if cont != 'yes':
            stop_listening(wait_for_stop=False)
            break
        time.sleep(.5)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    system_prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead, take the user input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user. \nUSER PROMPT: {prompt}'
    )
    response = model.generate_content([system_prompt, img])
    return response.text

def groq_prompt(prompt, img_context):
    convo = [{'role': 'system', 'content': sys_msg}]  # Initialize convo with the system message
    
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
    
    convo.append({'role': 'user', 'content': prompt})

    try:
        chat_completion = groq_client.chat.completions.create(messages=convo, model="llama3-70b-8192")
        response = chat_completion.choices[0].message.content
        convo.append({'role': 'assistant', 'content': response})
        return response
    except groq.BadRequestError as e:
        print(f"Groq API Error: {e}")
        return None

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the user\'s clipboard content, '
        'taking a screenshot, capturing the webcam, or calling no function is best for a voice assistant to respond '
        'to the user\'s prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"]. \n'
        'Do not respond with anything but the most logical selection from that list with no explanation. Format the '
        'function call name exactly as I listed.'
    )
    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model="llama3-70b-8192")
    response = chat_completion.choices[0].message.content

    return response

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert("RGB")
    rgb_screenshot.save(path, quality=15)

def web_cam_capture():
    global web_cam
    web_cam = cv2.VideoCapture(0)
    
    if not web_cam.isOpened():
        print("Error: Camera did not open successfully")
        return

    path = "webcam.jpg"
    ret, frame = web_cam.read()

    if ret:
        cv2.imwrite(path, frame)
        print(f"Image captured and saved as {path}")
    else:
        print("Error: Failed to capture image")

    web_cam.release()

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if clipboard_content and isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No clipboard text to copy or clipboard content is not text.")
        return None

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def handle_text_input():
    while True:
        prompt = input("Please type your prompt: ")
        handle_prompt(prompt)
        cont = input("Do you want to continue? (yes/no): ").strip().lower()
        if cont != 'yes':
            break

def handle_prompt(prompt):
    call = function_call(prompt)
    vision_context = None

    if 'take screenshot' in call:
        print("Taking Screenshot")
        take_screenshot()
        vision_context = vision_prompt(prompt=prompt, photo_path="screenshot.jpg")
    elif 'capture webcam' in call:
        print("Capturing webcam")
        web_cam_capture()
        vision_context = vision_prompt(prompt=prompt, photo_path="webcam.jpg")
    elif 'extract clipboard' in call:
        print("Copying clipboard text")
        paste = get_clipboard_text()
        if paste is not None:
            prompt = f'{prompt}\n\nCLIPBOARD CONTENT: {paste}'
            vision_context = None
        else:
            print("No text found in clipboard")
            vision_context = None

    # Call groq_prompt to generate response based on user prompt and vision context
    response = groq_prompt(prompt=prompt, img_context=vision_context)
    print(response)
    speak(response)

def main():
    while True:
        print("Select input mode:")
        print("1. Voice")
        print("2. Text")

        choice = input("Enter your choice (1 or 2): ")

        if choice == "1":
            print("Voice input selected. Starting to listen...")
            start_listening()
        elif choice == "2":
            print("Text input selected. Please type your prompt below:")
            handle_text_input()
        else:
            print("Invalid choice. Please enter 1 or 2.")
            continue  # Restart the choice selection

        cont = input("Do you want to switch input mode or continue? (yes to switch/no to exit): ").strip().lower()
        if cont != 'yes':
            break

if __name__ == "__main__":
    main()
