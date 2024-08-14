import os
import time
import re
from PIL import ImageGrab, Image
import cv2
import pyperclip
import pyttsx3
import speech_recognition as sr
from groq import Groq
import google.generativeai as genai

groq_client = Groq(api_key="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")

genai.configure(api_key="AIzaSyAL8p-6eRcdNC4PE5qRZAjLgsv8b51M-ag")

wake_word = "bro"
# Initialize other components
web_cam = cv2.VideoCapture(0)
engine = pyttsx3.init()

# System message for the assistant
sys_msg = (
    'You are a multi-modal AI assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previously generated images. Just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your response clear and concise, avoiding any verbosity.'
)
convo = [{'role': 'system', 'content': sys_msg}]

# Configuration for generative model
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

# Initialize recognizer
r = sr.Recognizer()
source = sr.Microphone()

def transcribe_audio(audio):
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

def call_back(recognizer, audio):
    prompt_text = transcribe_audio(audio)
    clean_prompt = extract_prompt(prompt_text, wake_word)
    
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

        response = groq_prompt(prompt=clean_prompt, img_context=vision_context)
        print(f"ASSISTANT: {response}")
        speak(response)

def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=2)

    print("\nSay", wake_word, "followed with your prompt.\n")
    r.listen_in_background(source, call_back)

    while True:
        time.sleep(.5)

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'{re.escape(wake_word)}[\s,.?!]*([A-za-z0-9].*)'
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
    convo = []  # Initialize convo as an empty list for each function call
    
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\nIMAGE CONTEXT: {img_context}'
    
    # Append the user's prompt to convo
    convo.append({'role': 'user', 'content': prompt})

    try:
        # Attempt to create completions using groq_client
        chat_completion = groq_client.chat.completions.create(messages=convo, model="llama3-70b-8192")
        response = chat_completion.choices[0].message.content
        convo.append({'role': 'assistant', 'content': response})  # Append the response to convo
        return response
    except groq.BadRequestError as e:
        print(f"Groq API Error: {e}")  # Handle the specific error if needed
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
    if not web_cam.isOpened():
        print("Error: Camera did not open successfully")
        exit()

    path = "webcam.jpg"
    ret, frame = web_cam.read()

    if ret:
        cv2.imwrite(path, frame)
        print(f"Image captured and saved as {path}")
    else:
        print("Error: Failed to capture image")

    web_cam.release()  # Release the webcam device

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if clipboard_content and isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print("No clipboard text to copy or clipboard content is not text.")
        return None
    
def speak(text):
    engine.say(text)         # Pass the text to the engine
    engine.runAndWait()      # Wait for the speech to finish

def handle_text_input():
    prompt = input("Please type your prompt: ")
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

    response = groq_prompt(prompt=prompt, img_context=vision_context)
    print(response)
    speak(response)

def main():
    wake_word = "bro"
    
    while True:
        print("\nChoose input mode:")
        print("1. Voice Input")
        print("2. Text Input")
        choice = input("Enter 1 or 2: ")

        if choice == '1':
            print("Voice input mode activated. Say your prompt after the wake word.")
            start_listening()
        elif choice == '2':
            handle_text_input()
        else:
            print("Invalid choice. Please enter 1 or 2.")

# Start the main function
if __name__ == "__main__":
    main()
