import gradio as gr
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import os
from datetime import datetime

# Initialize the CosyVoice model
cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M')

# Define function to process user input and generate TTS
def generate_tts(gender, emotion, text_to_speak):
    # Construct the path to the selected prompt speech audio
    input_path = f'/mnt/user/bufan/zwt/CosyVoice/elderly_data/input/{gender}/{emotion}.wav'

    # Load the prompt speech audio
    prompt_speech_16k = load_wav(input_path, 16000)

    # Create output directory if it doesn't exist
    output_directory = '/mnt/user/bufan/zwt/CosyVoice/elderly_data/output'
    os.makedirs(output_directory, exist_ok=True)

    # Add timestamp to avoid overwriting files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(output_directory, f'{gender}_{emotion}_{timestamp}_output.wav')

    # Generate TTS for the user's input text
    output = cosyvoice.inference_zero_shot(text_to_speak, text_to_speak, prompt_speech_16k)
    torchaudio.save(output_filename, output['tts_speech'], 22050)

    return output_filename

# Define Gradio interface
gender_options = ['male', 'female']
emotion_options = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'disgusted']

interface = gr.Interface(
    fn=generate_tts,
    inputs=[
        gr.Dropdown(label="Select Gender", choices=gender_options),
        gr.Dropdown(label="Select Emotion", choices=emotion_options),
        gr.Textbox(label="Enter Text for TTS", placeholder="Type your text here..."),
    ],
    outputs=gr.Audio(label="Generated Speech", type="filepath"),  # Ensure type is 'filepath'
    live=False,  # Set live to False so it only generates on button click
    title="ElderlyVoice TTS Generation",
    description="Select gender and emotion, enter the text you want to convert to speech, and click 'Generate' to produce the audio.",
)

# Launch the Gradio interface
interface.launch()
