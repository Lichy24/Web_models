import os
from flask import Flask, render_template, request
#from model.inference_orya_roi_gender import inf_rec, inf_upload
#from model.inference_gender import inf_rec, inf_upload
#from model.inference_age1 import inf_age
from model.inference_dor_dolev import recording,inference,inf_mood
from model.age_inference import inf_age
from model.gender_inference import inf_gender
from dadaNet import ConvNet
from model.Dataset import Data
from cnn_model_definition_gender import ConvNet_roi_orya
#from Binary_Age_Model.model_definition_age import Convolutional_Neural_Network_Age
#from model_definition_age import Convolutional_Neural_Network_Age
from model.language_inference import Recording_language_classification as rec
import torch
from net_V2 import pred
from net import Convolutional_Speaker_Identification
import sounddevice
from scipy.io.wavfile import write

current_dir = os.getcwd()

app = Flask(__name__)

def selected_models(request):
    print(request.form.getlist('models_selection'))
    return request.form.getlist('models_selection')

saved_model = {"gender":"model_gender_orya_roi.pth","age":"age_Binary_Model-e_11_Weights.pth","mood":"model_dor_dolev.pth","language_identification":"model_language_identification.pth"}


import wave
import numpy as np

def save_wav_channel(fn, wav, channel):
    '''
    Take Wave_read object as an input and save one of its
    channels into a separate .wav file.
    '''
    # Read data
    nch   = wav.getnchannels()
    depth = wav.getsampwidth()
    wav.setpos(0)
    sdata = wav.readframes(wav.getnframes())

    # Extract channel data (24-bit data not supported)
    typ = { 1: np.uint8, 2: np.uint16, 4: np.uint32 }.get(depth)
    if not typ:
        raise ValueError("sample width {} not supported".format(depth))
    if channel >= nch:
        raise ValueError("cannot extract channel {} out of {}".format(channel+1, nch))
    print ("Extracting channel {} out of {} channels, {}-bit depth".format(channel+1, nch, depth*8))
    data = np.fromstring(sdata, dtype=typ)
    ch_data = data[channel::nch]

    # Save channel to a separate file
    outwav = wave.open(fn, 'w')
    outwav.setparams(wav.getparams())
    outwav.setnchannels(1)
    outwav.writeframes(ch_data.tostring())
    outwav.close()



def record(filename,selected_models):
    sr = 16000
    sec = 4
    print("Recording")

    rec = sounddevice.rec(int(sec * sr), samplerate=sr, channels=1)
    sounddevice.wait()
    write(filename + ".wav", sr, rec)
    return send_to_models(filename + ".wav",selected_models)


def send_to_models(filename,selected_models):
    if not selected_models:
        return "No Model has been selected."
    result = ""
    if "gender" in selected_models:
        model_gender = load_model(saved_model["gender"])
        result += "Gender Model Result: \n\n"
        result += inf_gender(model_gender,filename) + "\n\n"
    if "age" in selected_models:
        model_age = load_model(saved_model["age"])
        result += "Age Model Result: \n\n"
        result += inf_age(model_age,filename) + "\n\n"
    if "mood" in selected_models:
        model_mood = load_model(saved_model["mood"])
        result += "Mood Model Result: \n\n"
        result += inf_mood(model_mood,filename) + "\n\n"
    if "accent" in selected_models:
        result += "Accent Model Result: \n"
        result += pred(filename) + "\n\n"
    if "language_identification" in selected_models:
        result += "Language Identification Model Result: \n\n"
        result += rec.get_string_of_ans('model/'+saved_model["language_identification"],filename) + "\n\n"


    print(result)
    return result






def load_model(name):
    model = torch.load(f"model/{name}", map_location=torch.device('cpu'))
    model.eval()

    return model



@app.route('/')
def upload():
    return render_template("index.html")


@app.route('/index', methods=['POST', 'GET'])
def send_text():
    if request.method == 'POST':
        if request.form['submit_button'] == 'rec':
            result = render_template("index.html", pred=record("recording",selected_models(request)))
            os.remove('recording.wav')
            return result

        elif request.form['submit_button'] == 'file':
            print("Upload file")
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                uploaded_file.save(uploaded_file.filename)
            print(uploaded_file.filename)
            result = render_template("index.html",
                                     pred=send_to_models(uploaded_file.filename,selected_models(request)))
            if uploaded_file.filename != '':
                os.remove(f'{uploaded_file.filename}')

            return result

        elif request.form['submit_button'] == 'clean':
            return render_template("index.html", pred="")

        else:
            print("error")
            return render_template("index.html", pred="error")


def upload_file():
    print("file")



if __name__ == '__main__':
    """
    NOTE: using WSL which does not support USB ports hence cannot notice any microphones connected,
    there for need to run the recording code on WINDOWS OS and use saved wav file to upload to models using LINUX OS.

    already working:
    4 models which can select which model to use

    need help with models:
    #language identification does not recevie a wav file

    need to run:
    #new age model
    """
    #WAV_FILENAME = r"C:\Users\ADILI\Downloads\you-call-that-fun.wav"
    #wav = wave.open(WAV_FILENAME)
    #save_wav_channel(r'C:\Users\ADILI\Downloads\you-call-that-fun.wav', wav, 0)
    #save_wav_channel(r'C:\Users\ADILI\Downloads\you-call-that-fun.wav', wav, 1)
    app.run(debug=True)
