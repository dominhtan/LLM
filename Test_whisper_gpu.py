import torch
import time
import whisper
import os

device = torch.device('cuda')
model_list = ['medium', 'large-v2']
fp16_bool = [True, False]
path = './Reports/'
file_list = os.listdir(path)
for i in model_list:
    for k in fp16_bool:
        model = whisper.load_model(name=i, device=device)
        duration_sum = 0
        for idx, j in enumerate(file_list):
            audio = whisper.load_audio(path + j, sr=16000)
            start = time.time()
            result = model.transcribe(audio, language='en', task='transcribe', fp16=k)
            end = time.time()
            duration_sum = duration_sum + end - start
        print("{} model with fp16 {} costs {:.2f}s".format(i, k, duration_sum))
        print(result["text"])
        del model
