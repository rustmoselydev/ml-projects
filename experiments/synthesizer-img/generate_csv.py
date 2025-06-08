import pandas as pd
from fastai.vision.all import *
import os
import json

directory_path = "./specs"
csv_path = './data.csv'

data_list = []

with os.scandir(directory_path) as entries:
    for file in entries:
        with open('data.json', 'r') as raw_json_data:
            # specs/keyboard_acoustic_007-094-025.wav.png
            print(f"processing {file.name}")
            just_name = str(file.name).split(".wav")[0]
            # print(just_name)
            json_data = json.loads(raw_json_data.read())
            record = json_data[just_name]
            # print(record)
            item = {
                "filename": file.name,
                "pitch": record["pitch"],
                "velocity": record["velocity"],
                "source": record["instrument_source"],
                "family": record["instrument_family"],
                "quality_bright": record["qualities"][0],
                "quality_dark": record["qualities"][1],
                "quality_distortion": record["qualities"][2],
                "quality_fast_decay": record["qualities"][3],
                "quality_long_release": record["qualities"][4],
                "quality_multiphonic": record["qualities"][5],
                "quality_nonlinear_env": record["qualities"][6],
                "quality_percussive": record["qualities"][7],
                "quality_reverb": record["qualities"][8],
                "quality_tempo_synced": record["qualities"][9],
            }
            data_list.append(item)

print("Finished processing JSON into a list")
print("Creating dataframe")
data_frame = pd.DataFrame(data_list)
print("Exporting CSV")
data_frame.to_csv("data.csv", index=False)
print("Success!")
