import os
import torch
import shutil
import numpy as np

def analyze_error_folder(error_folder):

    for file_name in os.listdir(error_folder):
        if file_name.endswith('.pth'):
            file_path = os.path.join(error_folder, file_name)
            model = torch.load(file_path)


            for i in model:
                print(f'layer name {i}')
                # print(model[i])
                if torch.isnan(model[i]).any().item():
                    print(f"NaN found in {i} layer")

            '''
            has_nan, key_name = check_nan_in_pth(file_path)
            if has_nan:
                print(f"NaN found in file: {file_name}, key: {key_name}")
            '''


error_pth_folder = 'errorModel'
analyze_error_folder(error_pth_folder)