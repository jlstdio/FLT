import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import json
import imageio
from pathlib import Path

def param_visualization(visualization_dir, initial_params, final_params):
    os.makedirs(visualization_dir, exist_ok=True)

    for name in initial_params:
        initial = initial_params[name]
        final = final_params[name]

        # 파라미터의 형태에 따라 2D 형태로 변환
        if initial.ndim > 2:
            # 예: Conv 레이어의 경우 (out_channels, in_channels, height, width)
            # 첫 두 축을 결합하여 2D로 변환
            initial_2d = initial.reshape(initial.shape[0], -1)
            final_2d = final.reshape(final.shape[0], -1)
        elif initial.ndim == 1:
            # 예: Bias 벡터
            initial_2d = initial.reshape(1, -1)
            final_2d = final.reshape(1, -1)
        else:
            # 2D 텐서
            initial_2d = initial
            final_2d = final

        # 변화량 계산 (절대 차이)
        difference = final_2d - initial_2d

        # 히트맵 그리기
        plt.figure(figsize=(10, 8))
        sns.heatmap(difference, cmap='coolwarm', center=0)
        plt.title(f'Parameter Change Heatmap: {name}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Parameter Dimension')

        # 파일 이름에 레이어 이름을 포함
        safe_name = name.replace('.', '_')  # 파일 이름에 점(.)이 있을 경우 언더스코어로 대체
        png_path = os.path.join(visualization_dir, f"{safe_name}_change.png")
        json_path = os.path.join(visualization_dir, f"{safe_name}_change.json")

        # PNG 파일로 저장
        plt.savefig(png_path)
        plt.close()
        # print(f"Saved parameter change heatmap for '{name}' at '{png_path}'")

        # JSON 파일로 저장
        # NumPy 배열을 리스트로 변환하여 JSON 직렬화 가능하게 함
        difference_list = difference.tolist()
        with open(json_path, 'w') as json_file:
            json.dump({
                'layer': name,
                'difference': difference_list
            }, json_file)
        # print(f"Saved parameter change data for '{name}' at '{json_path}'")


def create_gif_from_json(json_dir, gif_output_path, layer_name, num_rounds):
    """
    특정 레이어의 파라미터 변화 데이터를 기반으로 GIF를 생성합니다.

    :param json_dir: JSON 파일들이 저장된 디렉토리 경로
    :param gif_output_path: 생성될 GIF 파일의 경로
    :param layer_name: GIF를 생성할 레이어의 이름
    :param num_rounds: 총 라운드 수
    """
    images = []

    for round_num in range(1, num_rounds + 1):
        round_dir = os.path.join(json_dir, f"round_{round_num}")
        json_path = os.path.join(round_dir, f"{layer_name.replace('.', '_')}_change.json")

        if not os.path.exists(json_path):
            print(f"JSON file not found for round {round_num}: {json_path}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)
            difference = np.array(data['difference'])

        # 히트맵 생성
        plt.figure(figsize=(10, 8))
        sns.heatmap(difference, cmap='coolwarm', center=0)
        plt.title(f'Round {round_num} - Parameter Change: {layer_name}')
        plt.xlabel('Parameter Index')
        plt.ylabel('Parameter Dimension')

        # 임시 파일로 저장
        temp_png = os.path.join(json_dir, f"temp_round_{round_num}.png")
        plt.savefig(temp_png)
        plt.close()

        images.append(imageio.imread(temp_png))

        # 임시 파일 삭제
        os.remove(temp_png)

    if images:
        imageio.mimsave(gif_output_path, images, duration=1)  # duration은 각 프레임 간 시간 (초)
        print(f"Saved GIF for layer '{layer_name}' at '{gif_output_path}'")
    else:
        print(f"No images to create GIF for layer '{layer_name}'")

def generate_all_gifs(base_visualization_dir, output_gif_dir, num_clients, num_rounds, layers):
    os.makedirs(output_gif_dir, exist_ok=True)

    for client_id in range(num_clients):
        client_dir = os.path.join(base_visualization_dir, f"client_{client_id}")
        for layer in layers:
            gif_path = os.path.join(output_gif_dir, f"client_{client_id}_{layer.replace('.', '_')}_change.gif")
            create_gif_from_json(client_dir, gif_path, layer, num_rounds)