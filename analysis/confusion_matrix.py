import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ROOT_DIR = './testResult/m2 - case 1 - 2-1727514950'
# ROOT_DIR = './testResult/m2 - case 2 - 1-1727539558'
# ROOT_DIR = './testResult/m2 - case 2 - 2-1727588309'
# ROOT_DIR = './testResult/m2 - case 3 - 1-1727631960'
# ROOT_DIR = './testResult/m2 - case 3 - 2-1727658718'
ROOT_DIR = './testResult/m2 - case 3 - 12-1727677458'
NUM_CLIENTS = 10


def extract_last_target_output(csv_path):
    df = pd.read_csv(csv_path)
    target_cols = [col for col in df.columns if col.startswith('target_')]
    output_cols = [col for col in df.columns if col.startswith('output_')]

    if not target_cols or not output_cols:
        raise ValueError(f"No target or output columns found in {csv_path}")

    # 마지막 target과 output 컬럼 선택
    last_target_col = target_cols[-1]
    last_output_col = output_cols[-1]

    # last_target과 last_output 데이터를 추출
    last_target = df[last_target_col].values
    last_output = df[last_output_col].values

    # 문자열을 숫자 배열로 변환
    def parse_array_column(column_data):
        parsed = []
        for row in column_data:
            # 문자열에서 불필요한 문자 제거 후 숫자 변환
            row = row.strip('[]').replace('\n', ' ').replace(',', ' ')
            # 여러 공백을 하나로
            row = ' '.join(row.split())
            # 숫자로 변환
            try:
                numbers = [float(num) for num in row.split()]
                parsed.append(numbers)
            except ValueError as ve:
                raise ValueError(f"Error parsing row: {row}") from ve
        return np.array(parsed)

    # last_target과 last_output을 숫자 배열로 변환
    try:
        last_target = parse_array_column(last_target)
        last_output = parse_array_column(last_output)
    except ValueError as e:
        raise ValueError(f"Error parsing arrays in {csv_path}: {e}")

    # hot encoding -> single number
    if last_target.ndim == 1:
        # 단일 클래스일 경우
        target_argmax = last_target
    else:
        target_argmax = np.argmax(last_target, axis=1)

    if last_output.ndim == 1:
        # 단일 클래스일 경우
        output_argmax = last_output
    else:
        output_argmax = np.argmax(last_output, axis=1)

    return target_argmax, output_argmax


def plot_confusion_matrix(cm, title, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


# 혼동 행렬을 저장할 디렉토리 생성
PLOTS_DIR = os.path.join(ROOT_DIR, 'confusion_matrices')
os.makedirs(PLOTS_DIR, exist_ok=True)

## 각 클라이언트의 after-test와 pre-test에 대한 혼동 행렬
for client_id in range(NUM_CLIENTS):
    client_dir = os.path.join(ROOT_DIR, str(client_id))
    for test_type in ['pre-test.csv', 'after-test.csv']:
        csv_path = os.path.join(client_dir, test_type)
        if os.path.exists(csv_path):
            try:
                target, output = extract_last_target_output(csv_path)
                cm = confusion_matrix(target, output)
                title = f"Client {client_id} - {test_type} Confusion Matrix"
                # 시각화 및 저장
                save_filename = f"client_{client_id}_{test_type.replace('.csv', '')}_confusion_matrix.png"
                save_path = os.path.join(PLOTS_DIR, save_filename)
                plot_confusion_matrix(cm, title, save_path=save_path)
                print(f"Saved confusion matrix to {save_path}")
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
        else:
            print(f"{csv_path} does not exist.")

## aggregate.csv 파일의 혼동 행렬
aggregate_csv = os.path.join(ROOT_DIR, 'aggregate.csv')
if os.path.exists(aggregate_csv):
    try:
        target, output = extract_last_target_output(aggregate_csv)
        cm = confusion_matrix(target, output)
        title = "Aggregate.csv Confusion Matrix"
        # 시각화 및 저장
        save_filename = "aggregate_confusion_matrix.png"
        save_path = os.path.join(PLOTS_DIR, save_filename)
        plot_confusion_matrix(cm, title, save_path=save_path)
        print(f"Saved confusion matrix to {save_path}")
    except Exception as e:
        print(f"Error processing {aggregate_csv}: {e}")
else:
    print(f"{aggregate_csv} does not exist.")

## 클라이언트 0~8의 after-test와 pre-test에 대한 평균 혼동 행렬과 클라이언트 9의 혼동 행렬 생성
cm_sum_pre = None
cm_sum_after = None
count_pre = 0
count_after = 0

for client_id in range(NUM_CLIENTS):
    client_dir = os.path.join(ROOT_DIR, str(client_id))
    for test_type in ['pre-test.csv', 'after-test.csv']:
        csv_path = os.path.join(client_dir, test_type)
        if os.path.exists(csv_path):
            try:
                target, output = extract_last_target_output(csv_path)
                cm = confusion_matrix(target, output)
                if client_id < 9:
                    if test_type == 'pre-test.csv':
                        cm_sum_pre = cm_sum_pre + cm if cm_sum_pre is not None else cm
                        count_pre += 1
                    else:
                        cm_sum_after = cm_sum_after + cm if cm_sum_after is not None else cm
                        count_after += 1
                else:
                    title = f"Client {client_id} - {test_type} Confusion Matrix"
                    # 시각화 및 저장
                    save_filename = f"client_{client_id}_{test_type.replace('.csv', '')}_confusion_matrix.png"
                    save_path = os.path.join(PLOTS_DIR, save_filename)
                    plot_confusion_matrix(cm, title, save_path=save_path)
                    print(f"Saved confusion matrix to {save_path}")
            except Exception as e:
                print(f"Error processing {csv_path}: {e}")
        else:
            print(f"{csv_path} does not exist.")

# 평균 혼동 행렬 계산 및 출력
if count_pre > 0 and count_after > 0:
    avg_cm_pre = cm_sum_pre / count_pre
    avg_cm_after = cm_sum_after / count_after
    # 정수형으로 변환
    avg_cm_pre = avg_cm_pre.astype(int)
    avg_cm_after = avg_cm_after.astype(int)

    # 평균 혼동 행렬 시각화 및 저장
    title_pre = "Average Pre-test Confusion Matrix (Clients 0-8)"
    save_filename_pre = "average_pre_test_confusion_matrix_clients_0_8.png"
    save_path_pre = os.path.join(PLOTS_DIR, save_filename_pre)
    plot_confusion_matrix(avg_cm_pre, title_pre, save_path=save_path_pre)
    print(f"Saved average pre-test confusion matrix to {save_path_pre}")

    title_after = "Average After-test Confusion Matrix (Clients 0-8)"
    save_filename_after = "average_after_test_confusion_matrix_clients_0_8.png"
    save_path_after = os.path.join(PLOTS_DIR, save_filename_after)
    plot_confusion_matrix(avg_cm_after, title_after, save_path=save_path_after)
    print(f"Saved average after-test confusion matrix to {save_path_after}")
else:
    print("클라이언트 0~8의 pre-test 또는 after-test 데이터가 부족합니다.")
