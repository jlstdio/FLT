import pandas as pd
import os
import sys
import matplotlib.pyplot as plt


def plot_all_last_train_time_frequency(root, fileList, title):
    frequency = []
    fileName = []

    for f in fileList:
        csv_file_path = f'{root}/{f}'

        df = pd.read_csv(csv_file_path)
        fileName.append(f.split('.')[0])

        last_train_time_cols = [col for col in df.columns if 'client/performance/trainTime/lastTrainTime' in col]
        last_train_time_df = df[last_train_time_cols]
        all_last_train_times = last_train_time_df.melt(value_name='lastTrainTime')['lastTrainTime']

        all_last_train_times = all_last_train_times.dropna()  # 결측치 제거 (있을 경우)

        # input nanosecond (e.g. 731942518.0) -> sescond (e.g. 731.942518)
        all_last_train_times = all_last_train_times / 1500000  # slice the chunk into 0.5 second
        all_last_train_times = all_last_train_times.round()  # round up

        frequency.append(all_last_train_times.value_counts().sort_index())

    plt.figure(figsize=(15, 8))
    for plotD in frequency:
        plotD.plot(kind='line')
    plt.legend(fileName, loc='upper right')
    plt.xlabel(f'time (0.5 sec)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# 사용 예시
if __name__ == "__main__":
    # fileList = ['m1-case 1-1.csv', 'm1-case 1-2.csv', 'm1-case 1-3.csv', 'm1-case 1-4.csv']
    # fileList = ['m1-case 2-1.csv', 'm1-case 2-2.csv']
    fileList = ['m1-case 3-1.csv', 'm1-case 3-2.csv', 'm1-case 3-3.csv']
    root = f'./data'
    plot_all_last_train_time_frequency(root=root, fileList=fileList, title='m1-case1-(1~4)')
