'''

Goal:
1. 성능의 발전이 얼마나 걸렸는가
    a. Round 당 시간이 얼마나 걸렸는가
    b. 각 test 끼리 비교하여 -> 실험이 특정 accuracy에 도달하기 위해 시간이 얼마나 걸리는지

How to:
주어진 csv를 읽어들인다. 이 csv는 column으로 custom_step,_step 그리고 client/performance/trainTime/lastTrainTime/client0 부터 client/performance/trainTime/lastTrainTime/client99 까지 있다.
test5-1이라는 csv를 기준으로 각 client0 ~ client99 에 대하여 각 client가 하나의 custom_step에서 성능은 다르지만 동일한 시간에 train을 끝났다고 가정을 하고 나머지 다른 csv 파일의 데이터와 비교하여 성능 또는 시간을 비교하게 된다.
1. A csv의 데이터에서 client의 Type별로 평균치를 구한다
    a. client 0 ~ client 29 : A type | client 30 ~ client 59 : B type | client 60 ~ client 99 : C type
2. 가장 오래 걸린 type을 기준으로 나머지 type들의 실행 시간에서 몇 배 더 걸렸는지 구한다
3. 각 type 별로 type에 맞는 client에게 방금 구한 얼마나 더 걸렸는지 계산된 만큼 곱해준다.
4. 계산되어 update된 lastTrainTime(client 0 ~ 99)들을 plot한다.
5. B csv데이터에도 같은 type인 client에게 각 type에 알맞도록 얼마나 더 걸렸는지 계산된 만큼 곱해준다.
6. 계산되어 update된 B csv 에도 lastTrainTime(client 0 ~ 99)들을 plot한다.
'''

import pandas as pd
import matplotlib.pyplot as plt


def clean_timeFreq(df):
    df = df[client_cols]
    all_last_train_times = df.melt(value_name='lastTrainTime')['lastTrainTime']
    all_last_train_times = all_last_train_times.dropna()  # 결측치 제거 (있을 경우)

    # input nanosecond (e.g. 731942518.0) -> sescond (e.g. 731.942518)
    all_last_train_times = all_last_train_times / 1500000  # slice the chunk into 0.5 second
    all_last_train_times = all_last_train_times.round()  # round up

    frequency = all_last_train_times.value_counts().sort_index()

    return frequency

# 1. A CSV 읽기 및 클라이언트 타입별 평균 계산
dataFileNameRef = 'm1-case 5-1'
a_csv = pd.read_csv(f'./data/{dataFileNameRef}.csv')
client_cols = [col for col in a_csv.columns if 'client/performance/trainTime/lastTrainTime/' in col]

ref_before = clean_timeFreq(a_csv[client_cols])

type_A = client_cols[0:30]
type_B = client_cols[30:60]
type_C = client_cols[60:100]

a_csv[type_A] = a_csv[type_A].apply(pd.to_numeric, errors='coerce')
a_csv[type_B] = a_csv[type_B].apply(pd.to_numeric, errors='coerce')
a_csv[type_C] = a_csv[type_C].apply(pd.to_numeric, errors='coerce')

mean_A = a_csv[type_A].mean().mean()
mean_B = a_csv[type_B].mean().mean()
mean_C = a_csv[type_C].mean().mean()

print(f"Type A mean: {mean_A}")
print(f"Type B mean: {mean_B}")
print(f"Type C mean: {mean_C}")


# 2. 가장 오래 걸린 타입 기준으로 배수 계산
max_mean = max(mean_A, mean_B, mean_C)
print(f"가장 오래 걸린 타입의 평균: {max_mean}")

multiplier_A = mean_A / max_mean
multiplier_B = mean_B / max_mean
multiplier_C = mean_C / max_mean

print(f"Type A 배수: {multiplier_A}")
print(f"Type B 배수: {multiplier_B}")
print(f"Type C 배수: {multiplier_C}")

# 3. 각 타입별 배수를 클라이언트의 lastTrainTime에 적용
a_csv_updated = a_csv.copy()

a_csv_updated[type_A] = a_csv_updated[type_A] / multiplier_A
a_csv_updated[type_B] = a_csv_updated[type_B] / multiplier_B
a_csv_updated[type_C] = a_csv_updated[type_C] / multiplier_C

ref_after = clean_timeFreq(a_csv_updated[client_cols])

plt.figure(figsize=(15, 8))
ref_before.plot(kind='line')
ref_after.plot(kind='line')
plt.legend(['before', 'after'], loc='upper right')
plt.xlabel(f'time (0.5 sec)')
plt.ylabel('Frequency')
plt.title(f'{dataFileNameRef} ref data : before -> after')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 5. B CSV 데이터에도 동일한 방식 적용
dataFileNameCmp = 'm1-case 4-1'
b_csv = pd.read_csv(f'./data/{dataFileNameCmp}.csv')  # B CSV 파일명에 맞게 수정
b_client_cols = [col for col in b_csv.columns if 'client/performance/trainTime/lastTrainTime/' in col]

b_csv_updated = b_csv.copy()

cmp_before = clean_timeFreq(b_csv_updated[b_client_cols])

b_csv_updated[type_A] = b_csv_updated[type_A] / multiplier_A
b_csv_updated[type_B] = b_csv_updated[type_B] / multiplier_B
b_csv_updated[type_C] = b_csv_updated[type_C] / multiplier_C

cmp_after = clean_timeFreq(b_csv_updated[b_client_cols])

plt.figure(figsize=(15, 8))
cmp_before.plot(kind='line')
cmp_after.plot(kind='line')
plt.legend(['before', 'after'], loc='upper right')
plt.xlabel(f'time (0.5 sec)')
plt.ylabel('Frequency')
plt.title(f'{dataFileNameCmp} cmp data : before -> after')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 8))
ref_after.plot(kind='line')
cmp_after.plot(kind='line')
plt.legend(['ref', 'cmp'], loc='upper right')
plt.xlabel(f'time (0.5 sec)')
plt.ylabel('Frequency')
plt.title(f'{dataFileNameCmp} vs {dataFileNameRef} (time adjusted)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
