import pandas as pd
import os
import sys
import matplotlib.pyplot as plt


def plot_all_last_train_time_frequency(csv_file_path):
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file_path)

        # 'lastTrainTime'이 포함된 컬럼만 선택
        last_train_time_cols = [col for col in df.columns if 'client/performance/trainTime/lastTrainTime' in col]

        # print(last_train_time_cols)

        # 해당 컬럼들만 추출
        last_train_time_df = df[last_train_time_cols]
        # print(last_train_time_df['client/performance/trainTime/lastTrainTime/client99 lastTrainTime'].dropna())

        # 모든 클라이언트의 lastTrainTime 데이터를 하나로 모음
        all_last_train_times = last_train_time_df.melt(value_name='lastTrainTime')['lastTrainTime']
        # print(all_last_train_times)

        # 결측치 제거 (있을 경우)
        all_last_train_times = all_last_train_times.dropna()
        all_last_train_times = all_last_train_times/10000000
        print(all_last_train_times.count())
        all_last_train_times = all_last_train_times.round()

        # 데이터가 숫자형인지 확인하고, 필요시 변환
        try:
            all_last_train_times = pd.to_numeric(all_last_train_times, errors='coerce').dropna()
        except Exception as e:
            print(f"데이터를 숫자형으로 변환하는 중 오류가 발생했습니다: {e}")
            sys.exit(1)

        if all_last_train_times.empty:
            raise ValueError("숫자형으로 변환 후 모든 'lastTrainTime' 데이터가 비어있습니다.")

        # 빈도 계산
        frequency = all_last_train_times.value_counts().sort_index()

        # 빈도 출력 (선택 사항)
        print("각 'lastTrainTime'의 빈도:")
        print(frequency)

        # 빈도 시각화 (모든 데이터)
        try:
            plt.figure(figsize=(15, 8))
            frequency.plot(kind='line', color='skyblue')
            plt.xlabel('lastTrainTime')
            plt.ylabel('Frequency')
            plt.title('Frequency of lastTrainTime across All Clients')
            plt.xticks(rotation=90)  # x축 레이블 회전
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"빈도 시각화 중 오류가 발생했습니다: {e}")

        # 추가 데이터 요약 출력 (선택 사항)
        print("\n'lastTrainTime' 통계 요약:")
        print(all_last_train_times.describe())

    except FileNotFoundError as fnf_error:
        print(f"파일 오류: {fnf_error}")
    except pd.errors.EmptyDataError:
        print("CSV 파일이 비어 있습니다.")
    except pd.errors.ParserError:
        print("CSV 파일을 파싱하는 중 오류가 발생했습니다. 파일 형식을 확인하세요.")
    except ValueError as ve:
        print(f"값 오류: {ve}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

# 사용 예시
if __name__ == "__main__":
    # CSV 파일 경로 지정
    csv_file = './data/m1_case 3-1.csv'  # 실제 파일 경로로 변경하세요

    # 함수 호출
    plot_all_last_train_time_frequency(csv_file_path=csv_file)
