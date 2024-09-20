import os
import time

import wandb
import pandas as pd

def export_wandb_run_history(entity, project, run_id, output_csv_path, api_key=None):

    # Authenticate with wandb
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key

    else:
        # It will use the WANDB_API_KEY environment variable if set
        if "WANDB_API_KEY" not in os.environ:
            raise ValueError("Wandb API key not provided and WANDB_API_KEY environment variable not set.")

    api = wandb.Api()

    # Construct the run path
    run_path = f"{entity}/{project}/{run_id}"
    print(f"Accessing run: {run_path}")

    run = api.run(run_path)

    history = run.scan_history()

    # 빈 리스트 생성
    data = []

    # 'client0 lastTrainTime'부터 'client99 lastTrainTime'까지의 키 리스트 생성
    client_last_train_keys = [f'client/performance/trainTime/lastTrainTime/client{num} lastTrainTime' for num in range(100)]

    rowCount = 1
    lastRowCount = 0
    # history를 반복하여 각 row에서 필요한 키 추출
    for row in history:
        print(rowCount)
        rowCount += 1
        data_dict = {
            'custom_step': row.get('custom_step'),
            '_step': row.get('_step'),
        }

        # 'client0 lastTrainTime'부터 'client99 lastTrainTime'까지의 값 추출
        for key in client_last_train_keys:
            data_dict[key] = row.get(key, '')

        data.append(data_dict)

        if rowCount // 1000 > lastRowCount:
            print('saving')
            # DataFrame 생성
            df = pd.DataFrame(data)

            if not os.path.exists(output_csv_path):
                df.to_csv(output_csv_path, index=False, mode='w', encoding='utf-8-sig')
            else:
                df.to_csv(output_csv_path, index=False, mode='a', encoding='utf-8-sig', header=False)

            lastRowCount = rowCount // 1000
            data = []

    if not os.path.exists(output_csv_path):
        df.to_csv(output_csv_path, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(output_csv_path, index=False, mode='a', encoding='utf-8-sig', header=False)


if __name__ == "__main__":

    # Example usage
    # Replace these variables with your actual values
    idList = ['7vv646ua', 'ly9eqpb2']
    nameList = ['m1-case 4-2', 'm1-case 5-1']

    for i in range(len(idList)):
        print(f'working on {idList[i]} : {nameList[i]}')
        ENTITY = "jl-personal"  # e.g., "username" or "teamname"
        PROJECT = "FLT"  # e.g., "my_ml_project"
        RUN_ID = idList[i] # "fzxsr1et"  # e.g., "abc123def456"
        RUN_NAME = nameList[i] # "m1_case 3-1"
        ROOT = "./data"
        OUTPUT_CSV = f"{ROOT}/{RUN_NAME}.csv"  # Desired output file path
        API_KEY = "1c388b5685cc46b5a59e607d5a7440bad274c044"  # Optional: If not set, ensure WANDB_API_KEY env variable is set

        export_wandb_run_history(ENTITY, PROJECT, RUN_ID, OUTPUT_CSV, API_KEY)
