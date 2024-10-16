import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_project_accuracies(projects_csv_paths, output_dir=None, testCase='', testType=''):
    num_projects = len(projects_csv_paths)

    # 색상 팔레트 설정
    colors = plt.cm.viridis(np.linspace(0, 1, num_projects))

    # 통합 그래프를 위한 Figure와 Axes 생성
    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))

    for project_idx, project_csvs in enumerate(projects_csv_paths):
        acc_data = []

        # 각 CSV 파일에서 'acc' 컬럼 추출
        for csv_path in project_csvs:
            if not os.path.isfile(csv_path):
                print(f"경고: 파일을 찾을 수 없습니다 - {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            if 'acc' not in df.columns:
                print(f"경고: 'acc' 컬럼이 존재하지 않습니다 - {csv_path}")
                continue
            acc_data.append(df['acc'].values)

        if not acc_data:
            print(f"경고: 프로젝트 {project_idx + 1}에 유효한 'acc' 데이터가 없습니다.")
            continue

        acc_array = np.array(acc_data)  # Shape: (M, num_rows)

        # 각 행에 대한 평균과 표준편차 계산
        mean_acc = np.mean(acc_array, axis=0)
        std_acc = np.std(acc_array, axis=0)
        rows = np.arange(len(mean_acc))

        # 개별 프로젝트 그래프 생성
        fig_indiv, ax_indiv = plt.subplots(figsize=(8, 5))
        ax_indiv.plot(rows, mean_acc, label=f'Project {project_idx + 1} Mean Acc', color=colors[project_idx])
        ax_indiv.fill_between(rows, mean_acc - std_acc, mean_acc + std_acc, color=colors[project_idx], alpha=0.2,
                             label='Std Dev')
        ax_indiv.set_title(f'Project {project_idx + 1} Accuracy')
        ax_indiv.set_xlabel('Row')
        ax_indiv.set_ylabel('Accuracy')
        ax_indiv.legend()
        ax_indiv.grid(True)

        # 그래프 저장 (선택 사항)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig_indiv.savefig(os.path.join(output_dir, f'project_{project_idx + 1}_accuracy.png'))
        plt.close(fig_indiv)  # 개별 그래프를 화면에 표시하지 않음

        # 통합 그래프에 추가
        ax_combined.plot(rows, mean_acc, label=f'type {project_idx}', color=colors[project_idx])
        ax_combined.fill_between(rows, mean_acc - std_acc, mean_acc + std_acc, color=colors[project_idx], alpha=0.1)

    # 통합 그래프 설정
    ax_combined.set_title(f'{testCase} - {testType} - All Projects Accuracy')
    ax_combined.set_xlabel('Row')
    ax_combined.set_ylabel('Accuracy')
    ax_combined.legend()
    ax_combined.grid(True)

    # 통합 그래프 저장 (선택 사항)
    if output_dir:
        fig_combined.savefig(os.path.join(output_dir, f'{testCase}_{testType}.png'))

    # 모든 그래프를 한 번에 표시
    plt.show()

# 예시 사용법
if __name__ == "__main__":
    # N개의 프로젝트, 각 프로젝트에 M개의 CSV 파일 경로를 리스트로 구성
    m2_case0_3 = ['m2 - case 0 - 3 - 1-1728613463',
                  'm2 - case 0 - 3 - 2-1728626151',
                  'm2 - case 0 - 3 - 3-1728643167',
                  'm2 - case 0 - 3 - 4-1728657319',
                  'm2 - case 0 - 3 - 5-1728708479']

    m2_case0_4 = ['m2 - case 0 - 4 - 1-1728486542',
                  'm2 - case 0 - 4 - 2-1728525222',
                  'm2 - case 0 - 4 - 3-1728543247',
                  'm2 - case 0 - 4 - 4-1728558264',
                  'm2 - case 0 - 4 - 5-1728575264']

    m2_case0_5 = ['m2 - case 0 - 5 - 1-1728390555',
                  'm2 - case 0 - 5 - 2-1728404821',
                  'm2 - case 0 - 5 - 3-1728438390',
                  'm2 - case 0 - 5 - 4-1728456941',
                  'm2 - case 0 - 5 - 5-1728471395']

    m2_case1_1 = ['m2 - case 1 - 1 - 1-1728723382',
                  'm2 - case 1 - 1 - 2-1728737655',
                  'm2 - case 1 - 1 - 3-1728750432',
                  'm2 - case 1 - 1 - 4-1728779560',
                  'm2 - case 1 - 1 - 5-1728792397']

    m2_case1_2 = ['m2 - case 1 - 2 - 1-1728808656',
                  'm2 - case 1 - 2 - 2-1728821770',
                  'm2 - case 1 - 2 - 3-1728838278',
                  'm2 - case 1 - 2 - 4-1728870225',
                  'm2 - case 1 - 2 - 5-1728888982']

    testType = 'pref_pre-test'
    # testType = 'pref_after-test'
    testCase = 'case 1 - 2'

    projects_global_csv_paths = [
        [
            f'../testResult/{m2_case0_3[0]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_3[1]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_3[2]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_3[3]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_3[4]}/pref_aggregate.csv',
        ],
        [
            f'../testResult/{m2_case0_4[0]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_4[1]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_4[2]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_4[3]}/pref_aggregate.csv',
            f'../testResult/{m2_case0_4[4]}/pref_aggregate.csv',
        ],
        [
            f'../testResult/{m2_case1_1[0]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_1[1]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_1[2]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_1[3]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_1[4]}/pref_aggregate.csv',
        ],
        [
            f'../testResult/{m2_case1_2[0]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_2[1]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_2[2]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_2[3]}/pref_aggregate.csv',
            f'../testResult/{m2_case1_2[4]}/pref_aggregate.csv',
        ],
    ]

    '''
    projects_csv_paths = [
        [
            f'../testResult/{project1}/0/{testType}.csv',
            f'../testResult/{project2}/0/{testType}.csv',
            f'../testResult/{project3}/0/{testType}.csv',
            f'../testResult/{project4}/0/{testType}.csv',
            f'../testResult/{project5}/0/{testType}.csv',
            f'../testResult/{project1}/1/{testType}.csv',
            f'../testResult/{project2}/1/{testType}.csv',
            f'../testResult/{project3}/1/{testType}.csv',
            f'../testResult/{project4}/1/{testType}.csv',
            f'../testResult/{project5}/1/{testType}.csv',
            f'../testResult/{project1}/2/{testType}.csv',
            f'../testResult/{project2}/2/{testType}.csv',
            f'../testResult/{project3}/2/{testType}.csv',
            f'../testResult/{project4}/2/{testType}.csv',
            f'../testResult/{project5}/2/{testType}.csv',
            f'../testResult/{project1}/3/{testType}.csv',
            f'../testResult/{project2}/3/{testType}.csv',
            f'../testResult/{project3}/3/{testType}.csv',
            f'../testResult/{project4}/3/{testType}.csv',
            f'../testResult/{project5}/3/{testType}.csv',
        ],
        [
            f'../testResult/{project1}/4/{testType}.csv',
            f'../testResult/{project2}/4/{testType}.csv',
            f'../testResult/{project3}/4/{testType}.csv',
            f'../testResult/{project4}/4/{testType}.csv',
            f'../testResult/{project5}/4/{testType}.csv',
            f'../testResult/{project1}/5/{testType}.csv',
            f'../testResult/{project2}/5/{testType}.csv',
            f'../testResult/{project3}/5/{testType}.csv',
            f'../testResult/{project4}/5/{testType}.csv',
            f'../testResult/{project5}/5/{testType}.csv',
            f'../testResult/{project1}/6/{testType}.csv',
            f'../testResult/{project2}/6/{testType}.csv',
            f'../testResult/{project3}/6/{testType}.csv',
            f'../testResult/{project4}/6/{testType}.csv',
            f'../testResult/{project5}/6/{testType}.csv',
            f'../testResult/{project1}/7/{testType}.csv',
            f'../testResult/{project2}/7/{testType}.csv',
            f'../testResult/{project3}/7/{testType}.csv',
            f'../testResult/{project4}/7/{testType}.csv',
            f'../testResult/{project5}/7/{testType}.csv',
        ],
        [
            f'../testResult/{project1}/8/{testType}.csv',
            f'../testResult/{project2}/8/{testType}.csv',
            f'../testResult/{project3}/8/{testType}.csv',
            f'../testResult/{project4}/8/{testType}.csv',
            f'../testResult/{project5}/8/{testType}.csv',
            f'../testResult/{project1}/9/{testType}.csv',
            f'../testResult/{project2}/9/{testType}.csv',
            f'../testResult/{project3}/9/{testType}.csv',
            f'../testResult/{project4}/9/{testType}.csv',
            f'../testResult/{project5}/9/{testType}.csv',
            f'../testResult/{project1}/10/{testType}.csv',
            f'../testResult/{project2}/10/{testType}.csv',
            f'../testResult/{project3}/10/{testType}.csv',
            f'../testResult/{project4}/10/{testType}.csv',
            f'../testResult/{project5}/10/{testType}.csv',
            f'../testResult/{project1}/11/{testType}.csv',
            f'../testResult/{project2}/11/{testType}.csv',
            f'../testResult/{project3}/11/{testType}.csv',
            f'../testResult/{project4}/11/{testType}.csv',
            f'../testResult/{project5}/11/{testType}.csv',
        ]
    ]
    '''

    # 그래프를 저장하고 싶다면 output_dir을 지정
    # 그래프를 단순히 화면에 표시만 원한다면 output_dir을 None으로 설정
    plot_project_accuracies(projects_global_csv_paths, output_dir='plots', testCase=testCase, testType=testType)
