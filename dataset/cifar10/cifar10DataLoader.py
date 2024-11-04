import numpy as np
import pickle
import os


class cifar10Dataloader(object):
    def __init__(self, data_dir, normalize=True):
        """
        CIFAR-10 데이터 로더 초기화.

        Args:
            data_dir (str): CIFAR-10 데이터 파일이 위치한 디렉토리 경로.
            normalize (bool): 데이터를 정규화할지 여부. 기본값은 True.
        """
        self.data_dir = data_dir
        self.batch_files = [f'data_batch_{i}' for i in range(1, 6)]
        self.test_file = 'test_batch'
        self.normalize = normalize
        self.mean = None
        self.std = None

    def read_batch(self, file):
        with open(os.path.join(self.data_dir, file), 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        data = dict[b'data']
        labels = dict[b'labels']
        data = data.reshape(data.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
        return data, labels

    def load_data(self):
        x_train = []
        y_train = []

        # 학습 배치 파일 읽기
        for batch_file in self.batch_files:
            data, labels = self.read_batch(batch_file)
            x_train.append(data)
            y_train.append(labels)

        # 학습 데이터와 라벨을 하나의 배열로 결합
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)

        # 테스트 배치 파일 읽기
        x_test, y_test = self.read_batch(self.test_file)

        # 정규화 수행
        if self.normalize:
            self.mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
            self.std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
            print(f"Computed mean: {self.mean.flatten()}")
            print(f"Computed std: {self.std.flatten()}")

            # 정규화 적용
            x_train = (x_train - self.mean) / self.std
            x_test = (x_test - self.mean) / self.std

        return (x_train, y_train), (x_test, y_test)

    def get_mean_std(self):
        """
        계산된 평균과 표준편차를 반환.

        Returns:
            tuple: (mean, std) 각각은 (1, 1, 1, 3) 형태의 numpy 배열.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std have not been computed. Please call load_data() first.")
        return self.mean, self.std