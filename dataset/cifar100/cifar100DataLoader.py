import numpy as np
import pickle
import os
import urllib.request
import tarfile


class cifar100Dataloader(object):
    def __init__(self, data_dir='', normalize=True):
        self.data_dir = data_dir
        self.batch_files = ['train']
        self.test_file = 'test'
        self.normalize = normalize
        self.mean = None
        self.std = None
        self.url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.filename = 'cifar-100-python.tar.gz'
        self.extract_dir = 'cifar-100-python'

        # 데이터 디렉토리에 CIFAR-100 파일이 있는지 확인하고 없으면 다운로드
        if not self._check_files():
            self._download_and_extract()

    def _check_files(self):
        # train과 test 파일이 존재하는지 확인
        train_path = os.path.join(self.data_dir, self.extract_dir, 'train')
        test_path = os.path.join(self.data_dir, self.extract_dir, 'test')
        return os.path.exists(train_path) and os.path.exists(test_path)

    def _download_and_extract(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        file_path = os.path.join(self.data_dir, self.filename)

        # 파일 다운로드
        print("Downloading CIFAR-100 dataset...")
        urllib.request.urlretrieve(self.url, file_path)
        print("Download complete.")

        # 파일 추출
        print("Extracting files...")
        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=self.data_dir)
        print("Extraction complete.")

        # 다운로드한 tar 파일 삭제 (선택 사항)
        os.remove(file_path)

    def read_batch(self, file):
        file_path = os.path.join(self.data_dir, self.extract_dir, file)
        with open(file_path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        data = dict[b'data']
        labels = dict[b'fine_labels']  # CIFAR-100에서는 'fine_labels' 사용
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
