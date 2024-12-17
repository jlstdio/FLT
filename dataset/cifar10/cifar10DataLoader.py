import numpy as np
import pickle
import os
import urllib.request
import tarfile
import shutil


class cifar10Dataloader(object):
    CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    ARCHIVE_NAME = 'cifar-10-python.tar.gz'

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

        # Ensure the data directory exists
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory {self.data_dir}")

        # Check if all required files are present; if not, download and extract
        if not self._check_files_exist():
            print("Required CIFAR-10 files not found. Downloading dataset...")
            self._download_and_extract()
            print("Download and extraction complete.")
        else:
            print("All CIFAR-10 files are present.")

    def _check_files_exist(self):
        """
        Check if all required CIFAR-10 batch files exist in the data directory.

        Returns:
            bool: True if all files exist, False otherwise.
        """
        for batch_file in self.batch_files + [self.test_file]:
            file_path = os.path.join(self.data_dir, batch_file)
            if not os.path.isfile(file_path):
                print(f"Missing file: {file_path}")
                return False
        return True

    def _download_and_extract(self):
        """
        Download the CIFAR-10 dataset and extract it into the data directory.
        """
        archive_path = os.path.join(self.data_dir, self.ARCHIVE_NAME)

        # Download the dataset
        print(f"Downloading CIFAR-10 dataset from {self.CIFAR10_URL}...")
        try:
            urllib.request.urlretrieve(self.CIFAR10_URL, archive_path, self._download_progress)
            print("\nDownload finished.")
        except Exception as e:
            raise RuntimeError(f"Failed to download CIFAR-10 dataset: {e}")

        # Extract the archive
        print("Extracting the dataset...")
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=self.data_dir)
            print("Extraction complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to extract CIFAR-10 dataset: {e}")
        finally:
            # Optionally, remove the archive to save space
            if os.path.exists(archive_path):
                os.remove(archive_path)
                print(f"Removed archive {archive_path}")

            # Move extracted files to data_dir if they are in a subdirectory
            extracted_dir = os.path.join(self.data_dir, 'cifar-10-batches-py')
            if os.path.isdir(extracted_dir):
                for filename in os.listdir(extracted_dir):
                    shutil.move(os.path.join(extracted_dir, filename), self.data_dir)
                os.rmdir(extracted_dir)
                print(f"Moved files from {extracted_dir} to {self.data_dir}")

    def _download_progress(self, block_num, block_size, total_size):
        """
        Display download progress.

        Args:
            block_num (int): Number of blocks transferred so far.
            block_size (int): Block size in bytes.
            total_size (int): Total size of the file.
        """
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100
        percent = min(100, percent)
        print(f"\rDownload progress: {percent:.2f}%", end='')

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
