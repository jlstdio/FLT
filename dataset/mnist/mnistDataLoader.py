import numpy as np
import os
import urllib.request
import gzip
import shutil
import struct


class mnistDataloader(object):
    MNIST_URL = 'http://yann.lecun.com/exdb/mnist/'
    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self, data_dir, normalize=True, fit_rgb=True):
        """
        Initialize the MNIST Data Loader.

        Args:
            data_dir (str): Directory path where MNIST data files are located.
            normalize (bool): Whether to normalize the data. Default is True.
            fit_rgb (bool): Whether to convert grayscale images to RGB. Default is True.
        """
        self.data_dir = data_dir
        self.normalize = normalize
        self.fit_rgb = fit_rgb  # Store the fit_rgb parameter
        self.mean = None
        self.std = None

        # Ensure the data directory exists
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created directory {self.data_dir}")

        # Check if all required files are present; if not, download
        if not self._check_files_exist():
            print("Required MNIST files not found. Downloading dataset...")
            self._download_and_extract()
            print("Download and extraction complete.")
        else:
            print("All MNIST files are present.")

    def _check_files_exist(self):
        """
        Check if all necessary MNIST files exist in the data directory.

        Returns:
            bool: True if all files exist, False otherwise.
        """
        all_exist = True
        for key, filename in self.FILES.items():
            file_path = os.path.join(self.data_dir, filename[:-3])  # Remove .gz for extracted files
            if not os.path.isfile(file_path):
                print(f"Missing file: {file_path}")
                all_exist = False
        return all_exist

    def _download_and_extract(self):
        """
        Download and extract the MNIST dataset.
        """
        for key, filename in self.FILES.items():
            file_path = os.path.join(self.data_dir, filename)
            url = self.MNIST_URL + filename
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, file_path, self._download_progress)
                print("\nDownload finished.")
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename}: {e}")

            # Extract the .gz file
            try:
                with gzip.open(file_path, 'rb') as f_in:
                    extracted_path = os.path.join(self.data_dir, filename[:-3])  # Remove .gz
                    with open(extracted_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                print(f"Extracted {filename} to {extracted_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to extract {filename}: {e}")

    def _download_progress(self, block_num, block_size, total_size):
        """
        Display the download progress.

        Args:
            block_num (int): Number of blocks transferred so far.
            block_size (int): Block size in bytes.
            total_size (int): Total size of the file in bytes.
        """
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100
        percent = min(100, percent)
        print(f"\rDownload progress: {percent:.2f}%", end='')

    def _read_images(self, filepath):
        """
        Read MNIST image files.

        Args:
            filepath (str): Path to the image file.

        Returns:
            numpy.ndarray: Image data with shape (num_samples, 28, 28).
        """
        with open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Invalid magic number {magic} in image file: {filepath}")
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images

    def _read_labels(self, filepath):
        """
        Read MNIST label files.

        Args:
            filepath (str): Path to the label file.

        Returns:
            numpy.ndarray: Label data with shape (num_samples,).
        """
        with open(filepath, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Invalid magic number {magic} in label file: {filepath}")
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def load_data(self):
        """
        Load and process MNIST data, including padding and normalization.

        Returns:
            tuple: ((x_train, y_train), (x_test, y_test))
                - x_train: numpy.ndarray, shape (60000, 32, 32, 1) or (60000, 32, 32, 3)
                - y_train: numpy.ndarray, shape (60000,)
                - x_test: numpy.ndarray, shape (10000, 32, 32, 1) or (10000, 32, 32, 3)
                - y_test: numpy.ndarray, shape (10000,)
        """
        # Read training data
        train_images_path = os.path.join(self.data_dir, self.FILES['train_images'][:-3])  # Remove .gz
        train_labels_path = os.path.join(self.data_dir, self.FILES['train_labels'][:-3])  # Remove .gz
        print("Loading training images and labels...")
        x_train = self._read_images(train_images_path)
        y_train = self._read_labels(train_labels_path)

        # Read test data
        test_images_path = os.path.join(self.data_dir, self.FILES['test_images'][:-3])  # Remove .gz
        test_labels_path = os.path.join(self.data_dir, self.FILES['test_labels'][:-3])  # Remove .gz
        print("Loading test images and labels...")
        x_test = self._read_images(test_images_path)
        y_test = self._read_labels(test_labels_path)

        # Pad images from 28x28 to 32x32 by adding 2 pixels on each side
        print("Padding images from 28x28 to 32x32...")
        padding = ((0, 0), (2, 2), (2, 2))  # No padding for the sample axis
        x_train = np.pad(x_train, padding, mode='constant', constant_values=0)
        x_test = np.pad(x_test, padding, mode='constant', constant_values=0)
        print(f"Padded training images shape: {x_train.shape}")  # Should be (60000, 32, 32)
        print(f"Padded test images shape: {x_test.shape}")      # Should be (10000, 32, 32)

        # Expand dimensions to add the channel axis
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
        print(f"Training images shape after adding channel: {x_train.shape}")  # (60000, 32, 32, 1)
        print(f"Test images shape after adding channel: {x_test.shape}")      # (10000, 32, 32, 1)

        # If fit_rgb is True, convert grayscale images to RGB by duplicating channels
        if self.fit_rgb:
            print("Converting grayscale images to RGB by duplicating channels...")
            x_train = np.repeat(x_train, 3, axis=-1)
            x_test = np.repeat(x_test, 3, axis=-1)
            print(f"Training images shape after RGB conversion: {x_train.shape}")  # (60000, 32, 32, 3)
            print(f"Test images shape after RGB conversion: {x_test.shape}")      # (10000, 32, 32, 3)

        # Perform normalization if enabled
        if self.normalize:
            self.mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
            self.std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
            print(f"Computed mean: {self.mean.flatten()}")
            print(f"Computed std: {self.std.flatten()}")

            # Apply normalization
            x_train = (x_train - self.mean) / self.std
            x_test = (x_test - self.mean) / self.std
            print("Normalization applied to training and test images.")

        return (x_train, y_train), (x_test, y_test)

    def get_mean_std(self):
        """
        Retrieve the computed mean and standard deviation.

        Returns:
            tuple: (mean, std)
                - mean: numpy.ndarray, shape (1, 1, 1, 1) or (1, 1, 1, 3)
                - std: numpy.ndarray, shape (1, 1, 1, 1) or (1, 1, 1, 3)
        """
        if self.mean is None or self.std is None:
            raise ValueError("Mean and std have not been computed. Please call load_data() first.")
        return self.mean, self.std
