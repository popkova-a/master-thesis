import os
import copy
import h5py
import aedat
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Optional
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


# train dataloader change, pretrain done
# augmenter, tokenizer resolution, sensor size -> issue
class NCaltech101(Dataset):
    """
    Dataset class for the N-Caltech101 dataset.

    This dataset contains event-based data captured using a Dynamic Vision Sensor (DVS)
    and is based on the Caltech101 image dataset. Each sample consists of a sequence of
    events (t, x, y, p) and a corresponding label.

    Args:
        data_path (str): Path to the dataset directory.
        train (bool, optional): If True, creates dataset from training data.
        transform (callable, optional): A function/transform applied to an event sample.
        target_transform (callable, optional): A function/transform applied to an event label.
    """

    def __init__(self,
                 data_path: str = '/data/storage/anastasia/data/N-Caltech101',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.subfolder = self._search_subfolder(data_path)
        self.data_path = os.path.join(data_path, self.subfolder)

        # Dataset characteristics
        self.resolution = (180, 240)  # Resolution of the data (height, width) in pixels
        self.max_t = 0.325            # Maximum timestamp in seconds

        # Get sorted list of classes from directory structure
        self.classes = sorted([d for d in os.listdir(self.data_path)
                               if os.path.isdir(os.path.join(self.data_path, d))])
        self.class_folders = copy.deepcopy(self.classes)

        # Load all valid event files and corresponding labels
        self.files, self.labels = self._load_data()

        # Validate dataset consistency
        if len(self.files) != len(self.labels):
            raise ValueError(f'Mismatch between files ({len(self.files)}) and labels ({len(self.labels)})')

    def _search_subfolder(self,
                          data_path: str) -> str:
        """
        Searches for and returns the appropriate train or test subfolder within a directory.

        This helper method scans the given directory path for subfolders containing either
        'train', 'val' or 'test' in their name (depending on the instance's training mode) and
        returns the first matching subfolder found.

        Args:
            data_path: Path to the directory containing training/test subfolders.

        Returns:
            str: Name of the first found subfolder matching the requested type (train/val/test).

        Raises:
            ValueError: If no appropriate subfolder (train, val or test) is found in the directory.
        """

        if self.train:
            train_folder = [name for name in os.listdir(data_path)
                            if 'train' in name and os.path.isdir(os.path.join(data_path, name))]

            if not train_folder:
                raise ValueError(f'No train folder found in {data_path}.')
            else:
                return train_folder[0]
        else:
            val_folder = [name for name in os.listdir(data_path)
                          if 'val' in name and os.path.isdir(os.path.join(data_path, name))]
            test_folder = [name for name in os.listdir(data_path)
                           if 'test' in name and os.path.isdir(os.path.join(data_path, name))]
            if not val_folder:
                if not test_folder:
                    raise ValueError(f'No validation/test folder found in {data_path}.')
                else:
                    return test_folder[0]
            else:
                return val_folder[0]

    def _load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load event file paths and labels for the dataset.

        This method constructs lists of file paths and corresponding labels for all samples
        in the dataset.

        Returns:
            tuple: A tuple containing:
                - files (list): List of file paths for labeled samples.
                - labels (list): List of labels corresponding to `files`.
        """

        files, labels = [], []

        for class_idx, class_name in enumerate(self.class_folders):
            class_dir = os.path.join(self.data_path, class_name)

            # Get all files in class directory
            try:
                class_files = [os.path.join(class_dir, f)
                               for f in os.listdir(class_dir)
                               if f.endswith(('.npy', '.npz'))]
            except OSError:
                print(f"Warning: Could not read files for class {class_name}")
                continue

            if not class_files:
                print(f"Warning: Class {class_name} contains no valid event files")
                continue

            files.extend(class_files)
            labels.extend([class_idx] * len(class_files))

        return files, labels

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_events(event_path: str) -> torch.Tensor:
        """
        Load events from a .npy file.

        The method performs the following operations:
        1. Loads event data from numpy file
        2. Reorders columns from (x, y, t, p) to (t, x, y, p) format
        3. Converts to PyTorch tensor with float32 precision

        Args:
            event_path (str): Path to the event file.

        Returns:
            torch.Tensor: The event data as a tensor of shape [N, 4], where each row
                         contains (t, x, y, p) values.

        Raises:
            FileNotFoundError: If the specified event file doesn't exist
            ValueError: If the loaded data has incorrect dimensions
        """

        try:
            # Load events with original ordering: (x, y, t, p)
            events = np.load(event_path)

            # Validate input shape
            if events.ndim != 2 or events.shape[1] != 4:
                raise ValueError(f"Expected events with shape [N, 4], got {events.shape}")

            # Convert polarity values (-1 -> 0, 1 -> 1)
            if events[:, 3].min() < -0.5:
                events[:, 3] = (events[:, 3] > 0).astype(events.dtype)

            # Reorder columns to (t, x, y, p) using array slicing
            reordered_events = events[:, [2, 0, 1, 3]]

            # Convert to PyTorch tensor
            return torch.as_tensor(reordered_events, dtype=torch.float32)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Event file not found at path: {event_path}") from e

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve a data point from the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing:
                - events(torch.Tensor): The event data of shape [N, 4] containing (t, x, y, p) values with (0, 1) polarity.
                - label (int): The label of the data point as an integer between 0 and num_classes - 1.
        """

        file_path = self.files[idx]
        label = self.labels[idx]

        # Load the events from the file path
        events = self._load_events(event_path=file_path)

        # Apply transforms
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return events, label

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: The number of classes in the dataset.
        """

        return len(self.classes)


class NCars(NCaltech101):
    """
    Dataset class for the N-Cars dataset.

    This dataset contains event-based data captured using a Dynamic Vision Sensor (DVS)
    and is designed for car detection tasks. Each sample consists of a sequence of
    events (t, x, y, p) and a corresponding label.

    Args:
        data_path (str): Path to the dataset directory.
        train (bool, optional): If True, creates dataset from training data.
        transform (callable, optional): A function/transform applied to an event sample.
        target_transform (callable, optional): A function/transform applied to an event label.

    Notes:
        - This class inherits from `NCaltech101` and overrides the dataset-specific attributes
          (e.g., resolution, max_t) to match the N-Cars dataset.
    """

    def __init__(self,
                 data_path: str = '/data/storage/anastasia/data/N-Cars',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        # Initialize the parent class NCaltech101
        super().__init__(data_path=data_path,
                         train=train,
                         transform=transform,
                         target_transform=target_transform)

        # Dataset characteristics
        self.resolution = (100, 120)
        self.max_t = 0.1


class NImageNetMini(NCaltech101):
    """
    Dataset class for the N-ImageNet (Mini) dataset.

    This dataset contains event-based data captured using a Dynamic Vision Sensor (DVS)
    and is based on the ImageNet image dataset. Each sample consists of a sequence of
    events (t, x, y, p) and a corresponding label.

    Args:
        data_path (str): Path to the dataset directory.
        train (bool, optional): If True, creates dataset from training data.
        transform (callable, optional): A function/transform applied to an event sample.
        target_transform (callable, optional): A function/transform applied to an event label.
    """

    def __init__(self,
                 data_path: str = '/data/storage/anastasia/data/N-ImageNet',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        # Initialize the parent class NCaltech101
        super().__init__(data_path=data_path,
                         train=train,
                         transform=transform,
                         target_transform=target_transform)

        # Dataset characteristics
        self.resolution = (480, 640)
        self.max_t = 0.055

        # Extract class names for N-ImageNet Mini dataset
        self.folder2class = pd.read_csv(os.path.join(data_path, 'labels.txt'),
                                        sep=' ', header=None)
        self.folder2class = dict(zip(self.folder2class[0], self.folder2class[2]))
        self.classes = [self.folder2class[folder] for folder in self.class_folders]

    @staticmethod
    def _load_events(event_path: str) -> torch.Tensor:
        """
        Load events from a .npz file.

        The method performs the following operations:
        1. Loads event data from numpy file
        2. Reorders columns from (x, y, t, p) to (t, x, y, p) format
        3. Converts to PyTorch tensor with float32 precision

        Args:
            event_path (str): Path to the event file.

        Returns:
            torch.Tensor: The event data as a tensor of shape [N, 4], where each row
                         contains (t, x, y, p) values.

        Raises:
            FileNotFoundError: If the specified event file doesn't exist
            ValueError: If the loaded data has incorrect dimensions
        """

        try:
            # Load events with original ordering (x, y, t, p) and reorder to (t, x, y, p)
            events_np = np.load(event_path)['event_data']
            events = np.column_stack((events_np['t'],
                                      events_np['x'],
                                      events_np['y'],
                                      events_np['p']))

            # Validate input shape
            if events.ndim != 2 or events.shape[1] != 4:
                raise ValueError(f"Expected events with shape [N, 4], got {events.shape}")

            # Convert polarity values (-1 -> 0, 1 -> 1)
            if events[:, 3].min() < -0.5:
                events[:, 3] = (events[:, 3] > 0).astype(events.dtype)

            # Convert to PyTorch tensor
            return torch.as_tensor(events, dtype=torch.float32)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Event file not found at path: {event_path}") from e


class CIFAR10DVS(Dataset):
    """
    Dataset class for the CIFAR10DVS dataset.

    This dataset contains event-based data captured using a Dynamic Vision Sensor (DVS)
    and is designed for car detection tasks. Each sample consists of a sequence of
    events (t, x, y, p) and a corresponding label.

    Args:
        data_path (str): Path to the dataset directory.
        train (bool, optional): If True, creates dataset from training data.
        transform (callable, optional): A function/transform applied to an event sample.
        target_transform (callable, optional): A function/transform applied to an event label.
        seed (int, optional): Seed for the random number generator. Defaults to 42.

    Notes:
        - This class builds on spikingjelly CIFAR10DVS dataset.
    """

    def __init__(self,
                 data_path: str = '/data/storage/anastasia/data/CIFAR10DVS/events_np',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 seed: int = 42):

        self.train = train
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform

        # Dataset characteristics
        self.resolution = (128, 128)  # Resolution of the data (height, width) in pixels
        self.max_t = 1.2              # Maximum timestamp in seconds

        # Get sorted list of classes from directory structure
        self.classes = sorted([d for d in os.listdir(self.data_path)
                               if os.path.isdir(os.path.join(self.data_path, d))])
        self.class_folders = copy.deepcopy(self.classes)

        # Load all valid event files and corresponding labels
        self.files, self.labels = self._load_data()

        # Split the data into train and test
        if self.train:
            self.files, _, self.labels, _ = train_test_split(self.files, self.labels,
                                                             test_size=0.1, shuffle=True, random_state=seed)
        else:
            _, self.files, _, self.labels = train_test_split(self.files, self.labels,
                                                             test_size=0.1, shuffle=True, random_state=seed)

        # Validate dataset consistency
        if len(self.files) != len(self.labels):
            raise ValueError(f'Mismatch between files ({len(self.files)}) and labels ({len(self.labels)})')

    def _load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load event file paths and labels for the dataset.

        This method constructs lists of file paths and corresponding labels for all samples
        in the dataset.

        Returns:
            tuple: A tuple containing:
                - files (list): List of file paths for labeled samples.
                - labels (list): List of labels corresponding to `files`.
        """

        files, labels = [], []

        for class_idx, class_name in enumerate(self.class_folders):
            class_dir = os.path.join(self.data_path, class_name)

            # Get all files in class directory
            try:
                class_files = [os.path.join(class_dir, f)
                               for f in os.listdir(class_dir)
                               if f.endswith(('.npy', '.npz'))]
            except OSError:
                print(f"Warning: Could not read files for class {class_name}")
                continue

            if not class_files:
                print(f"Warning: Class {class_name} contains no valid event files")
                continue

            files.extend(class_files)
            labels.extend([class_idx] * len(class_files))

        return files, labels

    @staticmethod
    def _load_events(event_path: str) -> torch.Tensor:
        """
        Load events from a .npy file.

        The method performs the following operations:
        1. Loads event data from numpy file
        2. Reorders columns from (x, y, t, p) to (t, x, y, p) format
        3. Converts to PyTorch tensor with float32 precision

        Args:
            event_path (str): Path to the event file.

        Returns:
            torch.Tensor: The event data as a tensor of shape [N, 4], where each row
                         contains (t, x, y, p) values.

        Raises:
            FileNotFoundError: If the specified event file doesn't exist
            ValueError: If the loaded data has incorrect dimensions
        """

        try:
            # Load events with original ordering: (x, y, t, p)
            events_np = np.load(event_path)
            events = np.column_stack((events_np['t'],
                                      events_np['x'],
                                      events_np['y'],
                                      events_np['p']))

            # Validate input shape
            if events.ndim != 2 or events.shape[1] != 4:
                raise ValueError(f"Expected events with shape [N, 4], got {events.shape}")

            # Convert polarity values (-1 -> 0, 1 -> 1)
            if events[:, 3].min() < -0.5:
                events[:, 3] = (events[:, 3] > 0).astype(events.dtype)

            # Convert to PyTorch tensor
            return torch.as_tensor(events, dtype=torch.float32)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Event file not found at path: {event_path}") from e

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve a data point from the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing:
                - events(torch.Tensor): The event data of shape [N, 4] containing (t, x, y, p) values with (0, 1) polarity.
                - label (int): The label of the data point as an integer between 0 and num_classes - 1.
        """

        file_path = self.files[idx]
        label = self.labels[idx]

        # Load the events from the file path
        events = self._load_events(event_path=file_path)

        # Apply transforms
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return events, label

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: The number of classes in the dataset.
        """

        return len(self.classes)


class DailyDVS200(Dataset):
    """
    Dataset class for the DailyDVS-200 dataset.

    This dataset contains event-based data captured using a Dynamic Vision Sensor (DVS)
    and is designed for action recognition tasks. Each sample consists of a sequence of
    events (t, x, y, p) and a corresponding label.

    Args:
        data_path (str): Path to the dataset directory.
        train (bool, optional): If True, creates dataset from training data.
        transform (callable, optional): A function/transform applied to an event sample.
        target_transform (callable, optional): A function/transform applied to an event label.
    """

    def __init__(self,
                 data_path: str = '/data/storage/anastasia/data/DailyDvs-200',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):

        self.train = train
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform

        # Dataset characteristics
        self.resolution = (240, 320)  # Resolution of the data (height, width) in pixels
        self.max_t = 20               # Maximum timestamp in seconds

        # Download the action_idx-action_name mapping
        self.action_desc = pd.read_csv(self.data_path + '/action_description.csv')

        # Get sorted list of classes from directory structure
        self.class_folders = sorted([d for d in os.listdir(self.data_path)
                                     if os.path.isdir(os.path.join(self.data_path, d))])
        self.classes = self.action_desc['Action'].to_numpy()

        # Load all valid event files and corresponding labels
        self.files, self.labels = self._load_data()

        # Validate dataset consistency
        if len(self.files) != len(self.labels):
            raise ValueError(f'Mismatch between files ({len(self.files)}) and labels ({len(self.labels)})')

    def _correct_folder_path(self,
                             data_path: str) -> str:
        """
        Correctly formats folder path with zero-padded numbers.

        Args:
            data_path (str): Path to a file (e.g., 'action_1/C0P10M0S2_20231118_09_44_16.aedat4')

        Returns:
            str: Formatted folder path (e.g., 'action_001/C0P10M0S2_20231118_09_44_16.aedat4')
        """

        # Extract the number part
        try:
            base_name, file_name = os.path.split(data_path)
            number = int(base_name.split('_')[-1])
        except (IndexError, ValueError):
            raise ValueError(f"Folder name doesn't contain number in expected format: {data_path}")

        # Zero-pad the number to 3 digits
        padded_number = f"{number:03d}"

        # Reconstruct the folder path
        parts = base_name.split('_')[:-1]  # Get all parts except the number
        new_base = '_'.join(parts + [padded_number])

        return os.path.join(self.data_path, new_base, file_name)

    def _load_data(self) -> Tuple[List[str], List[int]]:
        """
        Load event file paths and labels for the dataset.

        This method constructs lists of file paths and corresponding labels for all samples
        in the dataset.

        Returns:
            tuple: A tuple containing:
                - files (list): List of file paths for labeled samples.
                - labels (list): List of labels corresponding to `files`.
        """

        if self.train:
            train_split = pd.read_csv(self.data_path + '/train.txt', sep=' ', header=None)
            files = train_split.iloc[:, 0].apply(lambda x: self._correct_folder_path(x)).to_numpy()
            labels = train_split.iloc[:, 1].to_numpy(dtype=int)
        else:
            test_split = pd.read_csv(self.data_path + '/test.txt', sep=' ', header=None)
            files = test_split.iloc[:, 0].apply(lambda x: self._correct_folder_path(x)).to_numpy()
            labels = test_split.iloc[:, 1].to_numpy(dtype=int)

        return files, labels

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_events(event_path: str) -> torch.Tensor:
        """
        Load events from an .aedat4 file.

        Args:
            event_path (str): Path to the event file.

        Returns:
            torch.Tensor: The event data as a tensor of shape [N, 4], where each row
                         contains (t, x, y, p) values.

        Raises:
            FileNotFoundError: If the specified event file doesn't exist
            ValueError: If the loaded data has incorrect dimensions
        """

        try:
            # First pass: count total events to pre-allocate array
            event_count = 0
            decoder = aedat.Decoder(event_path)
            for packet in decoder:
                if 'events' in packet:
                    event_count += len(packet['events'])

            if event_count == 0:
                return torch.empty((0, 4), dtype=torch.int64)

            # Pre-allocate array
            events_np = np.empty((event_count, 4), dtype=np.int64)

            # Second pass: fill the array
            decoder = aedat.Decoder(event_path)  # Reinitialize decoder
            idx = 0
            for packet in decoder:
                if 'events' in packet:
                    chunk = packet['events']
                    chunk_len = len(chunk)
                    events_np[idx:idx + chunk_len, 0] = chunk['t']  # timestamp
                    events_np[idx:idx + chunk_len, 1] = chunk['x']  # x
                    events_np[idx:idx + chunk_len, 2] = chunk['y']  # y
                    events_np[idx:idx + chunk_len, 3] = chunk['p']  # polarity
                    idx += chunk_len

            # Convert to tensor
            events_np[:, 0] = events_np[:, 0] - events_np[0, 0]  # Unix timestamp to microseconds
            events = torch.from_numpy(events_np).to(torch.float32)

            # Validate input shape
            if events.ndim != 2 or events.shape[1] != 4:
                raise ValueError(f"Expected events with shape [N, 4], got {events.shape}")

            return events

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Event file not found at path: {event_path}") from e

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieve a data point from the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing:
                - events(torch.Tensor): The event data of shape [N, 4] containing (t, x, y, p) values with (0, 1) polarity.
                - label (int): The label of the data point as an integer between 0 and num_classes - 1.
        """

        file_path = self.files[idx]
        label = self.labels[idx]

        # Load the events from the file path
        events = self._load_events(event_path=file_path)

        # Apply transforms
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return events, label

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: The number of classes in the dataset.
        """

        return len(self.classes)


class NInternVid(Dataset):
    """
    Dataset class for the N-InternVid event-language dataset.

    This dataset contains event-based data paired with textual captions, designed for
    event-based vision-language tasks. The event streams are generated using the v2e simulator
    (https://arxiv.org/abs/2006.07722) from the InternVid-10M dataset, with the following
    characteristics:

    - Source videos: Originally scraped from YouTube (max duration: 10 seconds)
    - Event format: Each event contains (t, x, y, p) values
    - Data pairing: Each event stream is associated with a descriptive text caption
    - Resolution: Native resolution of 180×240 pixels
    - Temporal span: Maximum 10 second duration per sample

    Args:
       data_path (str): Path to the dataset directory.
       train (bool, optional): If True, creates dataset from training data.
       transform (callable, optional): A function/transform applied to an event sample.
       target_transform (callable, optional): A function/transform applied to an event label.
       seed (int, optional): Seed for the random number generator. Defaults to 42.
    """

    def __init__(self,
                 data_path: str = '/data/storage/anastasia/data/N-InternVid',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 seed: int = 42):

        self.train = train
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform

        # Dataset characteristics
        self.resolution = (180, 240)  # Resolution of the data (height, width) in pixels
        self.max_t = 10               # Maximum timestamp in seconds

        # Download the video_id-caption mapping
        self.video_desc = pd.read_csv(self.data_path + '/dataset.csv', sep='\t', header=0)

        # Load all valid event files and corresponding labels
        self.files, self.captions = self._load_data()

        # Split the data into train and test
        if self.train:
            self.files, _, self.captions, _ = train_test_split(self.files, self.captions,
                                                               test_size=0.1, shuffle=True, random_state=seed)
        else:
            _, self.files, _, self.captions = train_test_split(self.files, self.captions,
                                                               test_size=0.1, shuffle=True, random_state=seed)

        # Validate dataset consistency
        if len(self.files) != len(self.captions):
            raise ValueError(f'Mismatch between files ({len(self.files)}) and captions ({len(self.captions)})')

    def _load_data(self) -> Tuple[List[str], List[str]]:
        """
        Load event file paths and their corresponding captions.

        This method constructs lists of file paths and their captions for all samples
        in the dataset.

        Returns:
            tuple: A tuple containing:
                - files (list): List of file paths for labeled samples.
                - captions (list): List of captions corresponding to `files`.
        """

        files = self.video_desc['YoutubeID'].apply(lambda x: self.data_path + f'/{x}.h5').tolist()
        captions = self.video_desc['Caption'].tolist()

        return files, captions

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_events(event_path: str) -> torch.Tensor:
        """
        Load events from an .h5 file.

        The method performs the following operations:
        1. Loads event data from .h5 file
        2. Converts to PyTorch tensor with float32 precision

        Args:
            event_path (str): Path to the event file.

        Returns:
            torch.Tensor: The event data as a tensor of shape [N, 4], where each row
                         contains (t, x, y, p) values.

        Raises:
            FileNotFoundError: If the specified event file doesn't exist
            ValueError: If the loaded data has incorrect dimensions
        """

        try:
            # Load events from .h5 file
            with h5py.File(event_path, 'r') as f:
                events = f['events'][()]

            # Validate input shape
            if events.ndim != 2 or events.shape[1] != 4:
                raise ValueError(f"Expected events with shape [N, 4], got {events.shape}")

            # Convert polarity values (-1 -> 0, 1 -> 1)
            if events[:, 3].min() < -0.5:
                events[:, 3] = (events[:, 3] > 0).astype(events.dtype)

            # Convert to PyTorch tensor
            return torch.as_tensor(events, dtype=torch.float32)

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Event file not found at path: {event_path}") from e

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieve a data point from the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing:
                - events(torch.Tensor): The event data of shape [N, 4] containing (t, x, y, p) values with (0, 1) polarity.
                - caption (str): The caption corresponding to the data point as a string.
        """

        file_path = self.files[idx]
        caption = self.captions[idx]

        # Load the events from the file path
        events = self._load_events(event_path=file_path)

        # Apply transforms
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            caption = self.target_transform(caption)

        return events, caption


class EventCLRDataset(Dataset):
    """
    A dataset class for contrastive learning with event-based data (SimCLR framework).
    Combines multiple event datasets and applies transformations from the same distribution
    to create positive pairs.

    Args:
        datasets (List[Dataset]): List of event datasets to combine (must all be in same mode)
        train (bool): Whether the dataset is for training (must match all input datasets)
        transform (Callable): Transformations to apply for contrastive learning

    Note:
        All input datasets must return (events, label) tuples in their __getitem__
    """

    def __init__(self,
                 datasets: List[Dataset],
                 train: bool = True,
                 transform: Optional[Callable] = None):

        self.train = train
        self.datasets = datasets
        self.transform = transform

        # Validate dataset modes
        self._validate_datasets()

        # Precompute dataset offsets for efficient indexing
        self._compute_offsets()

        # Dataset characteristics
        self.resolution = self._get_max_resolution()
        self.max_t = max([d.max_t for d in self.datasets])

    def _get_max_resolution(self) -> Tuple[int, int]:
        """
        Get maximum resolution across all datasets

        Returns:
            tuple: A tuple containing maximum height and maximum width across all
                  dataset resolutions.
        """

        max_h = max(d.resolution[0] for d in self.datasets)
        max_w = max(d.resolution[1] for d in self.datasets)
        return max_h, max_w

    def _validate_datasets(self):
        """
        Ensure all datasets are compatible and in the correct mode.

        Raises:
            ValueError: If no datasets are provided or the modes are incompatible.
        """

        if not self.datasets:
            raise ValueError("At least one dataset must be provided.")

        for i, dataset in enumerate(self.datasets):
            if dataset.train != self.train:
                mode = "training" if self.train else "validation/test"
                raise ValueError(f"Dataset {i} is in {'training' if dataset.train else 'validation'} mode, "
                                 f"but EventCLRDataset is configured for {mode}")

    def _compute_offsets(self):
        """
        Precompute dataset boundaries as [0, len(d1), len(d1)+len(d2), ...]
        """

        self.offsets = np.cumsum([0] + [len(d) for d in self.datasets])

    def __len__(self) -> int:
        """
        Total number of samples across all datasets.
        """

        return sum([len(d) for d in self.datasets])

    def _get_dataset_index(self,
                           idx: int) -> Tuple[int, int]:
        """
        Determine which dataset contains the given index and the relative index within that dataset.

        Args:
            idx: Absolute index across all datasets

        Returns:
            Tuple of (dataset_index, relative_index)

        Raises:
            IndexError: If the index is out of bounds.
        """

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")

        dataset_idx = np.searchsorted(self.offsets, idx, side='right') - 1

        if dataset_idx == 0:
            relative_idx = idx
        else:
            relative_idx = idx - self.offsets[dataset_idx]

        return dataset_idx, relative_idx

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample and create two transformed views for contrastive learning.

        Args:
            idx: A global index of the sample to retrieve

        Returns:
            Tuple containing:
            - view1: First transformed view of the events
            - view2: Second transformed view of the events
        """

        dataset_idx, relative_idx = self._get_dataset_index(idx)
        events, _ = self.datasets[dataset_idx][relative_idx]

        # Generate two views
        if self.transform is not None:
            view1 = self.transform(events)
            view2 = self.transform(events)
        else:
            view1 = view2 = events

        return view1, view2

    @property
    def classes(self) -> List[str]:
        """
        Combined class names from all datasets.

        Returns:
            list: A list of all class names.
        """

        return [cls for d in self.datasets for cls in d.classes]

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: The number of all classes from all datasets.
        """

        return len(self.classes)


class EventCLIPDataset(Dataset):
    def __init__(self,
                 datasets: List[Dataset],
                 train: bool = True,
                 transform: Optional[Callable] = None):
        """
        A dataset class for contrastive learning with event-text data (CLIP-style training).
        This dataset combines multiple event datasets into a single unified dataset, where each event sample
        is paired with a text prompt describing its class.

        Args:
            datasets (List[Dataset]): List of event datasets to combine
            train (bool): Whether the dataset is for training (default: True)
            transform (Optional[Callable]): Optional transform to be applied to event data

        Note:
            All input datasets must return (events, label) tuples in their __getitem__
        """

        self.train = train
        self.datasets = datasets
        self.transform = transform

        # Validate dataset modes
        self._validate_datasets()

        # Precompute dataset offsets for efficient indexing
        self._compute_offsets()

        # Dataset characteristics
        self.resolution = self._get_max_resolution()
        self.max_t = max([d.max_t for d in self.datasets])

        # Build unified class mapping by merging duplicates
        self._build_unified_classes()

    def _get_max_resolution(self) -> Tuple[int, int]:
        """
        Get maximum resolution across all datasets

        Returns:
            tuple: A tuple containing maximum height and maximum width across all
                  dataset resolutions.
        """

        max_h = max(d.resolution[0] for d in self.datasets)
        max_w = max(d.resolution[1] for d in self.datasets)
        return max_h, max_w

    def _validate_datasets(self):
        """
        Ensure all datasets are compatible and in the correct mode.

        Raises:
            ValueError: If no datasets are provided or the modes are incompatible.
        """

        if not self.datasets:
            raise ValueError("At least one dataset must be provided.")

        for i, dataset in enumerate(self.datasets):
            if dataset.train != self.train:
                mode = "training" if self.train else "validation/test"
                raise ValueError(f"Dataset {i} is in {'training' if dataset.train else 'validation'} mode, "
                                 f"but EventCLIPDataset is configured for {mode}")

    def _compute_offsets(self):
        """
        Precompute dataset boundaries as [0, len(d1), len(d1)+len(d2), ...]
        """

        self.offsets = np.cumsum([0] + [len(d) for d in self.datasets])

    def _build_unified_classes(self):
        """
        Create unified class names by merging duplicates across datasets.
        """

        self.class_mapping = {}    # {dataset_idx: {local_idx: global_idx}}
        self.unified_classes = []  # List of unique class names
        self.class_to_idx = {}     # {class_name: global_idx}

        current_idx = 0
        for dataset_idx, dataset in enumerate(self.datasets):
            self.class_mapping[dataset_idx] = {}
            for local_idx, class_name in enumerate(dataset.classes):

                # Normalize class name (lowercase, remove underscores)
                normalized_name = class_name.lower().replace('_', ' ')

                # Special handling for DailyDVS dataset
                if isinstance(dataset, DailyDVS200) and not normalized_name.startswith('action '):
                    normalized_name = f'action {normalized_name}'

                # Add to unified classes if not already present
                if normalized_name not in self.class_to_idx:
                    self.class_to_idx[normalized_name] = current_idx
                    self.unified_classes.append(normalized_name)
                    current_idx += 1

                # Map original class to unified index
                self.class_mapping[dataset_idx][local_idx] = self.class_to_idx[normalized_name]

    def __len__(self) -> int:
        """
        Total number of samples across all datasets.
        """

        return sum([len(d) for d in self.datasets])

    def _get_dataset_index(self,
                           idx: int) -> Tuple[int, int]:
        """
        Determine which dataset contains the given index and the relative index within that dataset.

        Args:
            idx: Absolute index across all datasets

        Returns:
            Tuple of (dataset_index, relative_index)

        Raises:
            IndexError: If the index is out of bounds.
        """

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")

        dataset_idx = np.searchsorted(self.offsets, idx, side='right') - 1

        if dataset_idx == 0:
            relative_idx = idx
        else:
            relative_idx = idx - self.offsets[dataset_idx]

        return dataset_idx, relative_idx

    def __getitem__(self,
                    idx: int) -> Tuple[torch.Tensor, str]:
        """
        Retrieves and processes an event-text pair from the dataset.

        Args:
            idx (int): A global index of the sample to retrieve

        Returns:
            Tuple[torch.Tensor, str]: A tuple containing:
                - events(torch.Tensor): The event data of shape [N, 4] containing (t, x, y, p) values with (0, 1) polarity.
                                        Transformed if the transform is provided.
                - global_label (int): The label of the data point as an integer between 0 and num_classes - 1.
                                      num_classes here is the number of unique classes in the unified dataset.
        """

        dataset_idx, relative_idx = self._get_dataset_index(idx)
        events, local_label = self.datasets[dataset_idx][relative_idx]

        # Apply transforms
        if self.transform is not None:
            events = self.transform(events)

        # Get a global label
        global_label = self.class_mapping[dataset_idx][local_label]

        return events, global_label

    @property
    def classes(self) -> List[str]:
        """
        Combined class names from all datasets.

        Returns:
            list: A list of all class names.
        """

        return self.unified_classes

    @property
    def num_classes(self) -> int:
        """
        Returns:
            int: The number of all classes from all datasets.
        """

        return len(self.classes)


def build_dataset(data_path: str,
                  train_transform: Optional[Callable] = None,
                  val_transform: Optional[Callable] = None,
                  target_transform: Optional[Callable] = None) -> Tuple[Dataset, Dataset]:
    """
    Loads the corresponding dataset and splits it into train/val sets.

    Args:
        data_path (str): Directory to load the dataset from.
                         Must contain the name of the dataset as the folder name.
        train_transform (callable, optional): A function/transform applied to a training event sample.
        val_transform (callable, optional): A function/transform applied to a validation event sample.
        target_transform (callable, optional): A function/transform applied to an event label.

    Returns:
        tuple: (train_dataset, val_dataset) - train/val split of the corresponding dataset.

    Raises:
        ValueError: If dataset is not supported or paths are invalid.
    """

    # Dataset name and class mapping
    dataset_map = {'N-Caltech101': NCaltech101,
                   'N-Cars': NCars,
                   'N-ImageNet': NImageNetMini,
                   'CIFAR10DVS': CIFAR10DVS,
                   'DailyDvs-200': DailyDVS200,
                   'N-InternVid': NInternVid}

    # Retrieve the dataset name
    if not 'CIFAR10DVS' in data_path:
        dataset_name = os.path.basename(data_path)
    else:
        dataset_name = 'CIFAR10DVS'

    # Check if the provided dataset is supported
    try:
        dataset_class = dataset_map[dataset_name]
    except KeyError:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                         f"Available: {list(dataset_map.keys())}")

    # Create training dataset
    train_dataset = dataset_class(data_path=data_path,
                                  train=True,
                                  transform=train_transform,
                                  target_transform=target_transform)

    # Create validation dataset
    val_dataset = dataset_class(data_path=data_path,
                                train=False,
                                transform=val_transform,
                                target_transform=target_transform)

    return train_dataset, val_dataset


def build_contrastive_dataset(config: dict,
                              transform: Optional[Callable] = None) -> Tuple[EventCLRDataset, EventCLRDataset]:
    """
    Builds train and validation datasets for event-based contrastive learning.

    Args:
        config: Configuration dictionary containing:
            - data_path: List of paths to event datasets to combine
        transform: Transformations to apply for creating contrastive views

    Returns:
        tuple: Tuple of (train_dataset, val_dataset) for contrastive learning

    Raises:
        ValueError: If dataset is not supported or paths are invalid.
    """

    # Retrieve dataset configuration
    data_config = config['data']

    # Dataset name and class mapping
    dataset_map = {'N-Caltech101': NCaltech101,
                   'N-Cars': NCars,
                   'N-ImageNet': NImageNetMini,
                   'CIFAR10DVS': CIFAR10DVS,
                   'DailyDvs-200': DailyDVS200}

    # Create corresponding dataset lists
    train_datasets = []
    val_datasets = []
    for data_path in data_config['data_path']:
        if not 'CIFAR10DVS' in data_path:
            dataset_name = os.path.basename(data_path)
        else:
            dataset_name = 'CIFAR10DVS'

        # Check if the provided dataset is supported
        try:
            dataset_class = dataset_map[dataset_name]
        except KeyError:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                             f"Available: {list(dataset_map.keys())}")

        train_datasets.append(dataset_class(data_path=data_path,   # Apply transformations in EventCLR dataset
                                            train=True))           # and not in the original event dataset
        val_datasets.append(dataset_class(data_path=data_path,
                                          train=False))

    # Create training EventCLR dataset
    train_dataset = EventCLRDataset(datasets=train_datasets,
                                    train=True,
                                    transform=transform)

    # Create validation EventCLR dataset
    val_dataset = EventCLRDataset(datasets=val_datasets,
                                  train=False,          # Apply transformation to the validation dataset
                                  transform=transform)  # to measure the model performance on SimCLR objective

    return train_dataset, val_dataset


def build_clip_dataset(config: dict,
                       transform: Optional[Callable] = None) -> Tuple[EventCLIPDataset, EventCLIPDataset]:
    """
    Builds train and validation datasets for event-text contrastive learning.

    Args:
        config: Configuration dictionary containing:
            - data_path: List of paths to event datasets to combine
        transform: Transformations to apply for creating contrastive views

    Returns:
        tuple: Tuple of (train_dataset, val_dataset) for CLIP-style contrastive learning
               and classification

    Raises:
        ValueError: If dataset is not supported or paths are invalid.
    """

    # Retrieve dataset configuration
    data_config = config['data']

    # Dataset name and class mapping
    dataset_map = {'N-Caltech101': NCaltech101,
                   'N-Cars': NCars,
                   'N-ImageNet': NImageNetMini,
                   'CIFAR10DVS': CIFAR10DVS,
                   'DailyDvs-200': DailyDVS200}

    # Create corresponding dataset lists
    train_datasets = []
    val_datasets = []
    for data_path in data_config['data_path']:
        if not 'CIFAR10DVS' in data_path:
            dataset_name = os.path.basename(data_path)
        else:
            dataset_name = 'CIFAR10DVS'

        # Check if the provided dataset is supported
        try:
            dataset_class = dataset_map[dataset_name]
        except KeyError:
            raise ValueError(f"Unsupported dataset: {dataset_name}. "
                             f"Available: {list(dataset_map.keys())}")

        train_datasets.append(dataset_class(data_path=data_path,  # Apply transformations to the training dataset
                                            train=True))
        val_datasets.append(dataset_class(data_path=data_path,
                                          train=False))

    # Create training EventCLIP dataset
    train_dataset = EventCLIPDataset(datasets=train_datasets,
                                     train=True,
                                     transform=transform)

    # Create validation EventCLIP dataset
    val_dataset = EventCLIPDataset(datasets=val_datasets,       # Do not apply transformations to the validation dataset
                                   train=False,
                                   transform=None)

    return train_dataset, val_dataset
