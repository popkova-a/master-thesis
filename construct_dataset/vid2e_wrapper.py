import os
import shutil
import subprocess
from tqdm import tqdm
from typing import Tuple, List
from multiprocessing import Pool, cpu_count


class VideoToEvents:
    def __init__(self,
                 video_data_path: str,
                 resolution: Tuple[int, int] = (346, 260),  # DAVIS346
                 verbose: bool = True,
                 upsampler_path: str = '/data/storage/anastasia/repos/rpg_vid2e/upsampling',
                 simulator_path: str = '/data/storage/anastasia/repos/rpg_vid2e/esim_torch/scripts',
                 contrast_threshold_neg: float = 0.2,
                 contrast_threshold_pos: float = 0.2,
                 gpu: List[int] = [7, 8],
                 workers_per_gpu: int = 1):

        self.video_data_path = video_data_path
        self.resolution = resolution
        self.verbose = verbose
        self.upsampler_path = upsampler_path
        self.simulator_path = simulator_path
        self.contrast_threshold_neg = contrast_threshold_neg
        self.contrast_threshold_pos = contrast_threshold_pos
        self.gpu = gpu
        self.workers_per_gpu = workers_per_gpu

        # Define the folders
        video_folder = os.path.basename(video_data_path)
        resized_folder = 'resized_' + video_folder
        upsampled_folder = 'upsampled_' + video_folder
        events_folder = 'events_' + video_folder

        # Create the folders for the resized, upsampled and event versions of the dataset
        root_path = os.path.abspath(self.video_data_path + "/../")
        self.resized_data_path = os.path.join(root_path, resized_folder)
        os.makedirs(self.resized_data_path, exist_ok=True)
        self.upsampled_data_path = os.path.join(root_path, upsampled_folder)
        os.makedirs(self.upsampled_data_path, exist_ok=True)
        self.events_data_path = os.path.join(root_path, events_folder)
        os.makedirs(self.events_data_path, exist_ok=True)

    def _resize_single_video(self,
                             video_path: str) -> None:

        if os.path.abspath(self.video_data_path) != os.path.abspath(video_path + "/../"):
            raise ValueError("Video is not located in the provided video folder!")

        if os.path.exists(video_path):

            # Get the current working directory
            cur_dir = os.getcwd()

            # Setting the video data directory as a working directory
            os.chdir(self.video_data_path)

            # Create a separate directory for the output video
            video_name = os.path.basename(video_path)
            folder_name = video_name.split('.')[0]
            folder_path = os.path.join(self.resized_data_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            resized_path = os.path.join(folder_path, video_name)

            # Execute the code
            cmd = (f'ffmpeg -i {video_name} -vf "scale={self.resolution[0]}:-2, '
                   f'pad={self.resolution[0]}:{self.resolution[1]}:(ow-iw)/2:(oh-ih)/2:black" '
                   f'-an "{resized_path}"')
            output_file = os.path.join(cur_dir, 'output.log')
            with open(output_file, 'a') as f:
                subprocess.run(cmd,
                               stdout=f,
                               stderr=subprocess.STDOUT,
                               shell=True,
                               text=True)

            # Set the working directory back
            os.chdir(cur_dir)

    def _resize(self) -> None:
        # Prepare arguments for each file
        tasks = []
        for file in os.listdir(self.video_data_path):
            tasks.append(os.path.join(self.video_data_path, file))

        # Create pool with optimal number of workers
        num_workers = min(cpu_count(), 8)
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(self._resize_single_video, tasks),
                      total=len(tasks),
                      disable=not self.verbose))

    def _upsample_single_video(self, args) -> None:

        resized_folder_path, gpu_id = args

        if os.path.abspath(self.resized_data_path) != os.path.abspath(resized_folder_path + "/../"):
            raise ValueError("Resized video is not located in the provided folder!")

        if os.path.exists(resized_folder_path):

            # Get the current working directory
            cur_dir = os.getcwd()

            # Setting the upsampler directory as a working directory
            os.chdir(self.upsampler_path)

            # Create a separate directory for the output of the upsampler
            folder_name = os.path.basename(resized_folder_path)
            upsampled_folder_path = os.path.join(self.upsampled_data_path, folder_name)

            # Upsampling code requires non-existence of the output directory.
            # Thus, if the output directory exists, but has no upsampling result
            # or the upsampling terminated before the full sequence was processed,
            # then we remove this directory.
            # However, if the output directory exists and the sequence was already
            # processed, then we skip it.
            if os.path.exists(upsampled_folder_path):
                if 'timestamps.txt' not in os.listdir(upsampled_folder_path):
                    shutil.rmtree(upsampled_folder_path)
                else:
                    return

            # Execute the code
            cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} python upsample.py "
                   f"--input_dir={resized_folder_path} "
                   f"--output_dir={upsampled_folder_path}")
            output_file = os.path.join(cur_dir, 'output.log')
            with open(output_file, 'a') as f:
                subprocess.run(cmd,
                               stdout=f,
                               stderr=subprocess.STDOUT,
                               shell=True,
                               text=True)

            # Set the working directory back
            os.chdir(cur_dir)

    def _upsample(self) -> None:

        print("Using GPUs:", self.gpu)

        # Resize the videos if they are not resized
        if not os.listdir(self.resized_data_path):
            print("Resizing the videos:")
            self._resize()

        # Prepare arguments for each video
        i = 0
        tasks = []
        for folder in os.listdir(self.resized_data_path):
            tasks.append((os.path.join(self.resized_data_path, folder),
                          self.gpu[i % len(self.gpu)]))
            i += 1

        # Create pool with optimal number of workers
        num_workers = min(len(self.gpu) * self.workers_per_gpu, len(tasks))
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(self._upsample_single_video, tasks),
                      total=len(tasks),
                      disable=not self.verbose))

    def _simulate_single_video(self, args) -> None:

        upsampled_folder_path, gpu_id = args

        if os.path.abspath(self.upsampled_data_path) != os.path.abspath(upsampled_folder_path + "/../"):
            raise ValueError("Upsampled video is not located in the provided folder!")

        if os.path.exists(upsampled_folder_path):

            # Get the current working directory
            cur_dir = os.getcwd()

            # Setting the simulator directory as a working directory
            os.chdir(self.simulator_path)

            # Create a separate directory for the output of the simulator
            folder_name = os.path.basename(upsampled_folder_path)
            events_folder_path = os.path.join(self.events_data_path, folder_name)

            # Simulation code requires non-existence of the output directory.
            # Thus, if the output directory exists, but has no simulation result
            # or the simulation terminated before the full sequence was processed,
            # then we remove this directory.
            # However, if the output directory exists and the sequence was already
            # processed, then we skip it.
            if os.path.exists(events_folder_path):
                if len(os.listdir(events_folder_path)) != len(os.listdir(upsampled_folder_path)):
                    shutil.rmtree(events_folder_path)
                else:
                    return

            # Execute the code
            cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} python generate_events.py "
                   f"--input_dir={upsampled_folder_path} "
                   f"--output_dir={events_folder_path} "
                   f"--contrast_threshold_neg={self.contrast_threshold_neg} "
                   f"--contrast_threshold_pos={self.contrast_threshold_pos} "
                   f"--refractory_period_ns=0")
            output_file = os.path.join(cur_dir, 'output.log')
            with open(output_file, 'a') as f:
                subprocess.run(cmd,
                               stdout=f,
                               stderr=subprocess.STDOUT,
                               shell=True,
                               text=True)

            # Set the working directory back
            os.chdir(cur_dir)

    def simulate(self) -> None:

        print("Using GPUs:", self.gpu)

        # Upsample the videos if they are not upsampled
        if not os.listdir(self.upsampled_data_path):
            print("Upsampling the videos:")
            self._upsample()

        # Prepare arguments for each video
        i = 0
        tasks = []
        for folder in os.listdir(self.upsampled_data_path):
            tasks.append((os.path.join(self.upsampled_data_path, folder),
                          self.gpu[i % len(self.gpu)]))
            i += 1

        # Create pool with optimal number of workers
        num_workers = min(len(self.gpu) * self.workers_per_gpu, len(tasks))
        with Pool(processes=num_workers) as pool:
            list(tqdm(pool.imap_unordered(self._simulate_single_video, tasks),
                      total=len(tasks),
                      disable=not self.verbose))

if __name__ == '__main__':
    converter = VideoToEvents(video_data_path='/data/storage/anastasia/data/SUTD-TrafficQA/compressed_videos')
    converter.simulate()
