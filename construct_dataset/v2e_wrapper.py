import os
import random
import subprocess
from tqdm import tqdm
from typing import Tuple, List, Union, Set
from multiprocessing import Pool, cpu_count


class VideoToEvents:
    """
    A class to convert videos to event-based data using the v2e simulator.

    This class provides functionality to simulate DVS (Dynamic Vision Sensor) events from
    conventional video files in parallel using multiple GPUs. It includes methods for
    processing videos, checking dataset consistency, and managing the conversion process.

    Args:
        input_folder (str): Path to the folder containing input video files.
        output_folder (str): Path to the folder where event files will be saved.
        resolution (str): DVS resolution. Options: 'dvs128', 'dvs240', 'dvs346',
                         'dvs640', 'dvs1024'. Default: 'dvs240'.
        dvs_exposure_duration (float): DVS exposure duration in seconds. Default: 0.033.
        input_frame_rate (int): Frame rate of input videos. Default: 30.
        input_slowmotion_factor (float): Slow motion factor of input videos. Default: 1.0.
        disable_slomo (bool): Whether to disable slow motion processing. Default: True.
        auto_timestamp_resolution (bool): Whether to auto-determine timestamp resolution. Default: False.
        pos_thres (float): Positive threshold for event generation. Default: 0.2.
        neg_thres (float): Negative threshold for event generation. Default: 0.2.
        sigma_thres (float): Threshold sigma for event generation. Default: 0.03.
        cutoff_hz (float): Cutoff frequency in Hz. Default: 30.
        leak_rate_hz (float): Leak rate in Hz. Default: 0.1.
        shot_noise_rate_hz (float): Shot noise rate in Hz. Default: 5.
        gpu (List[int]): List of GPU IDs to use for processing. Default: [7, 8].
        workers_per_gpu (int): Number of worker processes per GPU. Default: 1.
        simulator_path (str): Path to the v2e simulator directory. Default: '/data/storage/anastasia/repos/v2e'.

    Raises:
        ValueError: If provided paths don't exist or invalid resolution is specified.
    """

    def __init__(self,
                 input_folder: str,
                 output_folder: str,
                 resolution: str = 'dvs240',  # DAVIS240
                 dvs_exposure_duration: float = 0.033,
                 input_frame_rate: int = 30,
                 input_slowmotion_factor: float = 1.0,
                 disable_slomo: bool = True,
                 auto_timestamp_resolution: bool = False,
                 pos_thres: float = 0.2,
                 neg_thres: float = 0.2,
                 sigma_thres: float = 0.03,
                 cutoff_hz: float = 30,
                 leak_rate_hz: float = 0.1,
                 shot_noise_rate_hz: float = 5,
                 gpu: List[int] = [7, 8],
                 workers_per_gpu: int = 1,
                 simulator_path: str = '/data/storage/anastasia/repos/v2e'):

        self.input_folder = os.path.abspath(input_folder)
        self.output_folder = os.path.abspath(output_folder)
        self.simulator_path = os.path.abspath(simulator_path)

        # Verify the existance of folders
        if not(os.path.exists(self.input_folder) and os.path.exists(self.output_folder)
                and os.path.exists(self.simulator_path)):
            raise ValueError('The provided paths do not exist.')

        # Simulator settings
        self.resolution = resolution
        self.dvs_exposure_duration = dvs_exposure_duration
        self.input_frame_rate = input_frame_rate
        self.input_slowmotion_factor = input_slowmotion_factor
        self.disable_slomo = disable_slomo
        self.auto_timestamp_resolution = auto_timestamp_resolution
        self.pos_thres = pos_thres
        self.neg_thres = neg_thres
        self.sigma_thres = sigma_thres
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.shot_noise_rate_hz = shot_noise_rate_hz

        self.gpu = gpu
        self.workers_per_gpu = workers_per_gpu

        # Initialize counter for processed and corrupted videos
        self.processed = 0
        self.failed = 0

        # Verify sensor
        if resolution not in ['dvs128', 'dvs240', 'dvs346', 'dvs640', 'dvs1024']:
            raise ValueError(f'Invalid resolution: {resolution}')

    def _simulate_single_video(self,
                               args: Tuple[str, int]) -> Tuple[str, bool]:
        """
        Simulates DVS events for a single video file.

        Args:
            args (Tuple[str, int]): A tuple containing (video_filename, gpu_id).

        Returns:
            Tuple[str, bool]: A tuple containing (video_filename, success_status) where
                            success_status is True if simulation succeeded, False otherwise.
        """

        video_name, gpu_id = args
        input_path = os.path.join(self.input_folder, video_name)
        output_h5_name = os.path.splitext(video_name)[0]+'.h5'
        output_video_name = os.path.splitext(video_name)[0] + '.avi'
        cmd = ["torchrun",
               "--nproc_per_node",              "1",
               "--master_port",                 str(random.randint(1024, 65535)),
               "v2e.py",
               "-i",                            input_path,
               "-o",                            self.output_folder,
               "--overwrite",
               "--unique_output_folder",        "false",
               "--davis_output",
               "--dvs_h5",                      output_h5_name.lstrip('-'),
               "--dvs_vid",                     output_video_name.lstrip('-'),
               "--dvs_aedat2",                  "None",
               "--dvs_text",                    "None",
               "--no_preview",
               "--dvs_exposure",                "duration", str(self.dvs_exposure_duration),
               "--input_frame_rate",            str(self.input_frame_rate),
               "--input_slowmotion_factor",     str(self.input_slowmotion_factor),
               *(["--disable_slomo"] if self.disable_slomo else []),
               "--auto_timestamp_resolution",   str(self.auto_timestamp_resolution),
               "--pos_thres",                   str(self.pos_thres),
               "--neg_thres",                   str(self.neg_thres),
               "--sigma_thres",                 str(self.sigma_thres),
               "--cutoff_hz",                   str(self.cutoff_hz),
               "--leak_rate_hz",                str(self.leak_rate_hz),
               "--shot_noise_rate_hz",          str(self.shot_noise_rate_hz),
               f"--{self.resolution}"]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        try:
            with open("v2e_output.txt", "a+") as f:
                subprocess.run(cmd,
                               cwd=self.simulator_path,
                               env=env,
                               check=True,
                               stdout=f,
                               stderr=subprocess.STDOUT,
                               text=True)
            return video_name, True
        except Exception as e:
            print(f'Failed to simulate {video_name}: {e}')
            return video_name, False

    def simulate(self,
                 max_items: int = None) -> Union[List[Tuple[str, bool]], None]:
        """
        Processes all videos in the input folder to generate DVS events.

       Args:
           max_items (int, optional): Maximum number of videos to process. If None,
                                     processes all videos. Default: None.

       Returns:
           Union[List[Tuple[str, bool]], None]: List of tuples with processing results
                                              (filename, success) or None if no videos
                                              needed processing.
       """

        video_names = []
        for video in os.listdir(self.input_folder):

            # Stop if maximum is reached
            if len(video_names) == max_items:
                break

            # Append videos that are in input folder and are not yet simulated in the output folder
            if (video.endswith(('.mp4', '.mov', '.avi'))
                and (video.split('.')[0] + '.h5' not in os.listdir(self.output_folder))):
                video_names.append(video)

        if not video_names:
            print("All videos were already successfully simulated.")
            return None

        tasks = [(name, self.gpu[i % len(self.gpu)])
                 for i, name in enumerate(video_names)]

        # Process in parallel with a progress bar
        num_processes = min(self.workers_per_gpu * len(self.gpu), cpu_count())
        results = []

        with tqdm(total=len(tasks)) as pbar:
            with Pool(processes=num_processes) as pool:
                for video_name, success in pool.imap_unordered(self._simulate_single_video, tasks):
                    results.append((video_name, success))

                    # Update progress bar and counters
                    pbar.update(1)
                    if success:
                        self.processed += 1
                    else:
                        self.failed += 1

                    pbar.set_postfix(processed=self.processed, failed=self.failed)

        event_files = [event_f for event_f in os.listdir(self.output_folder) if event_f.endswith('.h5')]
        print(f"\nFinal Summary - Success: {self.processed}, Failed: {self.failed}")
        print(f"The length of the initial dataset: {len(video_names)}")
        print(f"The length of the resulting dataset: {len(event_files)}")
        return results

    @staticmethod
    def _get_file_names(folder_path: str,
                        extensions: Tuple[str] = ('.mp4', '.mov', '.avi')) -> Set[str]:
        """
        Retrieves filenames with specified extensions from a directory.

        Args:
            folder_path (str): Path to the directory to scan for files.
            extensions (tuple): Tuple of file extensions to include (e.g., '.mp4', '.mov').
                                Defaults to ('.mp4', '.mov', '.avi').

        Returns:
            set: A set of strings containing filenames that match the given extensions.
                Only returns the base filenames, not full paths.
        """
        return set([file.split('.')[0] for file in os.listdir(folder_path) if file.endswith(extensions)])

    def check_dataset(self):
        """
        Compares input videos with generated event files to identify mismatches.

        Returns:
            Tuple[Set[str], Set[str]]: Two sets containing:
                - Files present as videos but not as events
                - Files present as events but not as videos
        """

        videos = self._get_file_names(self.input_folder,
                                      extensions=('.mp4', '.mov', '.avi'))
        events = self._get_file_names(self.output_folder,
                                      extensions=('.h5'))

        # Videos, but not events
        videos_only = videos - events
        print(f"\nFiles as videos but not as events ({len(videos_only)}):")
        for f in sorted(videos_only):
            print(f" - {f}")

        # Events, but not videos
        events_only = events - videos
        print(f"\nFiles as events but not as videos ({len(events_only)}):")
        for f in sorted(events_only):
            print(f" - {f}")

        return videos_only, events_only

if __name__ == '__main__':
    simulator = VideoToEvents(input_folder='/data/storage/anastasia/data/InternVid-10M',
                              output_folder='/data/storage/anastasia/data/N-InternVid',
                              gpu=[0, 6],
                              workers_per_gpu=2)
    simulator.simulate(max_items=10000)
    simulator.check_dataset()
