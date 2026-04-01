import os
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
from multiprocessing import Pool
from datasets import load_dataset


class YoutubeScraper:
    """
    A class for scraping and downloading video segments from YouTube using the InternVid dataset.

    This class handles downloading video segments from YouTube based on timestamps provided in the dataset,
    with options for parallel processing, duration limits, and progress tracking.

    Args:
        dataset_path (str): Path to the dataset on Hugging Face Hub. Defaults to "OpenGVLab/InternVid".
        dataset_name (str): Name of the dataset to load. Defaults to "InternVid-10M".
        output_path (str): Directory where downloaded videos will be saved. Defaults to "/data/storage/anastasia/data/InternVid-10M".
        max_duration (float): Maximum allowed duration (in seconds) for video segments. Defaults to 10.
        num_processes (int): Number of parallel processes to use for downloading. Defaults to 4.
    """

    def __init__(self,
                 dataset_path: str = "OpenGVLab/InternVid",
                 dataset_name: str = "InternVid-10M",
                 output_path: str = "/data/storage/anastasia/data/InternVid-10M",
                 max_duration: float = 10,   # Maximum video duration in seconds
                 num_processes: int = 4):

        self.init_dataset = load_dataset(dataset_path, name=dataset_name)['FLT']
        self._res_dataset = {}   # Dict of resulting entries that were scraped (no long or private videos, etc.)
        self.output_path = output_path
        self.num_processes = num_processes
        self.max_duration = max_duration

        # Initialize counter for processed and corrupted videos
        self.processed = 0
        self.failed = 0

    def __len__(self) -> int:
        return len(os.listdir(self.output_path)) - 1  # Accounts for dataset.csv file which is not a video

    @property
    def res_dataset(self) -> list:
        """
        Get the list of successfully processed dataset entries.

        Returns:
            list: A list of dictionary entries for videos that were successfully processed.
        """

        return list(self._res_dataset.values())

    def save_res_dataset(self,
                         file_name: str = 'dataset.csv') -> None:
        """
        Save the resulting dataset to a CSV file.

        Args:
            file_name (str): Name of the output CSV file. Defaults to 'dataset.csv'.
        """

        df = pd.DataFrame(self.res_dataset)
        df.to_csv(os.path.join(self.output_path, file_name), sep='\t')

    @staticmethod
    def _format_timestamp(timestamp: str) -> str:
        """
        Convert a timestamp into HH:MM:SS format.

        Handles various input formats including seconds, MM:SS, and HH:MM:SS.

        Args:
            timestamp (str): Input timestamp in various formats.

        Returns:
            str: Timestamp formatted as HH:MM:SS.
        """

        if not isinstance(timestamp, str):
            timestamp = str(timestamp)

        # Remove .MS in HH:MM:SS.MS
        if '.' in timestamp:
            timestamp = timestamp.split('.')[0]

        # Handle HH:MM:SS or seconds
        if ":" in timestamp:
            # Already in HH:MM:SS or MM:SS format
            parts = timestamp.split(":")
            if len(parts) == 2:
                # MM:SS → pad to HH:MM:SS
                return f"00:{parts[0]}:{parts[1]}"
            elif len(parts) == 3:
                # HH:MM:SS → return as-is
                return timestamp
        else:
            # Raw seconds (e.g., "90.5" → "00:01:30")
            seconds = float(timestamp)
            return f"{int(seconds // 3600):02d}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """
        Convert a timestamp string to seconds.

        Args:
            timestamp (str): Timestamp in HH:MM:SS or HH:MM:SS.MS format.

        Returns:
            float: Total time in seconds.
        """

        hh, mm, ss = timestamp.split(':')
        ss, ms = ss.split('.') if "." in ss else (ss, '0')
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + float(f'0.{ms}')

    def _duration(self,
                  start_timestamp: str,
                  end_timestamp: str) -> float:
        """
        Calculate the duration between two timestamps.

        Args:
            start_timestamp (str): Start timestamp in HH:MM:SS format.
            end_timestamp (str): End timestamp in HH:MM:SS format.

        Returns:
            float: Duration in seconds.
        """

        start_time = self._timestamp_to_seconds(start_timestamp)
        end_time = self._timestamp_to_seconds(end_timestamp)
        return end_time - start_time

    def _download_single_video(self,
                               args: Tuple[str, int]) -> Tuple[str, bool]:
        """
        Download a single video segment from YouTube.

        Args:
            args (Tuple[str, int]): Tuple containing (video_id, start_time, end_time).

        Returns:
            Tuple[str, bool]: Tuple of (video_id, success) where success is a boolean indicating download status.
        """

        video_id, start_time, end_time = args

        # Video url
        url = f"https://www.youtube.com/watch?v={video_id}"

        # yt-dlp command to download only the segment
        cmd = ["yt-dlp",
               "-f",                         "best[height<=360]",  # Video only, max 360p
               "--download-sections",        f"*{start_time}-{end_time}",
               "--external-downloader",      "ffmpeg",
               "--external-downloader-args", "ffmpeg: -an",
               "--force-keyframes-at-cuts",
               "-o",                         f"{self.output_path}/{video_id}.mp4",
               "--limit-rate",               "5M",
               "--sleep-interval",           "5",
               "--max-sleep-interval",       "10",
               "--cookies",                  "cookies.txt",
               "--user-agent",               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
               "--add-header",               "Accept-Language: en-US,en;q=0.9",
               "--no-cache-dir",
               "--no-part",
               "--merge-output-format", "mp4",
               "--quiet",
               url]
        try:
            subprocess.run(cmd, check=True, timeout=600)
            return video_id, True
        except Exception as e:
            print(f'Failed to download {video_id}: {e}')
            return video_id, False

    def scrape(self,
               max_items: int = None) -> List[Tuple[str, bool]]:
        """
        Download video segments from the dataset.

        Args:
            max_items (int, optional): Maximum number of items to process. If None, processes all items. Defaults to None.

        Returns:
            List[Tuple[str, bool]]: List of tuples containing (video_id, success) for each download attempt.
        """

        tasks = []
        for sample in self.init_dataset:
            if max_items and len(tasks) >= max_items:
                break

            video_id = sample['YoutubeID']

            if self._duration(sample['Start_timestamp'],
                              sample['End_timestamp']) > self.max_duration:
                self.failed += 1
                print(f'Did not download {video_id}: the video is too long.')
                continue

            if os.path.exists(f"{self.output_path}/{video_id}.mp4"):
                # Add the entry to the resulting dataset
                self._res_dataset[video_id] = sample
                continue

            start_time = self._format_timestamp(sample['Start_timestamp'])
            end_time = self._format_timestamp(sample['End_timestamp'])
            tasks.append((video_id, start_time, end_time))

            # Add the entry to the resulting dataset
            self._res_dataset[video_id] = sample


        # Process in parallel with progress bar
        results = []
        with tqdm(total=len(tasks)) as pbar:
            with Pool(processes=self.num_processes) as pool:
                for video_id, success in pool.imap_unordered(self._download_single_video, tasks):
                    results.append((video_id, success))

                    # Update progress bar and counters
                    pbar.update(1)
                    if success:
                        self.processed += 1
                    else:
                        self.failed += 1

                        # Remove invalid entry
                        self._res_dataset.pop(video_id)
                    pbar.set_postfix(processed=self.processed, failed=self.failed)

        print(f"\nFinal Summary - Success: {self.processed}, Failed: {self.failed}")
        print(f"The length of the initial dataset: {len(self.init_dataset)}")
        print(f"The length of the resulting dataset: {len(self.res_dataset)}")
        return results

    def create_dataset(self) -> None:
        """
        Create a dataset of successfully downloaded videos by scanning the output directory.
        """

        with tqdm(total=len(self)) as pbar:
            for sample in self.init_dataset:
                video_id = sample['YoutubeID']

                if (os.path.exists(f"{self.output_path}/{video_id}.mp4")
                    and self._res_dataset.get(video_id) is None):

                    # Add the entry to the resulting dataset
                    self._res_dataset[video_id] = sample

                    # Save the altered dataset
                    self.save_res_dataset()

                    # Update progress bar
                    pbar.update(1)

                # Break if the length of the dataset matches the number of files in the output folder
                if len(self._res_dataset) == len(self):
                    break


if __name__ == '__main__':

    scraper = YoutubeScraper(num_processes=1)
    scraper.scrape(max_items=2000)
    scraper.save_res_dataset()
