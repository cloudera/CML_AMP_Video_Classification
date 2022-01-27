# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

from dataclasses import dataclass
import os
from typing import List

import glob
import numpy as np
import pandas as pd
import pathlib
import pickle
import shutil
import ssl

from vidbench.data.process import (
    load_and_resize_video,
    resample_video,
    video_acceptable,
)
from vidbench.data.fetch import get_kinetics_labels, download_file, joinurl


DATA_DIR = os.path.abspath("data")
ROOT_URL = "https://s3.amazonaws.com/kinetics/"

Path = str
Label = str


class KineticsLoader(object):
    """
    This class handles all functionality for downloading, unpacking, processing,
    and loading Kinetics videos for model consumption. Currently, this class
    only fully supports the Kinetics400 dataset but can be expanded to support
    Kinetics600 and Kinetics700.
    
    As of November 2021, the video files that make up the Kinetics datasets are stored 
    at [1] (as described at [2]). There are multiple versions of the Kinetics dataset 
    (400, 600, 700, etc.). Each version has three splits: training, test and validation. 
    Each split consists of thousands of videos in .mp4 format (the exact number depends 
    on the split). These videos are grouped into a series of directories and each 
    directory is packaged as a tar.gz file. These tar.gz files live at [1]. This class 
    allows the user to download and unpack from [1] either all or a limited number of 
    videos, via the download_n_videos method.

    [1] https://s3.amazonaws.com/kinetics/
    [2] https://github.com/cvdfoundation/kinetics-dataset
    
    A KineticsLoader object should be specified for each version-split the user wishes 
    to work with.

    Attributes
    ----------
    version:  the version of the Kinetics dataset to manage
    split:    the data split (train, test, val) to manage
    """

    def __init__(self, version: str = "400", split: str = "val") -> None:
        supported_versions = ("400", "600", "700")
        if version not in supported_versions:
            raise ValueError(
                f"Kinetics dataset version {version} not supported."
                f"Supported: {', '.join(supported_versions)}"
            )
        supported_splits = ("train", "test", "val")
        if split not in supported_splits:
            raise ValueError(
                f"Kinetics dataset split {split} not supported."
                f"Supported: {', '.join(supported_splits)}"
            )

        # Initialize some variables
        self.dataset_name = "kinetics"
        self.kinetics_version = version
        self.data_split = split
        self.context = ssl._create_unverified_context()  # Needed for downloads
        self.data_dir = os.path.join(
            DATA_DIR, "raw", self.dataset_name, self.kinetics_version, self.data_split,
        )
        self.processed_data_dir = os.path.join(
            DATA_DIR,
            "processed",
            self.dataset_name,
            self.kinetics_version,
            self.data_split,
        )
        self.targz_file_tracker = 0  # Points to next partition of videos to download

        # Create parent directory if it doesn't exist
        pathlib.Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.processed_data_dir).mkdir(parents=True, exist_ok=True)

        # Download text file with list of locations of zipped video files, e.g.,
        # https://s3.amazonaws.com/kinetics/400/train/k400_train_path.txt
        # (videos are stored in partitions)
        self.video_partitions_list = self._download_partition_files()
        self.video_filenames = glob.glob(os.path.join(self.data_dir, "*", "*.mp4"))
        self.video_filenames_processed = set(
            glob.glob(os.path.join(self.processed_data_dir, "*", "*.pkl"))
        )

        # Get ground truth labels
        self.ground_truth_labels = dict()
        for _, row in pd.read_csv(self._download_annotation_file()).iterrows():
            youtube_id = row["youtube_id"]
            label = row["label"]
            self.ground_truth_labels[youtube_id] = label

    @property
    def num_videos_downloaded(self) -> int:
        return len(self.video_filenames)

    @property
    def num_videos_processed(self) -> int:
        return len(self.video_filenames_processed)

    def _download_partition_files(self) -> List[Path]:
        """Downloads pathname list of video partitions"""
        video_partitions_list_filename = (
            f"k{self.kinetics_version}_{self.data_split}_path.txt"
        )
        url = joinurl(
            ROOT_URL,
            [self.kinetics_version, self.data_split, video_partitions_list_filename],
        )
        download_file(url, self.data_dir, self.context)
        with open(os.path.join(self.data_dir, video_partitions_list_filename)) as file:
            video_partitions_list = [line.rstrip() for line in file.readlines()]
        return video_partitions_list

    def _download_annotation_file(self) -> None:
        """Downloads annoations file"""
        # Download cvs file with labels, e.g.,
        # https://s3.amazonaws.com/kinetics/400/annotations/train.csv
        ext = ".csv"
        if self.kinetics_version == "600" and self.data_split in ("val", "train"):
            # For some reason, the Kinetics 600 annotations for val and train are txt
            #  rather than csv
            ext = ".txt"
        url = joinurl(
            ROOT_URL, [self.kinetics_version, "annotations", self.data_split + ext]
        )
        download_file(url, self.data_dir, self.context)
        return os.path.join(self.data_dir, self.data_split + ext)

    def _download_more_data(self) -> bool:
        """Downloads more video data while managing the targz_file_tracker.

        Returns
          true if more data available
        """
        if self.targz_file_tracker >= len(self.video_partitions_list):
            # Nothing left
            return False

        # Download and unpack the partition that the cursor is on.
        # e.g., k400_train_path.txt
        partition_file = self.video_partitions_list[self.targz_file_tracker]
        download_file(partition_file, self.data_dir, self.context)
        self.targz_file_tracker += 1

        # This complains for KINETICS_600_TEST_DATA_DIR, but I think it was able to do
        #  KINETICS_400_TEST_DATA_DIR
        unpack_files(self.data_dir)

        # Update video_filenames with newly-loaded files.
        self.video_filenames = glob.glob(os.path.join(self.data_dir, "*", "*.mp4"))

        return True

    def download_n_videos(self, num_videos: int) -> None:
        """Downloads and unpacks videos until at least num_videos have been acquired.

        If num_videos is greater than the total amount of available videos, then the
        function will continue until all videos have been downloaded.
        """
        while self.num_videos_downloaded < num_videos and self._download_more_data():
            pass

    def load_and_cache_video_example(
        self, video_path: str, resize_type: str, overwrite: bool = False,
    ) -> None:
        """ Load video and metadata into VideoExample object and save to disk."""
        processed_filename = video_path.replace("raw", "processed").replace(
            ".mp4", ".pkl"
        )
        if not os.path.exists(processed_filename) or overwrite:
            print(f"Processing {video_path}")
            
            # verify video "chunk" directory exists in processed_data_dir
            processed_dir = os.path.split(processed_filename)[0]
            if not os.path.exists(processed_dir):
                pathlib.Path(processed_dir).mkdir(parents=True, exist_ok=True)
            
            video_array = load_and_resize_video(video_path, resize_type=resize_type)

            if video_acceptable(video_array):
                # Collect video ground truth information and youtube file information
                # YouTube ids are the first 11 characters of the video filename
                youtube_id = os.path.split(video_path)[1][:11]
                video_example = VideoExample(
                    youtube_id=youtube_id,
                    label=self.ground_truth_labels[youtube_id],
                    array=video_array,
                    original_filename=video_path,
                )
                pickle.dump(video_example, open(processed_filename, "wb"))
                # track newly processed video filenames
                self.video_filenames_processed.add(os.path.abspath(processed_filename))
        else:
            print(
                f"Processed video already exists at {processed_filename}.",
                "Use overwrite=True to re-process.",
            )

    def load_and_cache_video_examples(
        self, num_videos: int, resize_type: str = "crop", overwrite: bool = False,
    ) -> None:
        """Loads, processes, and caches raw video files."""
        if self.num_videos_downloaded < num_videos:
            raise Exception(
                f"Fewer than {num_videos} videos available for processing.\n"
                f"Call loader.download_n_videos({num_videos}) to download more videos."
            )

        for video_path in self.video_filenames[:num_videos]:
            self.load_and_cache_video_example(video_path, resize_type, overwrite)

    def get_batches(self, num_videos: int, batch_size: int = 8, num_frames=100):
        """Loads batches until num_videos videos have been returned."""
        if self.num_videos_processed < num_videos:
            raise Exception(
                f"Fewer than num_videos have been processed and cached.\n"
                f"Call loader.load_and_cache_video_examples({num_videos})"
            )

        video_batch, labels, youtube_ids = list(), list(), list()
        for video_filename in list(self.video_filenames_processed)[:num_videos]:
            # load chunks of batch_size video_examples
            video_example = pickle.load(open(video_filename, "rb"))
            video_batch.append(np.expand_dims(video_example.array, axis=0))
            labels.append(video_example.label)
            youtube_ids.append(video_example.youtube_id)

            if len(video_batch) >= batch_size:
                # resample video frames to ensure fixed array dimensions
                video_batch_resampled = [
                    resample_video(video, num_frames=num_frames)
                    for video in video_batch
                ]
                video_batch = np.concatenate(video_batch_resampled, axis=0).astype(
                    "float32"
                )
                yield video_batch, labels, youtube_ids
                video_batch, labels, youtube_ids = list(), list(), list()

        # if there's a fraction of a batch
        if video_batch:
            video_batch_resampled = [
                resample_video(video, num_frames=num_frames) for video in video_batch
            ]
            video_batch = np.concatenate(video_batch_resampled, axis=0).astype(
                "float32"
            )
            yield video_batch, labels, youtube_ids


@dataclass
class VideoExample(object):
    """This class is made to contain video data: array and metadata."""

    youtube_id: str
    label: str
    array: np.array
    original_filename: str


def unpack_files(
    source_dir: str, extension: str = ".tar.gz", force_unpack: bool = False
) -> None:
    """Unpack all files with given extension in a directory."""
    tar_file_names = glob.glob(os.path.join(source_dir, "*" + extension))

    for targz_file_name in tar_file_names:
        # Extract to same directory as the tar.gz file
        extract_dir = targz_file_name[: -len(extension)]

        # Extract only if directory doesn't exist
        if (not pathlib.Path(extract_dir).exists()) or force_unpack:
            print("Extracting: ", extract_dir)
            shutil.unpack_archive(targz_file_name, extract_dir)
        else:
            print("File already unpacked: ", targz_file_name)
