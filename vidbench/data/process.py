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

import cv2
import imageio
import numpy as np
import pathlib
from tensorflow_docs.vis import embed


# Adapted from https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def crop_center_square(frame):
    """Crops a square from the center of a rectangular array."""
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def pad_to_square(frame):
    """Pads a rectangular array with zeros, so as to make it squared."""
    y, x = frame.shape[0:2]
    if y > x:
        add_x_left = (y - x) // 2
        add_x_right = y - x - add_x_left
        frame = cv2.copyMakeBorder(
            frame, 0, 0, add_x_left, add_x_right, cv2.BORDER_CONSTANT, value=0
        )
    else:
        add_y_up = (x - y) // 2
        add_y_down = x - y - add_y_up
        frame = cv2.copyMakeBorder(
            frame, add_y_down, add_y_up, 0, 0, cv2.BORDER_CONSTANT, value=0
        )

    return frame


# Adapted from https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def load_and_resize_video(path, resize=(224, 224), resize_type="crop"):
    """Convert video to Numpy array of shape and type expected by i3d model.

    The function resizes them to shape
    [max_frames, 224, 224, 3], in RGB format, with floating point values in
    range [0, 1], as expected by i3d.
    """

    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()  # frame is in BGR format
            if not ret:
                break

            if resize_type == "crop":
                frame = crop_center_square(frame)
            elif resize_type == "pad":
                frame = pad_to_square(frame)
            else:
                return ValueError("Invalid resize_type: " + resize_type)

            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # Convert from BGR to RGB
            frames.append(frame)
    finally:
        cap.release()
    return np.array(frames).astype("float32") / 255.0


def resample_video(video: np.array, num_frames: int) -> np.array:
    """ Resample a video to have num_frames number of frames.
    
    Video must have shape (1, current_num_frames, :, :, :)
    
    if num_frames < current_num_frames, video is downsampled by removing frames
    more or less evenly spaced throughout the duration of the video. 
    
    if num_frames > current_num_frames, video is upsampled by duplicating frames
    more or less evenly spaced throughout the duration of the video. 
    """
    current_num_frames = video.shape[1]
    indices = [(current_num_frames * i) // num_frames for i in range(num_frames)]
    return video[:, indices, :, :, :]


def video_acceptable(video_np, min_num_frames_acceptable: int = 128) -> bool:
    """Checks if video has minimum acceptable temporal length"""
    num_frames = video_np.shape[1]
    if num_frames < min_num_frames_acceptable:
        video_path_no_dir = pathlib.Path(video_path).name
        print(f"Skipping video {video_path_no_dir}, too few frames: {num_frames}")
        return False
    return True


# Adapted from https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    """Converts an array of images to gif."""
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave("./animation.gif", converted_images, fps=25)
    return embed.embed_file("./animation.gif")
