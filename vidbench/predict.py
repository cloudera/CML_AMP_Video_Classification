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

from collections import defaultdict
import time
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

Label = str


def sort_2d_array_rows(arr, mylabs, descending=True):
    """Orders rows of arr and applies ordering indices to mylabs.

    Useful to find top prediction scores and labels, e.g. in classification

    Example
    Input is a 2d array, where each row has the scores for each class of an example
    and different rows correspond to different examples in a batch.
        arr = [[0.3, 0.1, 0.4],[0.6, 0.9, 0.7]]
        mylabs = ['a', 'b', 'c']
        descending=True
    Output
        [[0.4, 0.3, 0.1], [0.9, 0.7, 0.6]]
        [['c', 'a', 'b'], ['b', 'c', 'a']]

    Args
        arr: Numpy array of floating points, ndim = 2. shape: [batch_size, num_classes]
        mylabs: List of size equal to the second dimension of arr, namely num_classes
        descending: Sorts array in descending order if True, and in increasing order otherwise.
    Returns
        Tuple of arrays, each of which has same shape as arr
    """

    assert arr.ndim == 2
    assert arr.shape[1] == len(mylabs)

    indices = np.argsort(arr, axis=1)
    if descending:
        indices = indices[..., ::-1]

    arr_sorted = np.take_along_axis(arr, indices, axis=1)
    labels_sorted = np.take(mylabs, indices)

    return arr_sorted, labels_sorted


def predict(video_np, model, verbose=False):
    """Predict the class of a video using TensorFlow model

    Args:
        video_np: Batch of videos on which to run prediction. Accepted shapes
            (            num_frames, height, width, num_channels)
            (batch_size, num_frames, height, width, num_channels)
        model: I3DLoader model class

    """

    video_tf = tf.constant(video_np, dtype=tf.float32)

    # Add batch axis
    if video_tf.ndim != 5:
        video_tf = video_tf[tf.newaxis, ...]

    if "i3d" in model.name.lower():
        I3D_MIN_ACCEPTABLE_FRAMES = 100  # TODO check actual value, fails with 4 frames
        num_frames = video_np.shape[1]
        if num_frames < I3D_MIN_ACCEPTABLE_FRAMES:
            raise ValueError(
                f"Too few frames: {num_frames}, required at least {I3D_MIN_ACCEPTABLE_FRAMES}"
            )

    # output is tensor of shape (batch_size, num_classes)
    # TODO process_time gives too high time estimate
    start_time = time.process_time()
    model_output = model.model(video_tf)  # logits or soft maxed probabilities?
    run_time = time.process_time() - start_time
    # TODO CHeck that output of i3d has shape batch_size, num_classes

    # i3d output is dictionary with key 'default', and value tensor of shape (batch_size, num_classes)
    # print('Model name ', model_name)
    # print('DEBUG ', model_output)
    if "i3d" in model.name.lower():
        model_output = model_output["default"]

    # Get probabilities of each video class
    probabilities = tf.nn.softmax(model_output)

    # Sort probabilities and generate a tensor of class labels sorted according to probability
    probabilities_sorted, labels_sorted = sort_2d_array_rows(
        probabilities.numpy(), model.labels
    )

    if verbose:
        # Print top 5 classes for at most the first 3 videos in the batch
        num_videos = min(3, video_tf.shape[0])
        num_top_classes = 5
        for video_count in range(num_videos):
            print()
            print("Top 5 predicted classes")
            for class_count in range(num_top_classes):
                prob = probabilities_sorted[video_count, class_count] * 100
                label = labels_sorted[video_count, class_count]
                print(f"  {label:30}: {prob:5.2f}%")

        print(f"Execution time (sec): {run_time: 10.4f}")

    args_max = np.argmax(probabilities, axis=1)

    return probabilities_sorted, labels_sorted, run_time, args_max


def store_results_in_dataframe(
    results_dict: dict, top_n: int = 5, savefile=None,
) -> pd.DataFrame:
    """Store results of an evaluation in a pandas DataFrame."""
    df = pd.DataFrame(results_dict)
    df = (
        df.join(
            [
                pd.DataFrame(
                    df["preds"].to_list(), columns=[f"pred_{i+1}" for i in range(top_n)]
                ),
                pd.DataFrame(
                    df["scores"].to_list(),
                    columns=[f"score_{i+1}" for i in range(top_n)],
                ),
            ]
        )
        .drop(columns=["scores", "preds"])
        .rename_axis(index="video_id")
    )
    if savefile:
        df.to_csv(savefile)
    return df


def compute_accuracy(results_df: pd.DataFrame, *, num_top_classes: int) -> float:
    """Compute prediction accuracy using up to num_top_classes classes"""
    # TODO: make this function more elegant
    # TODO: currently the column names depend on what they are set to in store_results_in_dataframe
    #       Remove that dependency or pass the column names
    results_df = results_df.copy()

    columns = [f"pred_{str(id+1)}" for id in range(num_top_classes)]

    for column in columns:
        results_df[column + "_correct"] = (
            results_df["Ground_Truth"] == results_df[column]
        )

    columns_correct = [col for col in results_df.columns if "correct" in col]
    correct_df = results_df[columns_correct]

    columns_top = [f"pred_{str(id+1)}_correct" for id in range(num_top_classes)]
    correct_any_df = correct_df[columns_top].any(axis=1)

    try:
        accuracy = correct_any_df.value_counts(normalize=True)[True] * 100
    except KeyError:  # no correct labels at all
        accuracy = 0
    print(f"top-{num_top_classes} accuracy: {accuracy:5.2f}%")

    return accuracy


def evaluate(model, dataset, num_videos, batch_size, top_n_results=5, **kwargs):
    """ Evaluate a model over a specific dataset."""
    batch_kwargs = dict()
    try:
        batch_kwargs["num_frames"] = kwargs.pop("num_frames")
    except:
        pass
    results = defaultdict(list)
    for batch in dataset.get_batches(
        num_videos=num_videos, batch_size=batch_size, **batch_kwargs
    ):
        video_batch, labels, youtube_ids = batch
        scores, predictions, _, _ = predict(video_batch, model)

        # collect metadata and model results
        for i in range(batch_size):
            results["YouTube_Id"].append(youtube_ids[i])
            results["Ground_Truth"].append(labels[i])

        for s, p in zip(scores, predictions):
            results["scores"].append(list(s[:top_n_results]))
            results["preds"].append(list(p[:top_n_results]))

    return store_results_in_dataframe(results, top_n_results, **kwargs)
