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

from typing import List, Optional

import glob


def make_video_table(
    data_dir: str,
    youtube_ids: List[str],
    ground_truth_labels: List[str],
    predictions: Optional[List[str]] = None,
) -> str:
    """
    Make an HTML table where each cell contains a video and metadata.

    Inputs:
        youtube_ids: list of strings of YouTube ids, one for each video to display
                     these videos should be part of the Kinetics dataset
        ground_truth_labels:  list of strings of ground truth labels, one for each video
        predictions: [optional] list of strings of model predictions, one for each video

    Outputs:
        video_html: a list of HTML tags that build a table; to be called with IPython.display.HTML

            Example:
                from IPython.display import HTML
                HTML(make_video_table(YOUTUBE_IDS, TRUE_LABELS_STR))
    """
    VIDEOS_PER_ROW = 4
    NO_ROWS = len(youtube_ids) // VIDEOS_PER_ROW + 1
    WIDTH = 210
    HEIGHT = WIDTH * 2 // 3

    # for videos to properly display, data directory must be relative to notebook dir
    try:
        data_dir = data_dir[data_dir.find("data"):]
    except:
        pass

    filepaths = []
    for youtube_id in youtube_ids:
        filepaths.append(glob.glob(f"{data_dir}/*/{youtube_id}_*.mp4")[0])

    # make html video table
    video_html = ["<table><tr>"]
    i = 0
    while i < len(filepaths):
        prediction_par = ""
        if predictions is not None:
            color = "black" if predictions[i] == ground_truth_labels[i] else "red"
            prediction_par = f"<p style='color:{color};'>{predictions[i]}</p>"

        video_html.append(
            f"""
            <td><h2>{i}</h2><p>{ground_truth_labels[i]}</p><video width="{WIDTH}" height="{HEIGHT}" controls> 
                <source src="{filepaths[i]}" type="video/mp4">
            </video>{prediction_par}</td>"""
        )

        i += 1
        if i % VIDEOS_PER_ROW == 0:
            video_html.append("</tr><tr>")
    video_html.append("</tr></table>")

    return "".join(video_html)
