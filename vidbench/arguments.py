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

import argparse
import sys
import os


class FancyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if not arg_line:
            return []

        if "#" in arg_line:
            if arg_line[0] == "#":
                return []

            return arg_line[: arg_line.index("#")].split()

        return arg_line.split()


parser = FancyArgumentParser(fromfile_prefix_chars="@")

# Required parameters
parser.add_argument(
    "--model_type", default="i3d", type=str, required=True,
)
parser.add_argument(
    "--model_trained_on",
    default="kinetics-400",
    type=str,
    required=True,
    help="I3D models are trained either on `kinetics-400` or `kinetics-600`.",
)
parser.add_argument(
    "--kinetics_version",
    default="400",
    type=str,
    required=True,
    help="Version of the Kinetics Dataset to benchmark against.",
)
parser.add_argument(
    "--kinetics_split",
    default="val",
    type=str,
    help="The Kinetics Dataset split to work with: train, test, val",
)
parser.add_argument(
    "--num_videos", default=None, type=int, help="Number of videos to evaluate over.",
)
parser.add_argument(
    "--num_frames",
    default=None,
    type=int,
    help="Number of frames to resample each video to.",
)
parser.add_argument(
    "--results_dir", default=".", type=str, required=False,
)
parser.add_argument(
    "--batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
)
parser.add_argument(
    "--top_n",
    default=5,
    type=int,
    help="The total number of best model predictions to store and compute accuracy over.",
)
parser.add_argument(
    "--verbose_logging",
    action="store_true",
    help="If true, all of the warnings related to data processing will be printed. ",
)
parser.add_argument(
    "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
)
parser.add_argument(
    "--overwrite_output_dir",
    action="store_true",
    help="Overwrite the content of the output directory",
)
parser.add_argument(
    "--overwrite_cache",
    action="store_true",
    help="Overwrite the cached evaluation sets.",
)

args = parser.parse_args()
