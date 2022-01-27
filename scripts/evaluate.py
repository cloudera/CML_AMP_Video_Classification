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

import os
import pathlib

from vidbench.data.load import KineticsLoader
from vidbench.models import I3DLoader
from vidbench.predict import evaluate, compute_accuracy


def main(args):
    if "i3d" not in args.model_type.lower():
        raise ValueError("EVAL: Evaluation script currently only supports I3D models.")
        # In the future, multiple models will be supported
        # Model selection will take place here.
    
    # load video classification model
    model = I3DLoader(trained_on=args.model_trained_on)

    # load video handler
    loader = KineticsLoader(version=args.kinetics_version, split=args.kinetics_split)
      
    # check that we have requested number of pre-processed videos 
    # if not, load_and_cache to build a processed data set
    if loader.num_videos_processed < args.num_videos: 
        print(f"EVAL: Only {loader.num_videos_processed} videos prepared for inference. Acquiring more.")
        loader.download_n_videos(args.num_videos)
        loader.load_and_cache_video_examples(args.num_videos)
        print(f"EVAL: {loader.num_videos_processed} ready for inference.")
    
    if not os.path.exists(args.results_dir):
        pathlib.Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    results_filename = f"{args.results_dir}/{model.name}_{args.kinetics_split}{args.kinetics_version}.csv"
    
    results_df = evaluate(
        model,
        loader, 
        num_videos=args.num_videos, 
        batch_size=args.batch_size, 
        top_n_results=args.top_n, 
        num_frames=args.num_frames, 
        savefile=results_filename
    )

    # generate report
    accuracy_top_1 = compute_accuracy(results_df, num_top_classes = 1)
    accuracy_top_5 = compute_accuracy(results_df, num_top_classes = args.top_n)

    return 


if __name__ == "__main__":
    from vidbench.arguments import args
    main(args)



