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
from typing import List

from urllib import request
from urllib.parse import urljoin


def joinurl(root_url: str, url_parts: List[str]) -> str:
    """Constructs and returns a url from a root_url and a list of possible url_parts"""
    return urljoin(root_url, "/".join(url_parts))


def download_file(url: str, target_dir: str, context) -> None:
    """If file doesn't exist locally, downloads from url and stores in target dir."""
    file_name = url.split("/")[-1]
    target_path = os.path.join(target_dir, file_name)

    if os.path.exists(target_path):
        print(f"No need to fetch, path already exists {target_path}")
        return

    print(f"Fetching {url} => {target_path}")
    data = request.urlopen(url, context=context).read()
    with open(target_path, "wb") as f:
        f.write(data)


def get_kinetics_labels(dataset_name: str) -> List[str]:
    if dataset_name == "kinetics_400":
        # TODO: Store .txt files in repository, to avoid external dependency??
        kinetics_labels_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
    elif dataset_name == "kinetics_600":
        kinetics_labels_url = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map_600.txt"
    else:
        raise ValueError("Dataset not supported: " + dataset_name)

    with request.urlopen(kinetics_labels_url) as obj:
        labels = [line.decode("utf-8").strip() for line in obj.readlines()]

    return labels
