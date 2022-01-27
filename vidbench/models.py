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

import tensorflow as tf
import tensorflow_hub as hub

from vidbench.data.fetch import get_kinetics_labels


class I3DLoader(object):
    """
    This class handles functionality for downloading and loading an I3d model
    from the TF Model Hub. It supports I3D models trained on Kinetics 400 and
    Kinetics 600 datasets and tracks the label names appropriate for that model.

    Carreira, Joao, and Andrew Zisserman.
    "Quo vadis, action recognition? a new model and the kinetics dataset."
    proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017.
    """

    def __init__(self, trained_on="kinetics-400") -> None:
        supported_versions = ("kinetics-400", "kinetics-600")
        if trained_on not in supported_versions:
            raise ValueError(
                f"I3D model {trained_on} not supported."
                f"Supported: {', '.join(supported_versions)}"
            )
        self.trained_on = trained_on
        self.name = f"i3d_k{self.trained_on[-3:]}"
        self.model = self.get_model()
        self.labels = get_kinetics_labels(self.trained_on.replace("-", "_"))

    def get_model(self):
        """Load I3D model from TF Model Hub"""
        # See example on how to do other models here
        # https://colab.research.google.com/github/tensorflow/models/blob/master/official/vision/beta/projects/movinet/movinet_tutorial.ipynb
        pretrained_model_url = f"https://tfhub.dev/deepmind/i3d-{self.trained_on}/1"
        return hub.load(pretrained_model_url).signatures["default"]


def get_movinet_model(model_name, metrics=None):
    if model_name == "MOVINET_A2_BASE_K600":
        pretrained_model_url = (
            "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"
        )
    elif model_name == "MOVINET_A2_STREAM_K600":
        pretrained_model_url = "https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification/3"
    else:
        raise ValueError(f"Model name not supported: {model_name}")

    # TODO: set trainable to False
    encoder = hub.KerasLayer(pretrained_model_url, trainable=True)

    inputs = tf.keras.layers.Input(
        shape=[None, None, None, 3], dtype=tf.float32, name="image"
    )

    # Shape of output: [batch_size, 600]
    # TODO 'image' is the name parameter passed to tf.keras.layers.Input above?
    outputs = encoder(dict(image=inputs))

    model = tf.keras.Model(inputs, outputs, name="movinet")

    if metrics:
        model.compile(metrics=metrics)

    return model
