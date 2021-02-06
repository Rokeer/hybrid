# -*- coding: utf-8 -*-
# @Author: colinzhang
# @Date:   2/17/20 9:53 AM

import tensorflow as tf
from transformers import TFAlbertModel


class AlbertLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            transformer_path="albert-base-v2",
            pooling="mean",
            trainable=False,
    ):
        super(AlbertLayer, self).__init__()
        self.trainable = trainable
        self.output_size = 768
        self.pooling = pooling
        self.transformer_path = transformer_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

    def build(self, input_shape):
        self.albert = TFAlbertModel.from_pretrained(self.transformer_path)

    def call(self, inputs):
        albert_inputs = [inputs[:, :, 0], inputs[:, :, 1], inputs[:, :, 2]] # [input_ids, input_mask, segment_ids]

        if self.pooling == "first":
            # get pooled_output
            pooled = self.albert(albert_inputs)[1]
        elif self.pooling == "mean":
            # get sequence_output
            result = self.albert(albert_inputs)[0]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(albert_inputs[1], tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)
