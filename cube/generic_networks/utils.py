#
# Author: Tiberiu Boros
#
# Copyright (c) 2018 Adobe Systems Incorporated. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import dynet as dy
import numpy as np


def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc):
    builder = dy.VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens, pc)
    for layer, params in enumerate(builder.get_parameters()):
        W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer > 0 else input_dims))
        W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
        params[0].set_value(np.concatenate([W_x] * 4, 0))
        params[1].set_value(np.concatenate([W_h] * 4, 0))
        b = np.zeros(4 * lstm_hiddens, dtype=np.float32)
        b[lstm_hiddens:2 * lstm_hiddens] = -1.0
        params[2].set_value(b)
    return builder


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print (output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in xrange(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                        np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))
