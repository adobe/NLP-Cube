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

class CNN:
    def __init__(self, model):
        self.model = model
        self.layers = []


    def add_layer_conv(self, kernel_x, kernel_y, stride_x, stride_y, num_filters, same=True):

        if len(self.layers)==0:
            num_input_chans=1
        else:
            num_input_chans=self.layers[-1].num_output_chans

        self.layers.append(CNNConvLayer(self.model, kernel_x, kernel_y, stride_x, stride_y, num_input_chans, num_filters,same==False))

    def add_layer_pooling(self, kernel_x, kernel_y, stride_x, stride_y):
        if len(self.layers)==0:
            num_input_chans=1
        else:
            num_input_chans=self.layers[-1].num_output_chans

        self.layers.append(CNNPoolingLayer(self.model, kernel_x, kernel_y, stride_x, stride_y, num_input_chans))

    def apply(self, input_x):
        #shape=input_x.npvalue().shape
        #input_x=dy.reshape(input_x, ((shape[0], shape[1], 1))
        #print ""
        for layer in self.layers:
            #print input_x.npvalue().shape
            input_x=layer.apply(input_x)
        return input_x

class CNNConvLayer:
    def __init__(self, model, x, y, s_x, s_y, num_input_chans, num_filters, is_valid):
        self.s_x = s_x
        self.s_y = s_y
        self.is_valid=is_valid

        self.model = model
        self.kernel = self.model.add_parameters((x, y, num_input_chans, num_filters))
        #self.kernel_s = self.model.add_parameters((x, y, num_input_chans, num_filters))
        #self.kernel_t = self.model.add_parameters((x, y, num_input_chans, num_filters))
        #self.bias_s=self.model.add_parameters((num_filters))
        #self.bias_t=self.model.add_parameters((num_filters))
        self.num_output_chans = num_filters# WTF - credeam ca se mareste numarul de feature maps !!!!!#num_input_chans * num_filters

    def apply(self, x_input):
        #print "\tapplying",self.kernel.expr().npvalue().shape,"convolution"
        #output_s = dy.conv2d_bias(x_input, self.kernel_s.expr(), self.bias_s.expr(), (self.s_x, self.s_y), is_valid=self.is_valid)
        #output_t = dy.conv2d_bias(x_input, self.kernel_t.expr(), self.bias_t.expr(), (self.s_x, self.s_y), is_valid=self.is_valid)
        #return dy.cmult(dy.tanh(output_t),dy.logistic(output_s))
        output=dy.conv2d(x_input, self.kernel.expr(update=True), (self.s_x, self.s_y), is_valid=self.is_valid)
        return dy.rectify(output)

class CNNPoolingLayer:
    def __init__(self, model, x, y, s_x, s_y, num_input_chans):
        self.s_x = s_x
        self.s_y = s_y
        self.x = x
        self.y = y
        self.model = model
        self.num_output_chans = num_input_chans

    def apply(self, x_input):
        return dy.maxpooling2d(x_input, (self.x, self.y), (self.s_x, self.s_y))
