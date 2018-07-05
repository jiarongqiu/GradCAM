#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jiarong Qiu

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
from scipy.misc import imread, imresize
import numpy as np
from skimage import io
from skimage.transform import resize


class GradCam:

    def __init__(self, model_path, nb_classes):
        self.model_path = model_path
        self.nb_classes = nb_classes
        self._load_model()
        self.fetch_key_tensors()

    # TODO create your own process
    def preprocess(self, x):
        x = x / 255.0
        x = x - np.array([0.485, 0.456, 0.406])
        x = x / np.array([0.229, 0.224, 0.225])
        return x

    # TODO fetch key tensors
    def fetch_key_tensors(self):
        """
            X:input tensor
            logit:logit output tensor
            layer:gradient dependent layer to the logit
            ... :fetch and set other tensors and their default values for inference
        """
        self.X = self.sess.graph.get_tensor_by_name('input:0')
        self.logit = self.sess.graph.get_tensor_by_name("finetune_dense1/BiasAdd:0")
        self.layer = self.sess.graph.get_tensor_by_name("ResNetnSequentialnlayer4nnBasicBlockn1nnReLUnrelun168:0")

        is_training = self.sess.graph.get_tensor_by_name("Placeholder:0")
        self.feed_dict = {is_training:False}

    def _load_model(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        if not os.path.exists(self.model_path + ".meta"):
            raise ValueError("model path not exist")
        tf.train.import_meta_graph(self.model_path + ".meta")

        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        saver.restore(sess, self.model_path)
        self.sess = sess

    def _visualize_img(self, img, feed_dict, out_path):
        pred = self.sess.run(self.logit, feed_dict=feed_dict)
        pred_class = np.argmax(pred[0])

        one_hot = tf.sparse_to_dense(pred_class, [self.nb_classes], 1.0)
        signal = tf.multiply(self.logit, one_hot)
        loss = tf.reduce_mean(signal)

        grads = tf.gradients(loss, self.layer)[0]
        # Normalizing the gradients
        norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

        output, grads_val = self.sess.run([self.layer, norm_grads], feed_dict=feed_dict)
        output = output[0]  # [7,7,512]
        grads_val = grads_val[0]  # [7,7,512]

        weights = np.mean(grads_val, axis=(0, 1))  # [512]
        cam = np.ones(output.shape[0: 2], dtype=np.float32)  # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = resize(cam, (224, 224))

        # Converting grayscale to 3-D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        img = img.astype(float)
        img /= img.max()

        # Superimposing the visualization with the image.
        new_img = img + 3 * cam3
        new_img /= new_img.max()

        # Display and save
        # io.imshow(new_img)
        out_path=out_path.split("/")
        out_path[-1]=str(pred_class)+'_'+out_path[-1]
        out_path="/".join(out_path)
        io.imsave(out_path, new_img)
        return pred_class

    def _load_image(self, img_path):
        img = imread(img_path, mode='RGB')
        img = imresize(img, (224, 224))
        # Converting shape from [224,224,3] tp [1,224,224,3]
        X = np.expand_dims(img, axis=0)
        # Converting RGB to BGR for VGG
        X = X[:, :, :, ::-1]
        return X, img

    def visualize_folder(self, in_folder, out_folder):
        img_list = [img for img in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, img))]
        for idx,img_name in enumerate(img_list):
            if idx %10==0:print(idx/len(img_list))
            img_path = os.path.join(in_folder, img_name)
            out_path = os.path.join(out_folder, img_name)
            x, img = self._load_image(img_path)
            x = self.preprocess(x)
            feed_dict= self.feed_dict
            feed_dict[self.X]=x
            pred = self._visualize_img(img, feed_dict, out_path)
            print(img_name, pred)

    def visualize_imgs(self,imgs_path,out_folder):
        ret =[]
        for idx,img_path in enumerate(imgs_path):
            if idx % 10 == 0: print(idx / len(imgs_path))
            img_name=img_path.split("/")[-1]
            out_path = os.path.join(out_folder, img_name)
            x, img = self._load_image(img_path)
            x = self.preprocess(x)
            feed_dict= self.feed_dict
            feed_dict[self.X]=x
            pred = self._visualize_img(img, feed_dict, out_path)
            ret.append(pred)
        return np.array(ret)

