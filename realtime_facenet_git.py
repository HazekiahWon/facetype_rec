from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2

import numpy as np
import argparse
import facenet
import detect_face
import os

import time

import pickle
import config
import glob
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--align_params', type=str, default=config.align_params,
                    help='the params file for the alignment module')
parser.add_argument('--show_flag', type=bool, default=False,
                    help='show the image while testing')
parser.add_argument('--model_params', type=str, default=config.model_params,
                    help='the params file for the model ')
parser.add_argument('--clf_name', type=str, default=config.clf_name,
                    help='classifier name')
parser.add_argument('--clf_dir', type=str, default=config.clf_dir,
                    help='classifier dir')
parser.add_argument('--rel_path', type=str, default=config.rel_path,
                    help='the relative path of the input data dir')
parser.add_argument('--output_file', type=str, default='test_results',
                    help='the output file name for the test results')

args = parser.parse_args()

show_flag = args.show_flag
import os, sys
# Translate asset paths to useable format for PyInstaller
def resource_path(relative_path):
  if hasattr(sys, '_MEIPASS'):
      return os.path.join(sys._MEIPASS, relative_path)
  return os.path.join(os.path.abspath('.'), relative_path)

def one_by_one(rel_path):
    print('Start Recognition!')
    prevTime = 0
    img_list = glob.glob(os.path.join(rel_path, '*'))
    results = list()
    # cnt = 0
    # ok_list = list()
    for img_path in img_list:  # for each image in the list
        res = None
        frame = cv2.imread(img_path)
        # ret, frame = video_capture.read()

        # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
        if frame is None: continue
        curTime = time.time()  # calc fps
        timeF = frame_interval

        if (c % timeF == 0):  # detect faces in the current image
            find_results = []

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('Detected_FaceNum: %d' % nrof_faces)

            if nrof_faces > 0:

                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces, 4), dtype=np.int32)

                for i in range(nrof_faces):  # crop all the faces
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is out of range!')
                        continue

                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[0] = facenet.flip(cropped[0], False)
                    scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                    scaled[0] = cv2.resize(scaled[0], (input_image_size, input_image_size),
                                           interpolation=cv2.INTER_CUBIC)
                    scaled[0] = facenet.prewhiten(scaled[0])
                    scaled_reshape.append(scaled[0].reshape(-1, input_image_size, input_image_size, 3))

                    feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    if i == 0:
                        res = best_class_indices[0]
                        # ok_list.append(cnt)
                        # cnt += 1
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)  # boxing face

                    # plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    # print('result: ', best_class_indices[0])
                    if show_flag:
                        for H_i in class_names:
                            if class_names[best_class_indices[0]] == H_i:
                                result_names = class_names[best_class_indices[0]]
                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
            else:
                print('Unable to align')

        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        str = 'FPS: %2.3f' % fps
        text_fps_x = len(frame[0]) - 150
        text_fps_y = 20
        if show_flag:
            cv2.putText(frame, str, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            # c+=1
            cv2.imshow('Video', frame)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        a,b,m,n = bb[0]
        if res is not None:
            results.append([res]+list(predictions[0])+[m-a,n-b])
        else: results.append([res]*10)

    # video_capture.release()
    # #video writer
    # out.release()
    cv2.destroyAllWindows()
    # pred = np.zeros_like(img_list)
    # print(len(ok_list),len(results))
    # pred[ok_list] = results
    # print(pred)
    results = np.array(results)
    # print(results.shape)
    # print(results)
    labels = [class_names[int(i)] if i is not None else None for i in results[:,0]]
    comb = np.concatenate([np.array(img_list).reshape((-1,1)),np.array(labels).reshape((-1,1)), results[:,1:]], axis=1)#list(zip(img_list, results))
    # print(comb.shape)
    pd.DataFrame(comb).to_csv(args.output_file+'.csv', index=False, header=['filename','label','circle','diamond','egg','long','polygon','square','triangle','dx','dy'])

def batch_inp(rel_path):
    print('Start Recognition!')
    prevTime = 0
    img_list = glob.glob(os.path.join(rel_path, '*'))
    results = list()
    cnt = 0
    ok_ind = list()
    for img_path in img_list:  # for each image in the list
        res = None
        frame = cv2.imread(img_path)
        # ret, frame = video_capture.read()

        # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

        curTime = time.time()  # calc fps
        timeF = frame_interval

        if (c % timeF == 0):  # detect faces in the current image
            find_results = []

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('Detected_FaceNum: %d' % nrof_faces)

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                scaled_reshape = []

                bb = [int(np.round(i)) for i in det[0]]
                # inner exception
                if bb[0] <= 0 or bb[1] <= 0 or bb[2] >= len(frame[0]) or bb[3] >= len(frame):
                    print('face is out of range!')
                    continue

                cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                cropped = facenet.flip(cropped, False)
                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                scaled = cv2.resize(scaled, (input_image_size, input_image_size),
                                       interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape.append(scaled.reshape(input_image_size, input_image_size, 3))
                ok_ind.append(cnt)
        cnt += 1

    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
    emb_array = sess.run(embeddings, feed_dict=feed_dict) # n,n_emb
    predictions = model.predict_proba(emb_array)
    best_class_indices = np.argmax(predictions, axis=1) # n,1
    # best_class_probabilities = np.max(predictions, axis=1)

    results = np.zeros_like(img_list)
    results[ok_ind] = [class_names[i] for i in best_class_indices]
    comb = list(zip(img_list, results))
    pd.DataFrame(comb).to_csv('test_results.csv')


print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, resource_path(args.align_params))

        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        frame_interval = 3
        batch_size = 1000
        image_size = 182
        input_image_size = 160

        # # classes = ['circle', 'diamond', 'egg', 'long', 'polygon', 'square', 'triangle']    #train human name
        # with open(config.classes_map, 'rb') as f:
        #     class_names = pickle.load(f)
        #     print(class_names)

        print('Loading feature extraction model')
        modeldir = args.model_params
        facenet.load_model(resource_path(modeldir))

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = os.path.join(args.clf_dir, args.clf_name)
        classifier_filename_exp = resource_path(os.path.expanduser(classifier_filename))
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        # video_capture = cv2.VideoCapture(0)
        c = 0
        one_by_one(resource_path(args.rel_path))
        # if show_flag:
        #     one_by_one(args.rel_path)
        # else: batch_inp(args.rel_path)
        print('finish.')






