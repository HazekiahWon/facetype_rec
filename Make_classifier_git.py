from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import config
import tensorflow as tf
import numpy as np
import argparse
import facenet
import detect_face
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit, GridSearchCV
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score


with tf.Graph().as_default():

    with tf.Session() as sess:

        datadir = config.aligned_train
        dataset = facenet.get_dataset(datadir)
        paths, labels = facenet.get_image_paths_and_labels(dataset)
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))

        print('Loading feature extraction model')
        modeldir = config.model_params
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        # Run forward pass to calculate embeddings
        print('Calculating features for images')
        batch_size = 100
        image_size = 160
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        for i in range(nrof_batches_per_epoch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

        classifier_filename = os.path.join(config.clf_dir, config.clf_name)
        os.makedirs(config.clf_dir,exist_ok=True)
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        train_val_rat = 0.15
        df = pd.concat([pd.Series(paths),pd.Series(labels)], axis=1) # n,2
        shuffled_ind = df.index
        np.random.shuffle(shuffled_ind)
        x = emb_array[shuffled_ind]
        y = labels[shuffled_ind]
        df.columns = ['path','label']
        val_list = list()
        # train_list = list()
        for n,g in df.groupby('label'):
            val_list.extend(np.random.choice(g.index, size=int(len(g)*train_val_rat)))
        train_list = list(set(df.index)-set(val_list))

        # Train classifier
        print('Training classifier')
        # model = SVC(kernel='rbf', probability=True)
        # model.fit(emb_array[train_list], labels[train_list])
        # val_pred = model.predict(emb_array[val_list])
        # acc = sum(labels[val_list]==val_pred)/len(val_list)#accuracy_score(labels[val_list],val_pred)
        # r2 = r2_score(labels[val_list],val_pred)
        # print(acc, r2)
        parameters = {
            'C': np.arange(1, 500 + 1, 5).tolist(),
            'kernel': ['poly', 'rbf','sigmoid'],  # precomputed,'poly', 'sigmoid'
            'degree': np.arange(0, 100 + 0, 1).tolist(),
            'gamma': np.arange(0.0, 10.0 + 0.0, 0.1).tolist(),
            'coef0': np.arange(0.0, 10.0 + 0.0, 0.1).tolist(),
            'shrinking': [True],
            'probability': [False],
            'tol': np.arange(0.001, 0.01 + 0.001, 0.001).tolist(),
            'cache_size': [2000],
            'class_weight': [None],
            'verbose': [False],
            'max_iter': [-1],
            'random_state': [None],
        }

        # C_range = np.logspace(-2, 10, 13)
        # gamma_range = np.logspace(-9, 3, 13)
        # param_grid = dict(gamma=gamma_range, C=C_range)
        # cv = StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=42)
        # model = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
        # model.fit(x,y)

        model = RandomizedSearchCV(n_iter=500,
                                               estimator=SVC(),
                                               param_distributions=parameters,
                                               n_jobs=4,
                                               iid=True,
                                               refit=True,
                                               cv=10,
                                               verbose=1,
                                               pre_dispatch='2*n_jobs'
                                               )  # scoring = 'accuracy'
        model.fit(emb_array, labels)
        print(model.best_estimator_)
        print(model.best_score_)
        print(model.best_params_)
        clf = model.best_estimator_
        pred = clf.predict(emb_array[val_list])
        print('val acc:', accuracy_score(labels[val_list], pred),
              r2_score(accuracy_score(labels[val_list], pred)))

        # Create a list of class names
        class_names = [cls.name.replace('_', ' ') for cls in dataset]

        # Saving classifier model
        with open(classifier_filename_exp, 'wb') as outfile:
            pickle.dump((model, class_names), outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)
        print('Goodluck')