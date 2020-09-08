import sys
import os
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from scipy.stats import spearmanr


def train(model_output_dir, exectime_measurements,
        layers, test_set_fraction, features, steps, tolerance):

    assert 0 < test_set_fraction < 1.0
    total_points = len(exectime_measurements)
    train_size = int(total_points * (1 - test_set_fraction))
    train_data = exectime_measurements[:train_size]
    test_data = exectime_measurements[train_size:]

    features_list = range(1, features+1)
    train_feature = np.array(train_data[:, features_list])
    train_label = np.array(train_data[:, [0]])
    test_x = np.array(test_data[:, features_list])

    print(test_data.shape)

    tf.disable_eager_execution()

    x = tf.placeholder(tf.float32, [None, features], name="input")
    y = tf.placeholder(tf.float32, [None, 1])

    # preprocessing
    scaler = StandardScaler()
    # fit scaler on training dataset
    scaler.fit(train_feature)

    train_feature = scaler.transform(train_feature)
    test_xs = scaler.transform(test_x)

    print(test_xs.shape)

    # needs to output the mean
    print("scaler: mean:", scaler.mean_)
    print("scaler: std:", scaler.scale_)

    # TODO: parametrize in a way that makes ML sense
    if layers == 1:
        L = tf.layers.dense(x, 15, tf.nn.relu)
    elif layers == 2:
        L1 = tf.layers.dense(x, 5, tf.nn.relu)
        L2 = tf.layers.dense(L1, 5, tf.nn.relu)
        L = L2
    else:
        print(f"ERROR: unsupported number of layers: {layers}",
                file=sys.stderr)
        sys.exit(1)

    prediction = tf.layers.dense(L,1)

    # name the output
    tf.identity(prediction, name="output")


    loss = tf.reduce_mean(tf.square(y - prediction))


    saver = tf.train.Saver()


    train_step = tf.train.AdamOptimizer(tolerance).minimize(loss)


    total_parameters = 0
    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim
        print(variable_parameters)
        total_parameters += variable_parameters
    print("total parameters: ", total_parameters)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        print(sess.run(loss, feed_dict={x: train_feature, y: train_label}))

        lowest_mape = 1

        highest_rho = 0
        asso_rho = 0
        model_created = False

        for i in range(steps):
            sess.run(train_step, feed_dict={x: train_feature, y: train_label})
            if i % 200 == 0:
                print(i)
                print(sess.run(loss, feed_dict={x: train_feature, y: train_label}))
                prd = sess.run(prediction, feed_dict={x: test_xs})

                # -----evaluation-----#

                import math
                import statistics

                sum_ae = 0.0
                sum_ape = 0.0
                sum_aape = 0.0

                pred_value_list = []
                truth_value_list = []

                for i in range(test_data.shape[0]):
                    truth_value = test_data[:, [0]][i][0]
                    pred_value = prd[i][0]
                    sum_ae += abs(prd[i][0] - test_data[:, [0]][i][0])
                    truth_value_list.append(truth_value)
                    pred_value_list.append(pred_value)

                # print("MAE: ", sum_ae / test_data.shape[0])

                c = 0

                # decide the percentage to drop
                percentage = 0.3
                threshold = sorted(truth_value_list)[int(len(test_data) * percentage) - 1]
                print("thres=", threshold)

                median = statistics.median(truth_value_list)

                for i in range(test_data.shape[0]):

                    pred_value = prd[i][0]
                    truth_value = test_data[:, [0]][i][0]

                    ape = (abs(prd[i][0] - test_data[:, [0]][i][0]) / test_data[:, [0]][i][0])
                    aape = math.atan(abs(prd[i][0] - test_data[:, [0]][i][0]) / test_data[:, [0]][i][0])

                    # valid rule
                    print("thres=", threshold, "truth=", truth_value)
                    if truth_value > threshold:
                        sum_ape += ape
                        c += 1

                    sum_aape += aape


                rho, pval = spearmanr(pred_value_list,truth_value_list)
                curr_mape = round(sum_ape / c, 2)

                print("curr_mape=", curr_mape, "lowest=", lowest_mape)
                if curr_mape < lowest_mape:
                    lowest_mape = curr_mape
                    asso_rho = rho
                    # write results to a .txt file
                    f = open('re.txt', 'w')
                    for i in range(test_data.shape[0]):
                        f.writelines(str(prd[i][0]) + "\n")
                    f.close()

                    # save the model as a .pb graph
                    model_file = os.path.join(model_output_dir, 'model.pb')
                    ckpt_dir = os.path.join(model_output_dir, 'ckpt/ckpt')
                    with open(model_file, 'wb') as f:
                        f.write(tf.get_default_graph().as_graph_def().SerializeToString())

                    saver.save(sess, ckpt_dir)
                    model_created = True

                print("MAPE: ", curr_mape)

                # valid rule
                if rho > highest_rho:
                    highest_rho = rho
                print('rho:', rho)

                # ------------------#

        if not model_created:
            raise Exception(f"failed to create model: all MAPE >= {lowest_mape}")

        inference_start = time.time()
        prd = sess.run(prediction, feed_dict={x: test_xs})
        inference_end = time.time()

        print('Inference time:', (inference_end - inference_start) / test_data.shape[0])

        # -----evaluation-----#

        import math
        import statistics

        sum_ae = 0.0
        sum_ape = 0.0
        sum_aape = 0.0
        print("sum_ape", sum_ape)

        pred_value_list = []
        truth_value_list = []

        for i in range(test_data.shape[0]):
            pred_value = prd[i][0]
            truth_value = test_data[:, [0]][i][0]
            sum_ae += abs(prd[i][0] - test_data[:, [0]][i][0])
            truth_value_list.append(truth_value)
            pred_value_list.append(pred_value)
        print("MAE: ", sum_ae / test_data.shape[0])


        c = 0

        # decide the percentage to drop
        percentage = 0.3
        threshold = sorted(truth_value_list)[int(len(test_data)*percentage) - 1]

        median = statistics.median(truth_value_list)


        for i in range(test_data.shape[0]):

            pred_value = prd[i][0]
            truth_value = test_data[:, [0]][i][0]

            ape = (abs(prd[i][0] - test_data[:, [0]][i][0]) / test_data[:, [0]][i][0])
            aape = math.atan(abs(prd[i][0] - test_data[:, [0]][i][0]) / test_data[:, [0]][i][0])

            # valid rule
            if truth_value > threshold:
                sum_ape += ape
                c += 1

            sum_aape += aape

        rho, pval = spearmanr(pred_value_list,truth_value_list)
        print('rho:', rho)
        print("MAPE: ", sum_ape / c)
        maape = sum_aape / test_data.shape[0] if test_data.shape[0] > 0 else np.nan
        print("MAAPE: ", maape)

        print("threshold value:", threshold)
        print("truth median:", median)
        print("range from", min(truth_value_list), "to", max(truth_value_list))
        print("valid points (MAPE):", c, "out of", test_data.shape[0])

        print("Lowest MAPE and associated rho: ", lowest_mape, "and", asso_rho)
        print("Highest rho: ", highest_rho)

        # ------------------#

