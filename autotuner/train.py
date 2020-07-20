import sys
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
from scipy.stats import spearmanr

FILENAME = sys.argv[1]
HEADER = int(sys.argv[2])
TRAIN_SIZE = int(sys.argv[3])
TRAIN_FEATURES = int(sys.argv[4])
TRAIN_STEPS = int(sys.argv[5])
TRAIN_TOL=float(sys.argv[6])
LAYERS=int(sys.argv[7])
MAPE_LIM1=float(sys.argv[8])
MAPE_LIM2=float(sys.argv[9])
MODEL_PATH_PREFIX = sys.argv[10]

header_arg = 0 if HEADER != 0 else None
data = np.array(pd.read_csv(FILENAME, header=header_arg))

train_data = data[:TRAIN_SIZE]
test_data = data[TRAIN_SIZE:]

features = range(1, TRAIN_FEATURES+1)
train_feature = np.array(train_data[:, features])
train_label = np.array(train_data[:, [0]])
test_x = np.array(test_data[:, features])

print(test_data.shape)

tf.disable_eager_execution()

x = tf.placeholder(tf.float32, [None, TRAIN_FEATURES], name="input")
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
if LAYERS == 1:
    L = tf.layers.dense(x, 15, tf.nn.relu)
elif LAYERS == 2:
    L1 = tf.layers.dense(x, 5, tf.nn.relu)
    L2 = tf.layers.dense(L1, 5, tf.nn.relu)
    L = L2
else:
    print(f"ERROR: unsupported number of layers: {LAYERS}",
            file=sys.stderr)
    sys.exit(1)

prediction = tf.layers.dense(L,1)

# name the output
tf.identity(prediction, name="output")


loss = tf.reduce_mean(tf.square(y - prediction))


saver = tf.train.Saver()


train_step = tf.train.AdamOptimizer(TRAIN_TOL).minimize(loss)


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

    for i in range(TRAIN_STEPS):
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
            curr_mape = round(sum_ape / c, 2)

            if curr_mape < lowest_mape:
                lowest_mape = curr_mape
                asso_rho = rho
                # write results to a .txt file
                f = open('re.txt', 'w')
                for i in range(test_data.shape[0]):
                    f.writelines(str(prd[i][0]) + "\n")
                f.close()

                # save the model as a .pb graph
                with open('model.pb', 'wb') as f:
                    f.write(tf.get_default_graph().as_graph_def().SerializeToString())

                saver.save(sess, "model/my-model.ckpt")

            print("MAPE: ", curr_mape)

            # valid rule
            if rho > highest_rho:
                highest_rho = rho
            print('rho:', rho)

            # ------------------#

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

