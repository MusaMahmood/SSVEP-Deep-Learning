import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from sklearn import preprocessing
import numpy as np
import itertools as it
import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import os
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import os.path as path

MODEL_NAME = 'ssvep_convnet'

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')

FLAGS = tf.app.flags.FLAGS

# GLOBALS:
localDirectory = \
    "F:\OneDrive - Georgia Institute of Technology\Publishing\_SSVEP\SSVEP Classification"
exportDirectory = \
    "F:\OneDrive - Georgia Institute of Technology\Publishing\_SSVEP\SSVEP Classification\exports"
exportFolderNameNew = 'v0.0.3'
pathtr = r'S1copy'
# pathtr=r'new/S1'
dflist = []
X_ = []
Y_ = []
trainingfiles = glob.glob(pathtr + "/*.csv")
trainbatch_size = 256
valbatch_size = 10

for filenametr in trainingfiles:
    df = pd.read_csv(filenametr, header=None)
    dflist.append(df)

data_ = pd.concat(dflist, axis=0)
data = np.asarray(data_)
print("data_array final shape: \n", data.shape)

def moving_window(data, length, step=1):
    streams = it.tee(data, length)
    return zip(*[it.islice(stream, i, None, step * length) for stream, i in zip(streams, it.count(step=step))])


dataList = list(moving_window(data, 300))
print("ArrayDims: ", len(dataList))
dataArray = np.asarray(dataList)
print("ArrayDims: ", dataArray.shape)

for dataWindow in dataArray:
    # Checks if all values in window (on 0 axis) match
    print("data_window[:, 2]",np.asarray(dataWindow).shape)
    print("np.all(dataWindow[:, 2], axis=0)", np.all(dataWindow[:, 2]))
    count = np.count_nonzero(dataWindow[:, 2] == dataWindow[0, 2])
    print("count", count)
    sys.exit()
    labeledData = np.all(dataWindow == dataWindow[0, :], axis=0)
    # All lables match?
    if labeledData[2]:
        # dW[:, start at coln 0, to column 1, step of 1]
        # index stop @ 2 means stop before 2????? Idiotic slicing/indexing.
        x = dataWindow[:, 0:2:1]
        # print("dataWindow shape: ", np.asarray(x).shape)
        # print("x: ", x)
        # sys.exit()
        # Not sure what this does: normalizing?
        minmax_scale = preprocessing.MinMaxScaler().fit(x)
        x = minmax_scale.transform(x)
        # append after modification.
        X_.append(x)
        Y_.append(dataWindow[0, 2])

y = pd.get_dummies(Y_)
y = y.values
y = np.asarray(y)
x = np.asarray(X_)
X = x.astype(np.float32)
Y = y.astype(np.float32)

Xtrain, Xtestval, Ytrain, Ytestval = train_test_split(X, Y, train_size=0.9, random_state=1)

# Starting the Session
sess = tf.InteractiveSession()

# creating plaaceholders
x = tf.placeholder(tf.float32, shape=[None, 300, 2])
y = tf.placeholder(tf.float32, shape=[None, 5])


# weights and bias functions for convolution
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convoluton and maxpooling functions
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 1, 1], padding='SAME')


# reshape in input
x_input = tf.reshape(x, [-1, 300, 2, 1])
# print("x_input, reshaped:", x_input)
# sys.exit()

# first convolution and pooling
W_conv1 = weight_variable([5, 1, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_input, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolution and pooling
W_conv2 = weight_variable([5, 1, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fully connected layer1,the shape of the patch should be defined
W_fc1 = weight_variable([75 * 2 * 64, 1024])
b_fc1 = bias_variable([1024])

# the input should be shaped/flattened
h_pool2_flat = tf.reshape(h_pool2, [-1, 75 * 2 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# fully connected layer2
W_fc2 = weight_variable([1024, 2048])
b_fc2 = bias_variable([2048])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# weight and bias of the output layer
W_fco = weight_variable([2048, 5])
b_fco = bias_variable([5])

y_conv = tf.matmul(h_fc2_drop, W_fco) + b_fco

X_EXPORT = tf.identity(x_input, name='X_EXPORT')

Y_EXPORT = tf.nn.softmax(y_conv, name='Y_EXPORT')

# training and reducing the cost/loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))

# Accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

# TODO: FIX THIS: FOR SAVING MODEL>
saver = tf.train.Saver(tf.global_variables())

train_writer = tf.summary.FileWriter(localDirectory + '/train', sess.graph)

validation_writer = tf.summary.FileWriter(localDirectory + '/validation')

tf.global_variables_initializer().run()

# save graph in .pbtxt
tf.train.write_graph(sess.graph_def, localDirectory, 'a1.pb', as_text=False)
# save list of words???
with gfile.GFile(
        os.path.join(localDirectory, 'a1_labels.txt'), 'w') as f:
    f.write('\nAlpha15Hz16Hz18Hz20Hz')
val_step = 0
for i in range(5000):
    offset = (i * trainbatch_size) % (Xtrain.shape[0] - trainbatch_size)
    batch_xtrain = Xtrain[offset:(offset + trainbatch_size)]
    batch_ytrain = Ytrain[offset:(offset + trainbatch_size)]
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xtrain, y: batch_ytrain, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    if i % 20 == 0:
        # Calculate batch loss and accuracy
        offset = (val_step * valbatch_size) % (Xtestval.shape[0] - valbatch_size)
        batch_xval = Xtestval[offset:(offset + valbatch_size), :, :]
        batch_yval = Ytestval[offset:(offset + valbatch_size), :]
        val_accuracy = accuracy.eval(feed_dict={x: batch_xval, y: batch_yval, keep_prob: 1.0})
        print("Validation step %d, validation accuracy %g" % (val_step, val_accuracy))
        val_step += 1

    train_step.run(feed_dict={x: batch_xtrain, y: batch_ytrain, keep_prob: 0.15})

    # test accuracy
TestAcc = sess.run(accuracy, feed_dict={x: Xtestval, y: Ytestval, keep_prob: 1.0})
print("Testing Accuracy:", TestAcc)

# export_path = os.path.join(
#             tf.compat.as_bytes(localDirectory),
#             tf.compat.as_bytes('exportModel'+'.ckpt'))

export_path = os.path.join(localDirectory,
                           'exportModel' + '.ckpt')
tf.train.write_graph(sess.graph_def, localDirectory, 'a2.pb', as_text=False)
saver.save(sess, export_path, global_step=val_step)
user_input = input('Continue?')


def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
                              'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
                              "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def, input_node_names, [output_node_name],
        tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


if user_input == "1":
    export_model(X_EXPORT, Y_EXPORT)

    # build signature def map:
    # serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
    # table = tf.contrib.lookup.index_to_string_table_from_tensor(
    #     tf.constant([str(i) for i in range(5)]))
    # values, indices = tf.nn.indices = tf.nn.top_k(Y_EXPORT, 5)
    # prediction_classes = table.lookup(tf.to_int64(indices))
    # classification_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    # classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
    # classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)
    # # sig:
    # classification_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={
    #             tf.saved_model.signature_constants.CLASSIFY_INPUTS:
    #                 classification_inputs
    #         },
    #         outputs={
    #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
    #                 classification_outputs_classes,
    #             tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
    #                 classification_outputs_scores
    #         },
    #         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
    #     )
    # )
    # tensor_info_x = tf.saved_model.utils.build_tensor_info(X_EXPORT)
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(Y_EXPORT)
    #
    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #         inputs={'raw_eeg_data': tensor_info_x}, # not sure what this is supposed to be
    #         outputs={'scores': tensor_info_y},
    #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    #
    # export_path_base = sys.argv[-1]
    # export_path = os.path.join(
    #     tf.compat.as_bytes(exportDirectory),
    #     tf.compat.as_bytes(exportFolderNameNew))
    #
    # print("Exporting trained model to:", export_path)
    #
    # builder = saved_model_builder.SavedModelBuilder(export_path)
    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    # builder.add_meta_graph_and_variables(
    #     sess, ["serve"],
    #     signature_def_map={
    #         'predict_data':
    #             prediction_signature,
    #         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
    #             classification_signature,
    #     },
    #     legacy_init_op=legacy_init_op)
    # # builder.add_meta_graph(sess, signature_def_map={
    # #     'predict_data':
    # #         prediction_signature
    # # }, legacy_init_op=legacy_init_op)
    # builder.save()
    print("Done Exporting")
else:
    print("Terminating")
