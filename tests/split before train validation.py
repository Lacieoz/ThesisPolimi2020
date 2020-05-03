import csv
import numpy as np
from os import walk
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, SimpleRNN, Reshape, Input, concatenate, LSTM
from keras.optimizers import Adam

import matplotlib.pyplot as plt

start_time = time.time()

minutes_diff = 60

step_diff = int(minutes_diff / 10)


def get_csv_files_in_folder(path):

    files = []

    for (dirpath, dirnames, filenames) in walk(path):
        for file in filenames:
            if file[-4:] == ".csv":
                files.append({"dirpath": dirpath, "filename": file})

    return files


def extract_data_from_file(filedata):
    dir = filedata["dirpath"]
    filename = filedata["filename"]

    data = []

    with open(dir + "/" + filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print(f'\tSensor {row[0]} registered {row[2]} in the date {row[1]}.')
                # if float(row[2]) > 400:
                #     print(row[1])
                # if row[1][5:7] == "11":
                year = int(row[1][0:4])
                if 2014 < year < 2019:
                    data.append(float(row[2]))
                line_count += 1

                # data.append(float(row[2]))
                # line_count += 1
        #print(f'Processed {line_count} lines.')

    return data


def import_data_input():
    data = []
    path = './data/PadernoDugnano'

    f = get_csv_files_in_folder(path)

    # print(f"CSV files = {f}")

    for file in f:
        data = data + extract_data_from_file(file)

    return data


def import_data_input_niguarda():
    data = []
    path = './data/MilanoNiguarda'

    f = get_csv_files_in_folder(path)

    # print(f"CSV files = {f}")

    for file in f:
        data = data + extract_data_from_file(file)

    return data


def import_data_output():
    data = []
    path = './data/MilanoNiguarda'

    f = get_csv_files_in_folder(path)

    # print(f"CSV files = {f}")

    for file in f:
        data = data + extract_data_from_file(file)

    return data


def set_same_length(data_input, data_output):
    input_length = len(data_input)
    output_length = len(data_output)

    delta = input_length - output_length

    if delta > 0:
        data_input = data_input[delta:]
    elif delta < 0:
        delta = delta * -1
        data_output = data_output[delta:]

    return data_input, data_output


def apply_step_diff(data_input_paderno, data_input_niguarda, data_output):
    input_length = len(data_input_paderno)

    data_input_paderno = data_input_paderno[:input_length-step_diff]
    data_input_niguarda = data_input_niguarda[:input_length-step_diff]
    data_output = data_output[step_diff:]

    return data_input_paderno, data_input_niguarda, data_output


all_data_input_paderno = import_data_input()
all_data_input_niguarda = import_data_input_niguarda()
all_data_output = import_data_output()

print(f"Numero dati input paderno = {len(all_data_input_paderno)}")
print(f"Numero dati input niguarda = {len(all_data_input_niguarda)}")
print(f"Numero dati output = {len(all_data_output)}")

print("--- %s seconds ---" % (time.time() - start_time))

all_data_input_paderno, all_data_input_niguarda, all_data_output = apply_step_diff(all_data_input_paderno, all_data_input_niguarda, all_data_output)

print(f"Numero dati input paderno = {len(all_data_input_paderno)}")
print(f"Numero dati input niguarda = {len(all_data_input_niguarda)}")
print(f"Numero dati output = {len(all_data_output)}")

print("--- %s seconds ---" % (time.time() - start_time))

all_data_input_paderno = np.array(all_data_input_paderno)
all_data_input_niguarda = np.array(all_data_input_niguarda)
all_data_output = np.array(all_data_output)

counter = 0

for index in range(len(all_data_input_paderno)):
    if all_data_input_paderno[index] == -999:
        if all_data_input_paderno[index+1] != -999:
            all_data_input_paderno[index] = (all_data_input_paderno[index - 1] + all_data_input_paderno[index+1])/2
        else:
            all_data_input_paderno[index] = all_data_input_paderno[index - 1]
        counter += 1
    else:
        all_data_input_paderno[index] = all_data_input_paderno[index] # + 30.0

print(f"counter input dirty = {counter}")

counter = 0

for index in range(len(all_data_input_niguarda)):
    if all_data_input_niguarda[index] == -999:
        if all_data_input_niguarda[index+1] != -999:
            all_data_input_niguarda[index] = (all_data_input_niguarda[index - 1] + all_data_input_niguarda[index+1])/2
        else:
            all_data_input_niguarda[index] = all_data_input_niguarda[index - 1]
        counter += 1
    else:
        all_data_input_niguarda[index] = all_data_input_niguarda[index] # + 30.0

print(f"counter input dirty = {counter}")

counter = 0

for index in range(len(all_data_output)):
    if all_data_output[index] == -999:
        if all_data_output[index + 1] != -999:
            all_data_output[index] = (all_data_output[index - 1] + all_data_output[index + 1]) / 2
        else:
            all_data_output[index] = all_data_output[index - 1]
        counter += 1
    else:
        all_data_output[index] = all_data_output[index] # + 30.0


print(f"counter output dirty = {counter}")

# GRAFICO
"""
plt.plot(all_data_input, all_data_output, 'ro')
plt.axis([-100, 450, -100, 550])
plt.show()
"""

# TRAIN, VALIDATION SEPARATION
"""
all_data_input_paderno_train, all_data_input_paderno_test, all_data_input_niguarda_train, all_data_input_niguarda_test, \
all_data_output_train, all_data_output_test = train_test_split(all_data_input_paderno, all_data_input_niguarda, all_data_output, test_size=0.25)
"""


# PROVA

index = 157675

all_data_input_paderno_train = np.array(all_data_input_paderno[:index])
all_data_input_paderno_test = np.array(all_data_input_paderno[index:])
all_data_input_niguarda_train = np.array(all_data_input_niguarda[:index])
all_data_input_niguarda_test = np.array(all_data_input_niguarda[index:])
all_data_output_train = np.array(all_data_output[:index])
all_data_output_test = np.array(all_data_output[index:])
"""
all_data_input_paderno_train, all_data_input_paderno_test_2, all_data_input_niguarda_train, all_data_input_niguarda_test_2, \
all_data_output_train, all_data_output_test_2 = train_test_split(all_data_input_paderno_train, all_data_input_niguarda_train, all_data_output_train, test_size=0.01)
"""
all_data_input_paderno_train_2, all_data_input_paderno_test, all_data_input_niguarda_train_2, all_data_input_niguarda_test, \
all_data_output_train_2, all_data_output_test = train_test_split(all_data_input_paderno_test, all_data_input_niguarda_test, all_data_output_test, test_size=0.99)

mean = np.mean(all_data_output_train)

sum = 0
for output_placeholder in all_data_output_train:
    sum += np.square(output_placeholder - mean)

print("sum " + str(sum))








# LINEAR MODEL

lr = 0.0005
loss = tf.keras.losses.MeanSquaredError()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False)

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
paderno_input = Input(shape=(1,), name='paderno_input')

niguarda_input = Input(shape=(1,), name='niguarda_input')

x = concatenate([paderno_input, niguarda_input])

# We stack a deep densely-connected network on top
x = Dense(16, activation='elu')(x)
x = Dense(16, activation='elu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='linear', name='main_output')(x)


def coeff_determination(y_true, y_pred):
    ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())


def mean_diff_pred(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def max_diff_pred(y_true, y_pred):
    return K.max(K.abs(y_pred - y_true))


optimizer = Adam(learning_rate=lr, amsgrad=False)

model = Model(inputs=[paderno_input, niguarda_input], outputs=main_output)

model.compile(optimizer, loss=loss, metrics=[coeff_determination, mean_diff_pred, max_diff_pred], loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
              target_tensors=None)

model.summary()


model.fit(x=[all_data_input_paderno_train, all_data_input_niguarda_train], y=all_data_output_train, batch_size=None, epochs=40, verbose=2, callbacks=[early_stop],
          validation_split=None, validation_data=([all_data_input_paderno_test, all_data_input_niguarda_test], all_data_output_test),
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
          validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)


# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
paderno_input = Input(shape=(1,), name='paderno_input')

niguarda_input = Input(shape=(1,), name='niguarda_input')

x = concatenate([paderno_input, niguarda_input])

x = Dense(16, activation='elu')(x)
x = Reshape((1, 16))(x)
x = LSTM(16, activation='elu')(x)
x = Reshape((1, 16))(x)
x = LSTM(16, activation='elu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='linear', name='main_output')(x)

optimizer = Adam(learning_rate=lr, amsgrad=False)

model_RNN = Model(inputs=[paderno_input, niguarda_input], outputs=main_output)

model_RNN.compile(optimizer, loss=loss, metrics=[coeff_determination], loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
              target_tensors=None)

model_RNN.summary()

model_RNN.fit(x=[all_data_input_paderno_train, all_data_input_niguarda_train], y=all_data_output_train, batch_size=None, epochs=40, verbose=2, callbacks=[early_stop],
          validation_split=None, validation_data=([all_data_input_paderno_test, all_data_input_niguarda_test], all_data_output_test),
          shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None,
          validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False)



