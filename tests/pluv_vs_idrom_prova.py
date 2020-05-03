import csv
import numpy as np
from os import walk
import time
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, SimpleRNN, Reshape, Input, concatenate, LSTM
from keras.optimizers import Adam

import matplotlib.pyplot as plt

minutes_diff = 40

step_diff = int(minutes_diff / 10)


def get_csv_files_in_folder(path):

    files = []

    for (dirpath, dirnames, filenames) in walk(path):
        for file in filenames:
            if file[-4:] == ".csv":
                files.append({"dirpath": dirpath, "filename": file})

    return files


def extract_data_from_file_idrometer(filedata):
    dir = filedata["dirpath"]
    filename = filedata["filename"]

    data = []

    with open(dir + "/" + filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        index = 0
        year = 0

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                year = int(row[1][0:4])
                if 2014 < year < 2019:
                    data.append(float(row[2]))
                line_count += 1

        # print(f'Processed {line_count} lines.')

    return data


def extract_data_from_file_rain(filedata):
    dir = filedata["dirpath"]
    filename = filedata["filename"]

    data = []
    data_for_mean = []

    for column in range(21):
        data.append([])

    with open(dir + "/" + filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif line_count == 1:
                for column in range(21):
                    try:
                        data[column].append(float(row[column + 2]))
                    except:
                        data[column].append(0.0)
                line_count += 1
            else:
                if line_count % 2 == 0:
                    for column in range(21):  # we have to make data 10 minutes apart instead of 5 minutes apart
                        try:
                            data_for_mean.append(float(row[column + 2]))
                        except:
                            data_for_mean.append(0.0)
                else:
                    for column in range(21):
                        try:
                            data_to_use = float(row[column + 2])
                        except:
                            data_to_use = 0.0

                        mean = (data_to_use + data_for_mean[column]) / 2
                        data[column].append(mean)

                line_count += 1

        print(f'Processed {line_count} lines.')

    return data


def import_data_input():
    data = []
    path = './data/Pluviometri'

    f = get_csv_files_in_folder(path)

    # print(f"CSV files = {f}")

    for file in f:
        data = (extract_data_from_file_rain(file))

    return data


def import_data_output():
    data = []
    path = './data/MilanoNiguarda'

    f = get_csv_files_in_folder(path)

    # print(f"CSV files = {f}")
    length = len(f)
    index = 0

    for file in f:
        if index != 0 and index != length-1:  # useless data, wouldn't be used anyway
            data = data + extract_data_from_file_idrometer(file)
        index += 1
    return data


def delta_length(data_input, data_output):
    input_length = len(data_input[0])
    output_length = len(data_output)

    delta = input_length - output_length

    print("delta = %s" % delta)

    return delta


def apply_step_diff(data_input_paderno, data_input, data_output):
    length = len(data_output)

    index = 0
    for input in data_input:
        data_input[index] = input[:length-step_diff]
        index += 1

    data_input_paderno = data_input_paderno[:length - step_diff]
    data_output = data_output[step_diff:]

    return data_input_paderno, data_input, data_output


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
                year = int(row[1][0:4])
                if 2014 < year < 2019:
                    data.append(float(row[2]))
                line_count += 1
        #print(f'Processed {line_count} lines.')

    return data


def import_data_input_paderno():
    data = []
    path = './data/PadernoDugnano'

    f = get_csv_files_in_folder(path)

    # print(f"CSV files = {f}")

    for file in f:
        data = data + extract_data_from_file(file)

    return data


start_time_file_rain = time.time()
all_data_input = import_data_input()
print("--- To extract all data rain = %s seconds ---" % (time.time() - start_time_file_rain))

start_time_file_rain = time.time()
all_data_input_paderno = import_data_input_paderno()
print("--- To extract all data paderno = %s seconds ---" % (time.time() - start_time_file_rain))

start_time_file_levels = time.time()
all_data_output = import_data_output()
print("--- To extract all data niguarda = %s seconds ---" % (time.time() - start_time_file_levels))

delta_length(all_data_input, all_data_output)

print(f"Numero dati input rain = {len(all_data_input[0])}")
print(f"Numero dati input levels = {len(all_data_input_paderno)}")
print(f"Numero dati output = {len(all_data_output)}")

all_data_input_paderno, all_data_input, all_data_output = apply_step_diff(all_data_input_paderno, all_data_input, all_data_output)

for input in all_data_input:
    input = np.array(input)
all_data_output = np.array(all_data_output)

for index in range(len(all_data_output)):
    if int(all_data_output[index]) == 1630:
        print(index)

counter = 0

for input in all_data_input:

    for index in range(len(input)):
        if input[index] == -999:
            if input[index+1] != -999:
                input[index] = (input[index - 1] + input[index+1]) / 2
            else:
                input[index] = input[index - 1]
            counter += 1
        else:
            input[index] = input[index] # + 30.0

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
"""
# GRAFICO
#for input in all_data_input:
plt.plot(all_data_input_paderno, all_data_output, 'ro')
plt.axis([-100, 400, -100, 550])
plt.show()
"""

all_data_input_paderno = np.array(all_data_input_paderno)
index = 0
for input in all_data_input:
    all_data_input[index] = np.array(input)
    index += 1
all_data_output = np.array(all_data_output)

# LINEAR MODEL

lr = 0.001
loss = tf.keras.losses.MeanSquaredError()


def coeff_determination(y_true, y_pred):
    ss_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    ss_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())


# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
rain_input = Input(shape=(1,), name='rain_input')

paderno_input = Input(shape=(1,), name='paderno_input')

x = concatenate([rain_input, paderno_input])

# We stack a deep densely-connected network on top
x = Dense(8, activation='elu')(x)
x = Dense(8, activation='elu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='linear', name='main_output')(x)

optimizer = Adam(learning_rate=lr, amsgrad=False)

model = Model(inputs=[rain_input, paderno_input], outputs=main_output)

model.compile(optimizer, loss=loss, metrics=[coeff_determination], loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
              target_tensors=None)

model.summary()


model.fit(x=[all_data_input[20], all_data_input_paderno], y=all_data_output, batch_size=None, epochs=40, verbose=2, callbacks=None,
          validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
          initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
          use_multiprocessing=False)


# RECURSION MODEL

lr = 0.0001
loss = tf.keras.losses.MeanSquaredError()
"""
model_RNN = Sequential([
    Dense(16, input_shape=(1,)),
    Activation('elu'),
    Reshape((1, 16)),
    SimpleRNN(64),
    Activation('elu'),
    Dense(1)
])
"""

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
rain_input = Input(shape=(1,), name='rain_input')

paderno_input = Input(shape=(1,), name='paderno_input')

x = concatenate([rain_input, paderno_input])

x = Dense(16, activation='elu')(x)
x = Reshape((1, 16))(x)
x = LSTM(16, activation='elu')(x)
x = Reshape((1, 16))(x)
x = LSTM(16, activation='elu')(x)

main_output = Dense(1, activation='linear', name='main_output')(x)

optimizer = Adam(learning_rate=lr, amsgrad=False)

model_RNN = Model(inputs=[rain_input, paderno_input], outputs=main_output)

model_RNN.compile(optimizer, loss=loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
              target_tensors=None)

model_RNN.summary()

model_RNN.fit(x=[all_data_input[3], all_data_input_paderno], y=all_data_output, batch_size=None, epochs=30, verbose=2, callbacks=None,
          validation_split=0.2, validation_data=None, shuffle=True, class_weight=None, sample_weight=None,
          initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1, max_queue_size=10, workers=1,
          use_multiprocessing=False)



