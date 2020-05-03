from libraries_and_data import library_datas
from models import lstm_model
import random
import time
import numpy as np

random.seed(a=1234, version=2)

# LN = LEVEL NIGUARDA
# LP = LEVEL PADERNO
# RP = RAIN PLUVIOMETERS
# LC = LEVEL CANTU

# SAVE HISTORY FILES
named_tuple = time.localtime()  # get struct_time
name_file = time.strftime("%m_%d_%Y_%H_%M_%S" + "_best_step", named_tuple)

index_pluviometer = 17

n_steps = 3

shuffle = True

# TRAIN DATA

input_rain = library_datas.get_data_by_years_and_type("RP", 2015, 2017)
index_rain = 0
input_rain_meno_1 = []

for single_input in input_rain:
    input_rain_meno_1.append([])
    input_rain_meno_1[index_rain] = [input_rain[index_rain][0]] + input_rain[index_rain][
                                                                  :len(input_rain[index_rain]) - 1]
    index_rain += 1

input_paderno = library_datas.get_data_by_years_and_type("LP", 2015, 2017)
input_paderno_meno_1 = [input_paderno[0]] + input_paderno[:len(input_paderno) - 1]
input_niguarda = library_datas.get_data_by_years_and_type("LN", 2015, 2017)
input_niguarda_meno_1 = [input_niguarda[0]] + input_niguarda[:len(input_niguarda) - 1]
input_cantu = library_datas.get_data_by_years_and_type("LC", 2015, 2017)
input_cantu_meno_1 = [input_cantu[0]] + input_cantu[:len(input_cantu) - 1]
output_niguarda = input_niguarda

input = [input_rain, input_rain_meno_1, input_paderno, input_paderno_meno_1, input_niguarda, input_niguarda_meno_1,
         input_cantu, input_cantu_meno_1]

input, output = library_datas.apply_step_diff(input, [output_niguarda])

input_rain = input[0]
input_rain_meno_1 = input[1]
input_paderno = input[2]
input_paderno_meno_1 = input[3]
input_niguarda = input[4]
input_niguarda_meno_1 = input[5]
input_cantu = input[6]
input_cantu_meno_1 = input[7]
output_niguarda = output[0]

# VALIDATION DATA

input_rain_val = library_datas.get_data_by_years_and_type("RP", 2018, 2018)
index_rain = 0
input_rain_meno_1_val = []

for single_input in input_rain_val:
    input_rain_meno_1_val.append([])
    input_rain_meno_1_val[index_rain] = [input_rain_val[index_rain][0]] + input_rain_val[index_rain][
                                                                          :len(input_rain_val[index_rain]) - 1]
    index_rain += 1

input_paderno_val = library_datas.get_data_by_years_and_type("LP", 2018, 2018)
input_paderno_meno_1_val = [input_paderno_val[0]] + input_paderno_val[:len(input_paderno_val) - 1]
input_niguarda_val = library_datas.get_data_by_years_and_type("LN", 2018, 2018)
input_niguarda_meno_1_val = [input_niguarda_val[0]] + input_niguarda_val[:len(input_niguarda_val) - 1]
input_cantu_val = library_datas.get_data_by_years_and_type("LC", 2018, 2018)
input_cantu_meno_1_val = [input_cantu_val[0]] + input_cantu_val[:len(input_cantu_val) - 1]
output_niguarda_val = input_niguarda_val

input_val = [input_rain_val, input_rain_meno_1_val, input_paderno_val, input_paderno_meno_1_val, input_niguarda_val,
             input_niguarda_meno_1_val, input_cantu_val, input_cantu_meno_1_val]

input_val, output_val = library_datas.apply_step_diff(input_val, [output_niguarda_val])

input_rain_val = input_val[0]
input_rain_meno_1_val = input_val[1]
input_paderno_val = input_val[2]
input_paderno_meno_1_val = input_val[3]
input_niguarda_val = input_val[4]
input_niguarda_meno_1_val = input_val[5]
input_cantu_val = input_val[6]
input_cantu_meno_1_val = input_val[7]
output_niguarda_val = output_val[0]

# CLEAN_DATA

input_rain, input_rain_val = library_datas.clean_data_with_val(input_rain, input_rain_val)
input_rain_meno_1, input_rain_meno_1_val = library_datas.clean_data_with_val(input_rain_meno_1, input_rain_meno_1_val)
input_paderno, input_paderno_val = library_datas.clean_data_with_val(input_paderno, input_paderno_val)
input_paderno_meno_1, input_paderno_meno_1_val = library_datas.clean_data_with_val(input_paderno_meno_1, input_paderno_meno_1_val)
input_niguarda, input_niguarda_val = library_datas.clean_data_with_val(input_niguarda, input_niguarda_val)
input_niguarda_meno_1, input_niguarda_meno_1_val = library_datas.clean_data_with_val(input_niguarda_meno_1, input_niguarda_meno_1_val)
input_cantu, input_cantu_val = library_datas.clean_data_with_val(input_cantu, input_cantu_val)
input_cantu_meno_1, input_cantu_meno_1_val = library_datas.clean_data_with_val(input_cantu_meno_1, input_cantu_meno_1_val)
output_niguarda, output_niguarda_val = library_datas.clean_data_with_val(output_niguarda, output_niguarda_val)

# LSTM TRAIN DATA

input_rain            = np.array(input_rain[index_pluviometer])
input_paderno         = np.array(input_paderno)
input_niguarda        = np.array(input_niguarda)
input_niguarda_meno_1 = np.array(input_niguarda_meno_1)
input_cantu           = np.array(input_cantu)

# convert to [rows, columns] structure
input_rain            = input_rain.reshape((len(input_rain), 1))
input_paderno         = input_paderno.reshape((len(input_paderno), 1))
input_niguarda        = input_niguarda.reshape((len(input_niguarda), 1))
input_niguarda_meno_1 = input_niguarda_meno_1.reshape((len(input_niguarda_meno_1), 1))
input_cantu           = input_cantu.reshape((len(input_cantu), 1))

# horizontally stack columns
dataset = np.hstack((input_rain, input_paderno, input_niguarda, input_niguarda_meno_1, input_cantu))

"""
train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(dataset, output_niguarda, n_steps, sampling_rate=1,
                      stride=1, start_index=0, end_index=None, shuffle=True, reverse=False, batch_size=32)
"""
final_dataset = []

for index in range(len(dataset) - (n_steps - 1)):
    final_dataset.append([dataset[index], dataset[index+1], dataset[index+2]])

# LSTM VALIDATION DATA

input_rain_val            = np.array(input_rain_val[index_pluviometer])
input_paderno_val         = np.array(input_paderno_val)
input_niguarda_val        = np.array(input_niguarda_val)
input_niguarda_meno_1_val = np.array(input_niguarda_meno_1_val)
input_cantu_val           = np.array(input_cantu_val)

# convert to [rows, columns] structure
input_rain_val            = input_rain_val.reshape((len(input_rain_val), 1))
input_paderno_val         = input_paderno_val.reshape((len(input_paderno_val), 1))
input_niguarda_val        = input_niguarda_val.reshape((len(input_niguarda_val), 1))
input_niguarda_meno_1_val = input_niguarda_meno_1_val.reshape((len(input_niguarda_meno_1_val), 1))
input_cantu_val           = input_cantu_val.reshape((len(input_cantu_val), 1))

# horizontally stack columns
dataset_val = np.hstack((input_rain_val, input_paderno_val, input_niguarda_val, input_niguarda_meno_1_val, input_cantu_val))

final_dataset_val = []

for index in range(len(dataset_val) - (n_steps - 1)):
    final_dataset_val.append([dataset_val[index], dataset_val[index+1], dataset_val[index+2]])

"""
validation_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(dataset_val, output_niguarda_val, n_steps,
                 sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=32)
"""

output_niguarda     = output_niguarda[n_steps-1:]
output_niguarda_val = output_niguarda_val[n_steps-1:]

if shuffle:
    # RANDOMIZE DATA TRAIN

    data = list(zip(final_dataset, output_niguarda))
    random.shuffle(data)
    final_dataset, output_niguarda = zip(*data)

    final_dataset         = list(final_dataset)
    output_niguarda       = list(output_niguarda)

    # RANDOMIZE DATA VALIDATION

    data_val = list(zip(final_dataset_val, output_niguarda_val))
    random.shuffle(data_val)
    final_dataset_val, output_niguarda_val = zip(*data_val)

    final_dataset_val         = list(final_dataset_val)
    output_niguarda_val       = list(output_niguarda_val)

# MODEL

model = lstm_model.LinearModel(n_steps=n_steps, n_features=5)

final_dataset = np.array(final_dataset)
final_dataset_val = np.array(final_dataset_val)

history = model.fit(x=final_dataset, y=output_niguarda, validation_data=(final_dataset_val, output_niguarda_val), epochs=100, verbose=2)

"""
history = model.fit_generator(generator=train_generator, validation_generator=validation_generator, epochs=100, verbose=2)


predictions = model.predict_generator(generator=validation_generator)

# calculate mae
sum = 0
count = 0
for prediction in predictions:
    sum = sum + (prediction - output_niguarda_val[2 + count])**2
    count = count + 1

print("mse = " + str(sum/count))

library_datas.plot_graph_predictions_vs_reality(predictions, output_niguarda_val[3:])
"""

predictions = model.predict(x=final_dataset_val)

library_datas.plot_graph_predictions_vs_reality(predictions, output_niguarda_val)
