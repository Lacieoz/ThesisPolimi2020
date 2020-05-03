from os import walk
import csv
import time
import matplotlib.pyplot as plt
import numpy as np


# -------------------- GLOBAL VARIABLES -------------------- #

path_by_types = {"LN": "./libraries_and_data/data/MilanoNiguarda", "LP": "./libraries_and_data/data/PadernoDugnano",
                 "RP": "./libraries_and_data/data/Pluviometri", "LC": "./libraries_and_data/data/Cantu"}
# LN = LEVEL NIGUARDA
# LP = LEVEL PADERNO
# RP = RAIN PLUVIOMETERS

minutes_diff = 40

step_diff = int(minutes_diff / 10)

# -------------------- EXTRACT FROM FILE -------------------- #


def get_csv_files_in_folder(path):

    files = []

    for (dirpath, dirnames, filenames) in walk(path):
        for file in filenames:
            if file[-4:] == ".csv":
                files.append({"dirpath": dirpath, "filename": file})

    return files


def extract_data_from_file_rain(filedata, start_year, end_year):
    dir = filedata["dirpath"]
    filename = filedata["filename"]

    data = []
    data_for_mean = []

    for column in range(21):
        data.append([])

    with open(dir + "/" + filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        first_line = True
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif first_line:

                year = int(row[1][0:4])
                if start_year <= year <= end_year:

                    for column in range(21):
                        try:
                            data[column].append(float(row[column + 2]))
                        except:
                            data[column].append(0.0)

                    first_line = False
                    line_count += 1
            else:
                year = int(row[1][0:4])
                if start_year <= year <= end_year:
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

        # print(f'Processed {line_count} lines.')

    return data


def extract_data_from_file_idrometer(filedata, start_year, end_year):
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
                if start_year <= year <= end_year:
                    data.append(float(row[2]))
                line_count += 1

        # print(f'Processed {line_count} lines.')

    return data


def get_data_by_years_and_type(type, start_year, end_year):

    start_time_to_extract = time.time()

    data = []
    path = path_by_types[type]

    files = get_csv_files_in_folder(path)

    if type == "LN":
        for file in files:
            data = data + extract_data_from_file_idrometer(file, start_year, end_year)
    elif type == "LP":
        for file in files:
            data = data + extract_data_from_file_idrometer(file, start_year, end_year)
    elif type == "RP":
        for file in files:
            data = data + extract_data_from_file_rain(file, start_year, end_year)
    elif type == "LC":
        for file in files:
            data = data + extract_data_from_file_idrometer(file, start_year, end_year)

    print("--- To extract data %s" % type + " = %s seconds ---" % (time.time() - start_time_to_extract))

    return data


def extract_data_from_file_idrometer_with_date(filedata, start_year, end_year):
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
                if start_year <= year <= end_year:
                    data.append(row[1])
                line_count += 1

        # print(f'Processed {line_count} lines.')

    return data


def get_output_dates_graph_apply_step(start_year, end_year):
    start_time_to_extract = time.time()

    data = []
    path = path_by_types["LN"]

    files = get_csv_files_in_folder(path)

    for file in files:
        data = data + extract_data_from_file_idrometer_with_date(file, start_year, end_year)

    # APPLY TIMESTEP
    data = data[step_diff:]

    print("--- To extract data %s" % type + " = %s seconds ---" % (time.time() - start_time_to_extract))

    return data


# -------------------- APPLY STEP DIFFERENCE -------------------- #

def dimensions_of_list(data):
    if not type(data) == list:
        return 0
    return 1 + dimensions_of_list(data[0])


def apply_step_diff(data_input, data_output):

    data_input_result = []
    data_output_result = []

    length = len(data_output[0])

    index = 0
    for input in data_input:

        if dimensions_of_list(input) == 1:
            data_input_result.append(input[:length-step_diff])
            index += 1

        elif dimensions_of_list(input) == 2:
            data_input_result.append([])
            index_result = len(data_input_result)
            inner_index = 0
            for inner_input in input:
                data_input_result[index_result - 1].append(inner_input[:length - step_diff])
                inner_index += 1
            index += 1

        else:
            print("ERROR!! Impossible to apply the step difference to some input")

    index = 0
    for output in data_output:
        if dimensions_of_list(output) == 1:
            data_output_result.append(output[step_diff:])
            index += 1

        elif dimensions_of_list(output) == 2:
            data_output_result.append([])
            index_result = len(data_input_result)
            inner_index = 0
            for inner_output in output:
                data_output_result[index_result - 1].append(inner_output[step_diff:])
                inner_index += 1
            index += 1

        else:
            print("ERROR!! Impossible to apply the step difference to some output")

    return data_input_result, data_output_result


def apply_step_diff_input(data_input, custom_timestep):

    data_input_result = []

    if custom_timestep >= minutes_diff:

        custom_step_diff = int(custom_timestep / 10)

        if dimensions_of_list(data_input) == 1:
            length = len(data_input)
            data_input_result = data_input[:length - custom_step_diff]

        elif dimensions_of_list(data_input) == 2:
            length = len(data_input[0])
            inner_index = 0
            for input in data_input:
                data_input_result.append([])
                data_input_result[inner_index] = input[:length - custom_step_diff]
                inner_index += 1

        else:
            print("ERROR!! Impossible to apply the step difference to some input")

    elif custom_timestep < minutes_diff:

        custom_step_diff_start = int((minutes_diff - custom_timestep) / 10)
        custom_step_diff_end   = int(custom_timestep / 10)

        if dimensions_of_list(data_input) == 1:
            length = len(data_input)
            data_input_result = data_input[custom_step_diff_start:length - custom_step_diff_end]

        elif dimensions_of_list(data_input) == 2:
            length = len(data_input[0])
            inner_index = 0
            for input in data_input:
                data_input_result.append([])
                data_input_result[inner_index] = input[custom_step_diff_start:length - custom_step_diff_end]
                inner_index += 1

        else:
            print("ERROR!! Impossible to apply the step difference to some input")

    return data_input_result


def additional_step_diff(data, custom_timestep):

    data_result = []

    if custom_timestep > minutes_diff:
        custom_step_diff_start = int((custom_timestep - minutes_diff) / 10)

        if dimensions_of_list(data) == 1:
            data_result = data[custom_step_diff_start:]

        elif dimensions_of_list(data) == 2:
            inner_index = 0
            for inner in data:
                data_result.append([])
                data_result[inner_index] = inner[custom_step_diff_start:]
                inner_index += 1

        else:
            print("ERROR!! Impossible to apply the step difference to some input")

    return data


# -------------------- CLEAN MISSING VALUE AND MAKE NUMBERS POSITIVE -------------------- #

def clean_data(data):
    dimensions = dimensions_of_list(data)

    counter = 0
    minimo = 1000

    if dimensions == 1:

        for index in range(len(data)):
            if data[index] == -999:
                if data[index + 1] != -999:
                    data[index] = (data[index - 1] + data[index + 1]) / 2
                else:
                    data[index] = data[index - 1]
                counter += 1
            else:
                if data[index] < minimo :
                    minimo = data[index]

        if minimo < 0.0:
            minimo += 0.5

            for index in range(len(data)):
                data[index] += minimo
    else:
        for index in range(len(data)):
            minimo = 0
            counter = 0
            for inner_index in range(len(data[index])):
                if data[index][inner_index] == -999:
                    if data[index][inner_index + 1] != -999:
                        data[index][inner_index] = (data[index][inner_index - 1] + data[index][inner_index + 1]) / 2
                    else:
                        data[index][inner_index] = data[index][inner_index - 1]
                    counter += 1
                else:
                    if data[index][inner_index] < minimo:
                        minimo = data[index][inner_index]

            if minimo < 0.0:
                minimo += 0.5

                for inner_index in range(len(data[index])):
                    data[index][inner_index] += minimo

    return data


def clean_data_with_val(data, data_val):
    dimensions = dimensions_of_list(data)

    minimo = 1000
    counter = 0
    counter_val = 0

    if dimensions == 1:

        for index in range(len(data)):
            if data[index] == -999:
                if data[index + 1] != -999:
                    data[index] = (data[index - 1] + data[index + 1]) / 2
                else:
                    data[index] = data[index - 1]
                counter += 1
            else:
                if data[index] < minimo:
                    minimo = data[index]

        for index in range(len(data_val)):
            if data_val[index] == -999:
                if data_val[index + 1] != -999:
                    data_val[index] = (data_val[index - 1] + data_val[index + 1]) / 2
                else:
                    data_val[index] = data_val[index - 1]
                counter_val += 1
            else:
                if data_val[index] < minimo:
                    minimo = data_val[index]

        if minimo < 0.0:
            minimo *= -1
            minimo += 0.5

            for index in range(len(data)):
                data[index] += minimo
            for index in range(len(data_val)):
                data_val[index] += minimo
    else:
        for index in range(len(data)):
            minimo = 1000
            counter = 0
            counter_val = 0
            for inner_index in range(len(data[index])):
                if data[index][inner_index] == -999:
                    if data[index][inner_index + 1] != -999:
                        data[index][inner_index] = (data[index][inner_index - 1] + data[index][inner_index + 1]) / 2
                    else:
                        data[index][inner_index] = data[index][inner_index - 1]
                    counter += 1
                else:
                    if data[index][inner_index] < minimo:
                        minimo = data[index][inner_index]

            for inner_index in range(len(data_val[index])):
                if data_val[index][inner_index] == -999:
                    if data_val[index][inner_index + 1] != -999:
                        data_val[index][inner_index] = (data_val[index][inner_index - 1] + data_val[index][inner_index + 1]) / 2
                    else:
                        data_val[index][inner_index] = data_val[index][inner_index - 1]
                    counter_val += 1
                else:
                    if data_val[index][inner_index] < minimo:
                        minimo = data_val[index][inner_index]

            if minimo < 0.0:
                minimo *= -1
                minimo += 0.5

                for inner_index in range(len(data[index])):
                    data[index][inner_index] += minimo
                for inner_index in range(len(data_val[index])):
                    data_val[index][inner_index] += minimo

    return data, data_val


def set_timestep(timestep):
    global minutes_diff
    minutes_diff = timestep

    global step_diff
    step_diff = int(minutes_diff / 10)


def compute_list_delta(list_delta):
    delta_list = []
    placeholder_diff = list_delta[0]
    delta_list.append(list_delta[0] - placeholder_diff)

    for index in range(len(list_delta)-1):
        placeholder_mem = list_delta[index + 1]
        delta_list.append(list_delta[index + 1] - placeholder_diff)
        placeholder_diff = placeholder_mem

    return list_delta


# PLOT GRAPHS
def plot_graph_validation(predictions, true_values, dates_val):
    # Plot things...
    max_length = 500

    for i in range(int(len(true_values) / max_length)):

        index_start = i * max_length
        index_end   = (i+1) * max_length

        fig = plt.figure()

        plt.plot_date(dates_val[index_start:index_end], true_values[index_start:index_end], 'b-')
        plt.plot_date(dates_val[index_start:index_end], predictions[index_start:index_end], 'r-')

        fig.autofmt_xdate()
        plt.show()


def plot_graph_predictions_vs_reality(predictions, true_values):

    plt.plot(predictions, true_values, 'bo')
    plt.xlabel('predictions', color='#1C2833')
    plt.ylabel('observations', color='#1C2833')

    # to plot line y = x
    x = np.linspace(0, 300, 1000)
    plt.plot(x, x + 0, '-b',)

    plt.axis([0, 300, 0, 300])
    plt.show()







