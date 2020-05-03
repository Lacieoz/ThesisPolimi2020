from libraries_and_data import library_datas
from models import linear_model_best_pluviometer
import random
import time

random.seed(a=1234, version=2)

# LN = LEVEL NIGUARDA
# LP = LEVEL PADERNO
# RP = RAIN PLUVIOMETERS

# SAVE HISTORY FILES
named_tuple = time.localtime() # get struct_time
name_file = time.strftime("%m_%d_%Y_%H_%M_%S" + "_best_pluv", named_tuple)

name_file = "04_26_2020_13_41_55_best_pluv"

# TRAIN DATA

input_rain      = library_datas.get_data_by_years_and_type("RP", 2015, 2017)
input_paderno   = library_datas.get_data_by_years_and_type("LP", 2015, 2017)
input_niguarda  = library_datas.get_data_by_years_and_type("LN", 2015, 2017)
output_niguarda = input_niguarda

input = [input_paderno, input_niguarda]

input, output = library_datas.apply_step_diff(input, [output_niguarda])

input_paderno   = input[0]
input_niguarda  = input[1]
output_niguarda = output[0]

# VALIDATION DATA

input_rain_val      = library_datas.get_data_by_years_and_type("RP", 2018, 2018)
input_paderno_val   = library_datas.get_data_by_years_and_type("LP", 2018, 2018)
input_niguarda_val  = library_datas.get_data_by_years_and_type("LN", 2018, 2018)
output_niguarda_val = input_niguarda_val

input_val = [input_paderno_val, input_niguarda_val]

input_val, output_val = library_datas.apply_step_diff(input_val, [output_niguarda_val])

input_paderno_val   = input_val[0]
input_niguarda_val  = input_val[1]
output_niguarda_val = output_val[0]

# CLEAN_DATA

input_rain, input_rain_val           = library_datas.clean_data_with_val(input_rain, input_rain_val)
input_paderno, input_paderno_val     = library_datas.clean_data_with_val(input_paderno, input_paderno_val)
input_niguarda, input_niguarda_val   = library_datas.clean_data_with_val(input_niguarda, input_niguarda_val)
output_niguarda, output_niguarda_val = library_datas.clean_data_with_val(output_niguarda, output_niguarda_val)

best_results = []

for index_pluviometer in range(len(input_rain)):

    index_pluviometer = index_pluviometer + 4

    results = []

    for placeholder in range(19):

        custom_timestep = placeholder * 10

        if index_pluviometer == 4:
            placeholder = placeholder + 100
            if placeholder > 190:
                continue

        # APPLY TIME_STEP
        input_rain_try          = library_datas.apply_step_diff_input(input_rain[index_pluviometer], custom_timestep)
        input_rain_val_try      = library_datas.apply_step_diff_input(input_rain_val[index_pluviometer], custom_timestep)
        input_paderno_try       = library_datas.additional_step_diff(input_paderno, custom_timestep)
        input_paderno_val_try   = library_datas.additional_step_diff(input_paderno_val, custom_timestep)
        input_niguarda_try      = library_datas.additional_step_diff(input_niguarda, custom_timestep)
        input_niguarda_val_try  = library_datas.additional_step_diff(input_niguarda_val, custom_timestep)
        output_niguarda_try     = library_datas.additional_step_diff(output_niguarda, custom_timestep)
        output_niguarda_val_try = library_datas.additional_step_diff(output_niguarda_val, custom_timestep)

        # SHUFFLE DATA

        data_val_try = list(zip(input_rain_val_try, input_paderno_val_try, input_niguarda_val_try, output_niguarda_val_try))
        random.shuffle(data_val_try)
        input_rain_val_try, input_paderno_val_try, input_niguarda_val_try, output_niguarda_val_try = zip(*data_val_try)

        input_rain_val_try      = list(input_rain_val_try)
        input_paderno_val_try   = list(input_paderno_val_try)
        input_niguarda_val_try  = list(input_niguarda_val_try)
        output_niguarda_val_try = list(output_niguarda_val_try)

        data_try = list(zip(input_rain_try, input_paderno_try, input_niguarda_try, output_niguarda_try))
        random.shuffle(data_try)
        input_rain_try, input_paderno_try, input_niguarda_try, output_niguarda_try = zip(*data_try)

        input_rain_try      = list(input_rain_try)
        input_paderno_try   = list(input_paderno_try)
        input_niguarda_try  = list(input_niguarda_try)
        output_niguarda_try = list(output_niguarda_try)

        # MODELS

        model = linear_model_best_pluviometer.LinearModel()

        history = model.fit(x=[input_rain_try, input_paderno_try, input_niguarda_try], y=output_niguarda_try, epochs=100, verbose=2, validation_split=None,
                  validation_data=([input_rain_val_try, input_paderno_val_try, input_niguarda_val_try], output_niguarda_val_try))

        results.append(min(history.history["val_loss"]))
        f = open("./logs/best_pluv/" + name_file + ".txt", "a")
        f.write("PLUVIOMETER " + str(index_pluviometer) + " - TIMESTEP " + str(custom_timestep) + "\n")
        f.write("   RESULT = " + str(min(history.history["val_loss"])) + "\n")
        f.close()

    best_result = min(results)
    index_best_result = results.index(best_result)

    f = open("./logs/best_pluv/" + name_file + ".txt", "a")
    f.write("PLUVIOMETER " + str(index_pluviometer) + " - BEST RESULT \n")
    f.write("   BEST RESULT = " + str(best_result) + " - AT TIMESTEP : " + str(index_best_result * 10) + "\n")
    f.close()

    best_results.append(best_result)

bestest_result = min(best_results)
index_best_result = best_results.index(bestest_result)

f = open("./logs/best_pluv/" + name_file + ".txt", "a")
f.write("ALL PLUVIOMETERS - BEST RESULT \n")
f.write("   BEST RESULT EVER = " + str(bestest_result) + " - OF PLUVIOMETER : " + str(index_best_result) + "\n")
f.close()



