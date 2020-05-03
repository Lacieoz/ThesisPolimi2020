from libraries_and_data import library_datas
from models import linear_model_no_cantu
import random
import time

random.seed(a=1234, version=2)

# LN = LEVEL NIGUARDA
# LP = LEVEL PADERNO
# RP = RAIN PLUVIOMETERS
# LC = LEVEL CANTU

# SAVE HISTORY FILES
named_tuple = time.localtime() # get struct_time
name_file = time.strftime("%m_%d_%Y_%H_%M_%S" + "_best_step", named_tuple)


index_pluviometer = 17

# TRAIN DATA

input_rain            = library_datas.get_data_by_years_and_type("RP", 2015, 2017)
input_paderno         = library_datas.get_data_by_years_and_type("LP", 2015, 2017)
input_niguarda        = library_datas.get_data_by_years_and_type("LN", 2015, 2017)
input_niguarda_meno_1 = [input_niguarda[0]] + input_niguarda[:len(input_niguarda)-1]
input_cantu           = library_datas.get_data_by_years_and_type("LC", 2015, 2017)
output_niguarda       = input_niguarda

input = [input_rain, input_paderno, input_niguarda, input_niguarda_meno_1]

input, output = library_datas.apply_step_diff(input, [output_niguarda])

input_rain             = input[0]
input_paderno          = input[1]
input_niguarda         = input[2]
input_niguarda_meno_1  = input[3]
output_niguarda        = output[0]

# VALIDATION DATA

input_rain_val            = library_datas.get_data_by_years_and_type("RP", 2018, 2018)
input_paderno_val         = library_datas.get_data_by_years_and_type("LP", 2018, 2018)
input_niguarda_val        = library_datas.get_data_by_years_and_type("LN", 2018, 2018)
input_niguarda_meno_1_val = [input_niguarda_val[0]] + input_niguarda_val[:len(input_niguarda_val)-1]
input_cantu_val           = library_datas.get_data_by_years_and_type("LC", 2018, 2018)
output_niguarda_val       = input_niguarda_val

input_val = [input_rain_val, input_paderno_val, input_niguarda_val, input_niguarda_meno_1_val]

input_val, output_val = library_datas.apply_step_diff(input_val, [output_niguarda_val])

input_rain_val             = input_val[0]
input_paderno_val          = input_val[1]
input_niguarda_val         = input_val[2]
input_niguarda_meno_1_val  = input_val[3]
output_niguarda_val        = output_val[0]

# CLEAN_DATA

input_rain, input_rain_val                         = library_datas.clean_data_with_val(input_rain, input_rain_val)
input_paderno, input_paderno_val                   = library_datas.clean_data_with_val(input_paderno, input_paderno_val)
input_niguarda, input_niguarda_val                 = library_datas.clean_data_with_val(input_niguarda, input_niguarda_val)
input_niguarda_meno_1, input_niguarda_meno_1_val   = library_datas.clean_data_with_val(input_niguarda_meno_1, input_niguarda_meno_1_val)
input_cantu, input_cantu_val                       = library_datas.clean_data_with_val(input_cantu, input_cantu_val)
output_niguarda, output_niguarda_val               = library_datas.clean_data_with_val(output_niguarda, output_niguarda_val)



# RANDOMIZE DATA

data_val_try = list(zip(input_rain_val[index_pluviometer], input_paderno_val, input_niguarda_val,
                        input_niguarda_meno_1_val, output_niguarda_val))
random.shuffle(data_val_try)
input_rain_val_try, input_paderno_val_try, input_niguarda_val_try, input_niguarda_meno_1_val_try, \
                        output_niguarda_val_try = zip(*data_val_try)

input_rain_val_try            = list(input_rain_val_try)
input_paderno_val_try         = list(input_paderno_val_try)
input_niguarda_val_try        = list(input_niguarda_val_try)
input_niguarda_meno_1_val_try = list(input_niguarda_meno_1_val_try)
output_niguarda_val_try       = list(output_niguarda_val_try)

data_try = list(zip(input_rain[index_pluviometer], input_paderno, input_niguarda, input_niguarda_meno_1, output_niguarda))
random.shuffle(data_try)
input_rain_try, input_paderno_try, input_niguarda_try, input_niguarda_meno_1_try, output_niguarda_try = zip(*data_try)

input_rain_try            = list(input_rain_try)
input_paderno_try         = list(input_paderno_try)
input_niguarda_try        = list(input_niguarda_try)
input_niguarda_meno_1_try = list(input_niguarda_meno_1_try)
output_niguarda_try       = list(output_niguarda_try)

# MODELS

model = linear_model_no_cantu.LinearModel()

history = model.fit(x=[input_rain_try, input_paderno_try, input_niguarda_try, input_niguarda_meno_1_try],
                    y=output_niguarda_try, epochs=100, verbose=2, validation_split=None,
                    validation_data=([input_rain_val_try, input_paderno_val_try, input_niguarda_val_try,
                    input_niguarda_meno_1_val_try], output_niguarda_val_try))

predictions = model.predict(x=[input_rain_val_try, input_paderno_val_try, input_niguarda_val_try,
                               input_niguarda_meno_1_val_try])

library_datas.plot_graph_predictions_vs_reality(predictions, output_niguarda_val_try)

