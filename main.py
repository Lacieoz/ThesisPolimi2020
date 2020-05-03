from libraries_and_data import library_datas
from models import linear_model, linear_model_better_loss, linear_model_no_rain, linear_model_2_pluviometers
import random

random.seed(a=1234, version=2)

# LN = LEVEL NIGUARDA
# LP = LEVEL PADERNO
# RP = RAIN PLUVIOMETERS

index_pluviometer = 17
index_pluviometer_2 = 7

# TRAIN DATA

input_rain      = library_datas.get_data_by_years_and_type("RP", 2015, 2017)
input_paderno   = library_datas.get_data_by_years_and_type("LP", 2015, 2017)
input_niguarda  = library_datas.get_data_by_years_and_type("LN", 2015, 2017)
output_niguarda = input_niguarda

input = [input_rain, input_paderno, input_niguarda]

input, output = library_datas.apply_step_diff(input, [output_niguarda])

input_rain      = input[0]
input_paderno   = input[1]
input_niguarda  = input[2]
output_niguarda = output[0]

# VALIDATION DATA

input_rain_val      = library_datas.get_data_by_years_and_type("RP", 2018, 2018)
input_paderno_val   = library_datas.get_data_by_years_and_type("LP", 2018, 2018)
input_niguarda_val  = library_datas.get_data_by_years_and_type("LN", 2018, 2018)
output_niguarda_val = input_niguarda_val

input_val = [input_rain_val, input_paderno_val, input_niguarda_val]

input_val, output_val = library_datas.apply_step_diff(input_val, [output_niguarda_val])

input_rain_val      = input_val[0]
input_paderno_val   = input_val[1]
input_niguarda_val  = input_val[2]
output_niguarda_val = output_val[0]

# CLEAN_DATA

input_rain, input_rain_val           = library_datas.clean_data_with_val(input_rain, input_rain_val)
input_paderno, input_paderno_val     = library_datas.clean_data_with_val(input_paderno, input_paderno_val)
input_niguarda, input_niguarda_val   = library_datas.clean_data_with_val(input_niguarda, input_niguarda_val)
output_niguarda, output_niguarda_val = library_datas.clean_data_with_val(output_niguarda, output_niguarda_val)

# RANDOMIZE DATA

data_val = list(zip(input_rain_val[index_pluviometer], input_rain_val[index_pluviometer_2], input_paderno_val, input_niguarda_val, output_niguarda_val))
random.shuffle(data_val)
input_rain_val[index_pluviometer], input_rain_val[index_pluviometer_2], input_paderno_val, input_niguarda_val, output_niguarda_val = zip(*data_val)

input_rain_val[index_pluviometer]   = list(input_rain_val[index_pluviometer])
input_rain_val[index_pluviometer_2] = list(input_rain_val[index_pluviometer_2])
input_paderno_val                   = list(input_paderno_val)
input_niguarda_val                  = list(input_niguarda_val)
output_niguarda_val                 = list(output_niguarda_val)

data = list(zip(input_rain[index_pluviometer], input_rain[index_pluviometer_2], input_paderno, input_niguarda, output_niguarda))
random.shuffle(data)
input_rain[index_pluviometer], input_rain[index_pluviometer_2], input_paderno, input_niguarda, output_niguarda = zip(*data)

input_rain[index_pluviometer]     = list(input_rain[index_pluviometer])
input_rain[index_pluviometer_2]   = list(input_rain[index_pluviometer_2])
input_paderno                     = list(input_paderno)
input_niguarda                    = list(input_niguarda)
output_niguarda                   = list(output_niguarda)

# MODELS
"""
model_no_rain = linear_model_no_rain.LinearModel()

model_no_rain.fit(x=[input_paderno, input_niguarda], y=output_niguarda, epochs=40, verbose=2, validation_split=None,
          validation_data=([input_paderno_val, input_niguarda_val], output_niguarda_val))


model = linear_model_better_loss.LinearModel()

model.fit(x=[input_rain[index_pluviometer], input_paderno, input_niguarda], y=output_niguarda, epochs=100, verbose=2, validation_split=None,
          validation_data=([input_rain_val[index_pluviometer], input_paderno_val, input_niguarda_val], output_niguarda_val))
"""
model = linear_model.LinearModel()

model.fit(x=[input_rain[index_pluviometer], input_paderno, input_niguarda], y=output_niguarda, epochs=100, verbose=2, validation_split=None,
          validation_data=([input_rain_val[index_pluviometer], input_paderno_val, input_niguarda_val], output_niguarda_val))
"""
model = linear_model_2_pluviometers.LinearModel()

model.fit(x=[input_rain[index_pluviometer], input_rain[index_pluviometer_2], input_paderno, input_niguarda], y=output_niguarda, epochs=100, verbose=2, validation_split=None,
          validation_data=([input_rain_val[index_pluviometer], input_rain_val[index_pluviometer_2], input_paderno_val, input_niguarda_val], output_niguarda_val))
"""
# TRAIN DATA

input_rain      = library_datas.get_data_by_years_and_type("RP", 2015, 2017)
input_paderno   = library_datas.get_data_by_years_and_type("LP", 2015, 2017)
input_niguarda  = library_datas.get_data_by_years_and_type("LN", 2015, 2017)
output_niguarda = input_niguarda

input = [input_rain, input_paderno, input_niguarda]

input, output = library_datas.apply_step_diff(input, [output_niguarda])

input_rain      = input[0]
input_paderno   = input[1]
input_niguarda  = input[2]
output_niguarda = output[0]

# VALIDATION DATA

input_rain_val      = library_datas.get_data_by_years_and_type("RP", 2018, 2018)
input_paderno_val   = library_datas.get_data_by_years_and_type("LP", 2018, 2018)
input_niguarda_val  = library_datas.get_data_by_years_and_type("LN", 2018, 2018)
output_niguarda_val = input_niguarda_val

input_val = [input_rain_val, input_paderno_val, input_niguarda_val]

input_val, output_val = library_datas.apply_step_diff(input_val, [output_niguarda_val])

input_rain_val      = input_val[0]
input_paderno_val   = input_val[1]
input_niguarda_val  = input_val[2]
output_niguarda_val = output_val[0]

# CLEAN_DATA

input_rain, input_rain_val           = library_datas.clean_data_with_val(input_rain, input_rain_val)
input_paderno, input_paderno_val     = library_datas.clean_data_with_val(input_paderno, input_paderno_val)
input_niguarda, input_niguarda_val   = library_datas.clean_data_with_val(input_niguarda, input_niguarda_val)
output_niguarda, output_niguarda_val = library_datas.clean_data_with_val(output_niguarda, output_niguarda_val)


predictions = model.predict(x=[input_rain_val[index_pluviometer], input_paderno_val, input_niguarda_val])

# EXTRACT DATA WITH DATA AND APPLY STEP
dates_val = library_datas.get_output_dates_graph_apply_step(2018, 2018)

library_datas.plot_graph_validation(predictions, output_niguarda_val, dates_val)

print("x")

