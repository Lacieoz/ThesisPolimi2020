from libraries_and_data import library_datas
from models import linear_model
import random

import matplotlib.pyplot as plt

random.seed(a=1234, version=2)

# LN = LEVEL NIGUARDA
# LP = LEVEL PADERNO
# RP = RAIN PLUVIOMETERS

# TRAIN DATA

input_rain      = library_datas.get_data_by_years_and_type("RP", 2015, 2017)
input_paderno   = library_datas.get_data_by_years_and_type("LP", 2015, 2017)
input_niguarda  = library_datas.get_data_by_years_and_type("LN", 2015, 2017)
input_cantu     = library_datas.get_data_by_years_and_type("LC", 2015, 2017)
output_niguarda = input_niguarda

input_list = [input_rain, input_paderno, input_niguarda, input_cantu]

# VALIDATION DATA

input_rain_val      = library_datas.get_data_by_years_and_type("RP", 2018, 2018)
input_paderno_val   = library_datas.get_data_by_years_and_type("LP", 2018, 2018)
input_niguarda_val  = library_datas.get_data_by_years_and_type("LN", 2018, 2018)
input_cantu_val     = library_datas.get_data_by_years_and_type("LC", 2018, 2018)
output_niguarda_val = input_niguarda_val

input_val = [input_rain_val, input_paderno_val, input_niguarda_val, input_cantu_val]

graph = True

sensore = input("sensore?")
sensore = int(sensore)

while graph:

    timestep = input("timestep?")
    timestep = int(timestep)


    library_datas.set_timestep(timestep)

    # TIME STEP

    input_list_graph, output_graph = library_datas.apply_step_diff(input_list, [output_niguarda])

    input_rain_graph = input_list_graph[0]
    input_paderno_graph = input_list_graph[1]
    input_niguarda_graph = input_list_graph[2]
    input_cantu_graph    = input_list_graph[3]
    output_niguarda_graph = output_graph[0]

    input_val_graph, output_val_graph = library_datas.apply_step_diff(input_val, [output_niguarda_val])

    input_rain_val_graph = input_val_graph[0]
    input_paderno_val_graph = input_val_graph[1]
    input_niguarda_val_graph = input_val_graph[2]
    input_cantu_val_graph    = input_val_graph[3]
    output_niguarda_val_graph = output_val_graph[0]



    # CLEAN_DATA

    input_rain_graph, input_rain_val_graph           = library_datas.clean_data_with_val(input_rain_graph, input_rain_val_graph)
    input_paderno_graph, input_paderno_val_graph     = library_datas.clean_data_with_val(input_paderno_graph, input_paderno_val_graph)
    input_niguarda_graph, input_niguarda_val_graph   = library_datas.clean_data_with_val(input_niguarda_graph, input_niguarda_val_graph)
    input_cantu_graph, input_cantu_val_graph         = library_datas.clean_data_with_val(input_cantu_graph, input_cantu_val_graph)
    output_niguarda_graph, output_niguarda_val_graph = library_datas.clean_data_with_val(output_niguarda_graph, output_niguarda_val_graph)
    delta_output_graph = library_datas.compute_list_delta(output_niguarda_graph)

    # GRAFICO

    plt.plot(input_cantu_graph, output_niguarda_graph, 'ro')
    plt.xlabel('cantu', color='#1C2833')
    plt.ylabel('niguarda', color='#1C2833')
    plt.axis([0, 200, 0, 200])
    plt.show()
    """
    plt.plot(input_paderno_graph, output_niguarda_graph, 'ro')
    plt.xlabel('paderno', color='#1C2833')
    plt.ylabel('niguarda', color='#1C2833')
    plt.axis([0, 300, 0, 300])
    plt.show()
    
    plt.plot(input_niguarda_graph, output_niguarda_graph, 'ro')
    plt.axis([0, 500, 0, 250])
    plt.show()
    """
    plt.plot(input_rain_graph[sensore], output_niguarda_graph, 'ro')
    plt.xlabel('rain', color='#1C2833')
    plt.ylabel('niguarda', color='#1C2833')
    plt.axis([0, 10, 0, 200])
    plt.show()
    """
    plt.plot(input_paderno_val_graph, output_niguarda_val_graph, 'ro')
    plt.axis([0, 300, 0, 300])
    plt.show()
    
    plt.plot(input_niguarda_val_graph, output_niguarda_val_graph, 'ro')
    plt.axis([0, 500, 0, 250])
    plt.show()
    
    plt.plot(input_rain_graph[sensore], output_niguarda_graph, 'ro')
    plt.axis([0, 20, 0, 250])
    plt.show()
    """

