import pandas as pd
from utils import NEW_ACCIDENT_PATH, NEW_VEHICLE_PATH, NEW_CASUALTY_PATH


def create_kb(path: str, name: str):
    print("Reading data...")
    accident = pd.read_csv(NEW_ACCIDENT_PATH, low_memory=False)
    vehicle = pd.read_csv(NEW_VEHICLE_PATH, low_memory=False)
    casualty = pd.read_csv(NEW_CASUALTY_PATH, low_memory=False)
    prolog_file = open(path + name, "w")
    print(f"Creating knowledge base at {path + name}...")
    prolog_file.write(":-style_check(-discontiguous).\n")

    # accident
    print("Writing accident facts...")
    for index, row in accident.iterrows():
        accident_index = f"accident(\"{row['accident_index']}\")"
        prolog_file.write(f"accident_severity({accident_index}, {row['accident_severity']}).\n")
        prolog_file.write(f"number_vehicle({accident_index}, {row['number_of_vehicles']}).\n")
        prolog_file.write(f"number_casualty({accident_index}, {row['number_of_casualties']}).\n")
        prolog_file.write(f"date({accident_index}, {row['date']}).\n")
        time = row['time'].replace(":", ".")
        prolog_file.write(f"time({accident_index}, {time}).\n")
        prolog_file.write(f"accident_ons({accident_index}, \"{row['local_authority_ons_district']}\").\n")
        prolog_file.write(f"road_type({accident_index}, {row['road_type']}).\n")
        prolog_file.write(f"light_conditions({accident_index}, {row['light_conditions']}).\n")
        prolog_file.write(f"weather_conditions({accident_index}, {row['weather_conditions']}).\n")
        prolog_file.write(f"road_surface_conditions({accident_index}, {row['road_surface_conditions']}).\n")
        prolog_file.write(f"special_conditions_at_site({accident_index}, {row['special_conditions_at_site']}).\n")
        prolog_file.write(f"carriageway_hazards({accident_index}, {row['carriageway_hazards']}).\n")
        prolog_file.write(f"first_road_class({accident_index}, {row['first_road_class']}).\n")
        prolog_file.write(f"first_road_number({accident_index}, {row['first_road_number']}).\n")
        prolog_file.write(f"second_road_class({accident_index}, {row['second_road_class']}).\n")
        prolog_file.write(f"second_road_number({accident_index}, {row['second_road_number']}).\n")
    print("-accident done")

    # vehicle
    print("Writing vehicle facts...")
    for index, row in vehicle.iterrows():
        v = f"vehicle(accident(\"{row['accident_index']}\"), {row['vehicle_reference']})"
        prolog_file.write(f"{v}.\n")
        prolog_file.write(f"vehicle_type({v}, {row['vehicle_type']}).\n")
        prolog_file.write(f"engine_capacity({v}, {row['engine_capacity_cc']}).\n")
        prolog_file.write(f"propulsion({v}, {row['propulsion_code']}).\n")
        prolog_file.write(f"vehicle_age({v}, {row['age_of_vehicle']}).\n")
        prolog_file.write(f"driver_sex({v}, {row['sex_of_driver']}).\n")
        prolog_file.write(f"driver_age({v}, {row['age_of_driver']}).\n")
        prolog_file.write(f"towing_and_articulation({v}, {row['towing_and_articulation']}).\n")
        prolog_file.write(f"first_point_of_impact({v}, {row['first_point_of_impact']}).\n")
    print("-vehicle done")

    # casualty
    print("Writing casualty facts...")
    for index, row in casualty.iterrows():
        c = f"casualty(accident(\"{row['accident_index']}\"), {row['casualty_reference']})"
        prolog_file.write(f"{c}.\n")
        prolog_file.write(f"casualty_sex({c}, {row['sex_of_casualty']}).\n")
        prolog_file.write(f"casualty_age({c}, {row['age_of_casualty']}).\n")
        prolog_file.write(f"casualty_severity({c}, {row['casualty_severity']}).\n")
        prolog_file.write(f"casualty_type({c}, {row['casualty_type']}).\n")
    print("-casualty done")

    prolog_file.close()
    print("Knowledge base created!")


create_kb("./", "kb.pl")
