import pandas as pd
from utils import *


def preprocessing():
    print("Reading data...")
    accident = pd.DataFrame()
    vehicle = pd.DataFrame()
    casualty = pd.DataFrame()
    for i in range(16, 22):
        print(f"Reading 20{i}...")
        a_df = pd.read_csv(f"../dataset/dft-road-casualty-statistics-accident-20{i}.csv", low_memory=False)
        v_df = pd.read_csv(f"../dataset/dft-road-casualty-statistics-vehicle-20{i}.csv", low_memory=False)
        c_df = pd.read_csv(f"../dataset/dft-road-casualty-statistics-casualty-20{i}.csv", low_memory=False)
        try:
            v.drop(['lsoa_of_driver'], axis=1, inplace=True)
            c.drop(['lsoa_of_casualty'], axis=1, inplace=True)
        except:
            pass
        accident = pd.concat([accident, a_df], ignore_index=True)
        vehicle = pd.concat([vehicle, v_df], ignore_index=True)
        casualty = pd.concat([casualty, c_df], ignore_index=True)

    print("Preprocessing...")
    # accident
    print("Preprocessing accident...")
    drop_column_accident = ['latitude', 'longitude', 'accident_reference', 'accident_year', 'location_easting_osgr',
                            'location_northing_osgr', 'local_authority_district', 'lsoa_of_accident_location']
    accident.drop(drop_column_accident, axis=1, inplace=True)
    column_check_na_accident = ['accident_index', 'accident_severity', 'number_of_vehicles', 'number_of_casualties',
                                'date', 'time', 'local_authority_ons_district', 'road_type', 'speed_limit',
                                'light_conditions', 'weather_conditions', 'road_surface_conditions',
                                'special_conditions_agit pt_site', 'carriageway_hazards', 'first_road_class',
                                'first_road_number', 'second_road_class', 'second_road_number',
                                'pedestrian_crossing_physical_facilities']
    accident.dropna(subset=column_check_na_accident, inplace=True)
    accident = accident[accident['road_type'] != -1]
    accident = accident[accident['first_road_number'] != -1]
    accident = accident[accident['second_road_number'] != -1]

    # vehicle
    print("Preprocessing vehicle...")
    drop_column_vehicle = ['accident_reference', 'accident_year', 'age_band_of_driver', 'driver_imd_decile',
                           'driver_home_area_type']
    vehicle.drop(drop_column_vehicle, axis=1, inplace=True)
    column_check_na_vehicle = ['accident_index', 'vehicle_reference', 'vehicle_type', 'engine_capacity_cc',
                               'age_of_vehicle', 'propulsion_code', 'sex_of_driver', 'age_of_driver',
                               'towing_and_articulation', 'first_point_of_impact']
    vehicle.dropna(subset=column_check_na_vehicle, inplace=True)

    # casualty
    print("Preprocessing casualty...")
    drop_column_casualty = ['accident_reference', 'accident_year', 'age_band_of_casualty', 'casualty_imd_decile',
                            'casualty_home_area_type']
    casualty.drop(drop_column_casualty, axis=1, inplace=True)
    column_check_na_casualty = ['accident_index', 'casualty_reference', 'casualty_severity', 'casualty_type']
    casualty.dropna(subset=column_check_na_casualty, inplace=True)
    casualty = casualty[~casualty['age_of_casualty'] < 0]

    stats(accident, vehicle, casualty, save=True, display=False)
    print("Preprocessing done!")

    print("Balancing data...")
    min_samples = accident['accident_severity'].value_counts().min()
    balanced_accident = pd.DataFrame()
    for severity_level in accident['accident_severity'].unique():
        sampled_data = accident[accident['accident_severity'] == severity_level].sample(n=min_samples, random_state=42)
        balanced_accident = balanced_accident._append(sampled_data)

    deleted_rows = list(set(accident['accident_index']) - set(balanced_accident['accident_index']))
    balanced_vehicle = vehicle[~vehicle['accident_index'].isin(deleted_rows)]
    balanced_casualty = casualty[~casualty['accident_index'].isin(deleted_rows)]
    print("Balancing done!")

    return balanced_accident, balanced_vehicle, balanced_casualty


a, v, c = preprocessing()
print("Saving preprocessed data...")
a.to_csv(NEW_ACCIDENT_PATH, index=False)
v.to_csv(NEW_VEHICLE_PATH, index=False)
c.to_csv(NEW_CASUALTY_PATH, index=False)
print("Preprocessed data saved!")
