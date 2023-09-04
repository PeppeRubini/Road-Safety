import pandas as pd
from pyswip import Prolog
from utils import NEW_ACCIDENT_PATH


def define_clause(kb_path: str, kb_name: str) -> Prolog:
    prolog = Prolog()
    prolog.consult(f"{kb_path}{kb_name}")
    prolog.assertz("same_ons(accident(A1), accident(A2)) :- accident_ons(accident(A1), ONS), accident_ons(accident(A2), ONS)")
    prolog.assertz("num_accident_in_ons(accident(A1), N) :- findall(A2, same_ons(accident(A1), accident(A2)), L), length(L, N)")

    prolog.assertz("same_road_type(accident(A1), accident(A2)) :- road_type(accident(A1), RT), road_type(accident(A2), RT)")
    prolog.assertz("num_accident_in_same_road_type(accident(A1), N) :- findall(A2, same_road_type(accident(A1), accident(A2)), L), length(L, N)")

    prolog.assertz("same_road_class(accident(A1), accident(A2)) :- first_road_class(accident(A1), FC), first_road_class(accident(A2), FC)")
    prolog.assertz("same_road_number(accident(A1), accident(A2)) :- first_road_number(accident(A1), FN), first_road_number(accident(A2), FN)")
    prolog.assertz("same_road(accident(A1), accident(A2)) :- same_road_class(accident(A1), accident(A2)), same_road_number(accident(A1), accident(A2))")
    prolog.assertz("num_accident_in_same_road(accident(A1), N) :- findall(A2, same_road(accident(A1), accident(A2)), L), length(L, N)")

    prolog.assertz("same_second_road_class(accident(A1), accident(A2)) :- second_road_class(accident(A1), SC), second_road_class(accident(A2), SC)")
    prolog.assertz("same_second_road_number(accident(A1), accident(A2)) :- second_road_number(accident(A1), SN), second_road_number(accident(A2), SN)")
    prolog.assertz("same_second_road(accident(A1), accident(A2)) :- same_second_road_class(accident(A1), accident(A2)), same_second_road_number(accident(A1), accident(A2))")
    prolog.assertz("num_accident_in_same_second_road(accident(A1), N) :- findall(A2, same_second_road(accident(A1), accident(A2)), L), length(L, N)")

    prolog.assertz("is_night(accident(A)) :- time(accident(A), T), (T >= 21; T =< 4)")

    prolog.assertz("is_casualty_child(casualty(accident(A), C)) :- casualty_age(casualty(accident(A), C), CA), CA =< 12")
    prolog.assertz("is_casualty_old(casualty(accident(A), C)) :- casualty_age(casualty(accident(A), C), CA), CA >= 70")

    prolog.assertz("is_pedestrian(casualty(accident(A), C)) :- casualty_type(casualty(accident(A), C), CT), CT == 0")
    prolog.assertz("is_car(vehicle(accident(A), V)) :- vehicle_type(vehicle(accident(A), V), VT), (VT == 8; VT == 9; VT == 10; VT == 19)")
    prolog.assertz("is_heavy_vehicle(vehicle(accident(A), V)) :- vehicle_type(vehicle(accident(A), V), VT), (VT == 11; VT == 20; VT == 21; VT == 98)")
    prolog.assertz("is_motorcycle(vehicle(accident(A), V)) :- vehicle_type(vehicle(accident(A), V), VT), (VT == 2; VT == 3; VT == 4; VT == 5; VT == 22; VT == 23; VT == 97)")
    prolog.assertz("is_pedal_cycle(vehicle(accident(A), V)) :- vehicle_type(vehicle(accident(A), V), VT), VT == 1")

    prolog.assertz("num_cars(accident(A), N) :- findall(V, is_car(vehicle(accident(A), V)), L), length(L, N)")
    prolog.assertz("num_heavy_vehicles(accident(A), N) :- findall(V, is_heavy_vehicle(vehicle(accident(A), V)), L), length(L, N)")
    prolog.assertz("num_motorcycles(accident(A), N) :- findall(V, is_motorcycle(vehicle(accident(A), V)), L), length(L, N)")
    prolog.assertz("num_pedal_cycles(accident(A), N) :- findall(V, is_pedal_cycle(vehicle(accident(A), V)), L), length(L, N)")

    return prolog


def query_to_dict_list(prolog: Prolog):
    accident = pd.read_csv(NEW_ACCIDENT_PATH, low_memory=False)
    dict_list = []
    for a in accident['accident_index']:
        print(accident[accident['accident_index'] == a].index[0], "/", len(accident))
        try:
            features_dict = {}
            a = f"\"{a}\""
            features_dict["accident_index"] = a
            features_dict["accident_severity"] = list(prolog.query(f"accident_severity(accident({a}), AS)"))[0]["AS"]
            features_dict["num_accident_in_ons"] = list(prolog.query(f"num_accident_in_ons(accident({a}), N)"))[0]["N"]
            features_dict["num_accident_in_same_road_type"] = list(prolog.query(f"num_accident_in_same_road_type(accident({a}), N)"))[0]["N"]

            features_dict["num_accident_in_same_road"] = list(prolog.query(f"num_accident_in_same_road(accident({a}), N)"))[0]["N"]
            features_dict["num_accident_in_same_second_road"] = list(prolog.query(f"num_accident_in_same_second_road(accident({a}), N)"))[0]["N"]

            features_dict["num_casualties"] = list(prolog.query(f"number_casualty(accident({a}), NC)"))[0]["NC"]
            features_dict["num_vehicle_involved"] = list(prolog.query(f"number_vehicle(accident({a}), NV)"))[0]["NV"]

            features_dict["weather_conditions"] = list(prolog.query(f"weather_conditions(accident({a}), WC)"))[0]["WC"]
            features_dict["light_conditions"] = list(prolog.query(f"light_conditions(accident({a}), LC)"))[0]["LC"]
            features_dict["road_surface_conditions"] = list(prolog.query(f"road_surface_conditions(accident({a}), SC)"))[0]["SC"]
            features_dict["carriageway_hazards"] = list(prolog.query(f"carriageway_hazards(accident({a}), H)"))[0]["H"]
            features_dict["special_conditions_at_site"] = list(prolog.query(f"special_conditions_at_site(accident({a}), C)"))[0]["C"]

            features_dict["is_night"] = int(bool(list(prolog.query(f"is_night(accident({a}))"))))

            features_dict["num_cars"] = list(prolog.query(f"num_cars(accident({a}), N)"))[0]["N"]
            features_dict["num_heavy_vehicles"] = list(prolog.query(f"num_heavy_vehicles(accident({a}), N)"))[0]["N"]
            features_dict["num_motorcycles"] = list(prolog.query(f"num_motorcycles(accident({a}), N)"))[0]["N"]
            features_dict["num_pedal_cycles"] = list(prolog.query(f"num_pedal_cycles(accident({a}), N)"))[0]["N"]

            features_dict["num_pedestrians"] = 0

            child = False
            old = False
            for c in list(prolog.query(f"casualty(accident({a}), C)")):
                c = c["C"]
                features_dict["num_pedestrians"] += int(
                    bool(list(prolog.query(f"is_pedestrian(casualty(accident({a}), {c}))"))))
                if not child:
                    child = bool(list(prolog.query(f"is_casualty_child(casualty(accident({a}), {c}))")))
                if not old:
                    old = bool(list(prolog.query(f"is_casualty_old(casualty(accident({a}), {c}))")))

            features_dict["is_child_involved"] = int(child)
            features_dict["is_old_involved"] = int(old)

            dict_list.append(features_dict)
        except:
            print("exception", accident[accident['accident_index'] == a].index[0])
    return dict_list


print("Defining clauses...")
p = define_clause("../Dataset", "kb.pl")
print("Querying...")
new_dataset = pd.DataFrame(query_to_dict_list(p))
new_dataset.to_csv("../Dataset/new_dataset.csv", index=False)
print("Done!")
