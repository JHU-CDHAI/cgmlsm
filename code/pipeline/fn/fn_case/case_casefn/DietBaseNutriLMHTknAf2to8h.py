import pandas as pd

import numpy as np

CaseFnName = "DietBaseNutriLMHTknAf2to8h"

Ckpd_to_CkpdObsConfig = {'Af2to8H': {'DistStartToPredDT': 121,
             'DistEndToPredDT': 481,
             'TimeUnit': 'min',
             'StartIdx5Min': 25,
             'EndIdx5Min': 96}}

RO_to_ROName = {'Diet': 'hP.rDiet5Min.cAf2to8H'}

ROName_to_RONameInfo = {'hP.rDiet5Min.cAf2to8H': {'HumanName': 'P', 'RecordName': 'Diet5Min', 'CkpdName': 'Af2to8H'}}

HumanRecordRecfeat_Args = {'P': {'P': [], 'Diet5Min': []}}

COVocab = {'idx2tkn': ['<pad>', 'low calories', 'medium calories', 'high calories', 'low carbs',
             'medium carbs', 'high carbs', 'low fiber', 'medium fiber', 'high fiber', 'low fat',
             'medium fat', 'high fat', 'low protein', 'medium protein', 'high protein'],
 'tkn2tid': {'<pad>': 0,
             'low calories': 1,
             'medium calories': 2,
             'high calories': 3,
             'low carbs': 4,
             'medium carbs': 5,
             'high carbs': 6,
             'low fiber': 7,
             'medium fiber': 8,
             'high fiber': 9,
             'low fat': 10,
             'medium fat': 11,
             'high fat': 12,
             'low protein': 13,
             'medium protein': 14,
             'high protein': 15}}

def fn_CaseFn(case_example,     # <--- case to process
               ROName_list,      # <--- from COName
               ROName_to_ROData, # <--- in scope of case_example
               ROName_to_ROInfo, # <--- in scope of CaseFleshingTask
               COVocab,          # <--- in scope of CaseFleshingTask, from ROName_to_ROInfo
               caseset,          # <--- in scope of CaseFleshingTask,
               ):

    assert len(ROName_list) == 1
    ROName = ROName_list[0]
    df = ROName_to_ROData[ROName]

    # THRESHOLDS = {
    #     "Carbs": (15, 50),
    #     "Fiber": (2, 5),
    #     "Fat": (5, 20),
    #     "Protein": (5, 15),

    #     "SaturatedFat": (2, 5),

    #     "Calories": (150, 500),
    #     "Sugar": (5, 15),
    #     "AddedSugars": (5, 10),
    #     "Sodium": (200, 800),
    #     "Cholesterol": (30, 100),
    #     "Potassium": (200, 700),
    #     "TransFat": (0, 0.5),  # Any trans fat should be avoided
    # }

    THRESHOLDS = {
        "Calories": (150, 500),
        "Carbs": (15, 50),
        "Protein": (5, 15),
        "Fat": (5, 20),
        "Fiber": (2, 5),
    }

    def classify_nutrient(value, low, high):
        """Classifies a nutrient value as 'Low', 'Medium', or 'High' based on given thresholds."""
        if value < low:
            return "low"
        elif value > high:
            return "high"
        else:
            return "medium"


    def generate_food_description(row):
        """Generates a concise sentence describing the food's nutritional category."""
        # description = f"{row['FoodName']} is a "
        # activity_type = row['ActivityType']
        # if 'Breakfast' in activity_type:
        #     description += 'breakfast'
        # elif 'Lunch' in activity_type:
        #     description += 'lunch'
        # elif 'Dinner' in activity_type:
        #     description += 'dinner'
        # elif 'Snack' in activity_type:
        #     description += 'snack'

        # description = description + ': ' 

        description = ''
        categories = []
        for nutrient, (low, high) in THRESHOLDS.items():

            nutrient_list = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Calories', 'Sugar', ]
            if nutrient not in nutrient_list:
                continue 
            if nutrient in row:
                value = row[nutrient]
                if value <= 0:
                    continue 
                classification = classify_nutrient(value, low, high)
                categories.append(f"{classification} {nutrient.lower()}")

        description += "; ".join(categories) # + "."
        return description



    ObsDT = case_example['ObsDT']
    tkn2idx = COVocab['tkn2tid']



    if df is not None and len(df) > 0:
        tid_all = []
        fiveminutes_all = []
        for idx, rec in df.iterrows():
            DT_s = rec['DT_s']

            time_delta = DT_s - ObsDT
            minutes = time_delta.total_seconds() / 60
            fiveminute_index = int(minutes / 5)
            description = generate_food_description(rec)

            tkn = [i for i in description.split('; ')]
            tid = [tkn2idx[i] for i in tkn]
            tid_all.append(tid)
            fiveminutes_all.append(fiveminute_index)


        output = {
            '-tid': tid_all,
            '-timestep': fiveminutes_all,
        }

        # pprint(output)
        # make sure the d_total's keys are consistent.  
        #############################################
    else:
        output = {
            '-tid': [[]],
            '-timestep': [],
        }
    return output


MetaDict = {
	"CaseFnName": CaseFnName,
	"Ckpd_to_CkpdObsConfig": Ckpd_to_CkpdObsConfig,
	"RO_to_ROName": RO_to_ROName,
	"ROName_to_RONameInfo": ROName_to_RONameInfo,
	"HumanRecordRecfeat_Args": HumanRecordRecfeat_Args,
	"COVocab": COVocab,
	"fn_CaseFn": fn_CaseFn
}