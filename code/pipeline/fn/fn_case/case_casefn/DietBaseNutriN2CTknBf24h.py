import pandas as pd

import numpy as np

CaseFnName = "DietBaseNutriN2CTknBf24h"

Ckpd_to_CkpdObsConfig = {'Bf24H': {'DistStartToPredDT': -24,
           'DistEndToPredDT': 1e-06,
           'TimeUnit': 'h',
           'StartIdx5Min': -288,
           'EndIdx5Min': 0}}

RO_to_ROName = {'Diet': 'hP.rDiet5Min.cBf24H'}

ROName_to_RONameInfo = {'hP.rDiet5Min.cBf24H': {'HumanName': 'P', 'RecordName': 'Diet5Min', 'CkpdName': 'Bf24H'}}

HumanRecordRecfeat_Args = {'P': {'P': [], 'Diet5Min': []}}

Attr_to_AttrConfig = {'Calories': {'Max': 1200, 'Min': 0, 'INTERVAL': 10},
 'Carbs': {'Max': 130, 'Min': 0, 'INTERVAL': 10},
 'Fat': {'Max': 80, 'Min': 0, 'INTERVAL': 5},
 'Fiber': {'Max': 30, 'Min': 0, 'INTERVAL': 1},
 'Protein': {'Max': 140, 'Min': 0, 'INTERVAL': 5}}

COVocab = {'idx2tkn': ['<pad>', '0 calories', '0 carbs', '0 fat', '0 fiber', '0 protein', '0-1 fiber',
             '0-10 calories', '0-10 carbs', '0-5 fat', '0-5 protein', '1-2 fiber', '10-11 fiber',
             '10-15 fat', '10-15 protein', '10-20 calories', '10-20 carbs', '100-105 protein',
             '100-110 calories', '100-110 carbs', '1000-1010 calories', '1010-1020 calories',
             '1020-1030 calories', '1030-1040 calories', '1040-1050 calories', '105-110 protein',
             '1050-1060 calories', '1060-1070 calories', '1070-1080 calories', '1080-1090 calories',
             '1090-1100 calories', '11-12 fiber', '110-115 protein', '110-120 calories',
             '110-120 carbs', '1100-1110 calories', '1110-1120 calories', '1120-1130 calories',
             '1130-1140 calories', '1140-1150 calories', '115-120 protein', '1150-1160 calories',
             '1160-1170 calories', '1170-1180 calories', '1180-1190 calories', '1190-1200 calories',
             '12-13 fiber', '120-125 protein', '120-130 calories', '120-130 carbs', '1200 calories',
             '125-130 protein', '13-14 fiber', '130 carbs', '130-135 protein', '130-140 calories',
             '135-140 protein', '14-15 fiber', '140 protein', '140-150 calories', '15-16 fiber',
             '15-20 fat', '15-20 protein', '150-160 calories', '16-17 fiber', '160-170 calories',
             '17-18 fiber', '170-180 calories', '18-19 fiber', '180-190 calories', '19-20 fiber',
             '190-200 calories', '2-3 fiber', '20-21 fiber', '20-25 fat', '20-25 protein',
             '20-30 calories', '20-30 carbs', '200-210 calories', '21-22 fiber', '210-220 calories',
             '22-23 fiber', '220-230 calories', '23-24 fiber', '230-240 calories', '24-25 fiber',
             '240-250 calories', '25-26 fiber', '25-30 fat', '25-30 protein', '250-260 calories',
             '26-27 fiber', '260-270 calories', '27-28 fiber', '270-280 calories', '28-29 fiber',
             '280-290 calories', '29-30 fiber', '290-300 calories', '3-4 fiber', '30 fiber',
             '30-35 fat', '30-35 protein', '30-40 calories', '30-40 carbs', '300-310 calories',
             '310-320 calories', '320-330 calories', '330-340 calories', '340-350 calories',
             '35-40 fat', '35-40 protein', '350-360 calories', '360-370 calories',
             '370-380 calories', '380-390 calories', '390-400 calories', '4-5 fiber', '40-45 fat',
             '40-45 protein', '40-50 calories', '40-50 carbs', '400-410 calories',
             '410-420 calories', '420-430 calories', '430-440 calories', '440-450 calories',
             '45-50 fat', '45-50 protein', '450-460 calories', '460-470 calories',
             '470-480 calories', '480-490 calories', '490-500 calories', '5-10 fat', '5-10 protein',
             '5-6 fiber', '50-55 fat', '50-55 protein', '50-60 calories', '50-60 carbs',
             '500-510 calories', '510-520 calories', '520-530 calories', '530-540 calories',
             '540-550 calories', '55-60 fat', '55-60 protein', '550-560 calories',
             '560-570 calories', '570-580 calories', '580-590 calories', '590-600 calories',
             '6-7 fiber', '60-65 fat', '60-65 protein', '60-70 calories', '60-70 carbs',
             '600-610 calories', '610-620 calories', '620-630 calories', '630-640 calories',
             '640-650 calories', '65-70 fat', '65-70 protein', '650-660 calories',
             '660-670 calories', '670-680 calories', '680-690 calories', '690-700 calories',
             '7-8 fiber', '70-75 fat', '70-75 protein', '70-80 calories', '70-80 carbs',
             '700-710 calories', '710-720 calories', '720-730 calories', '730-740 calories',
             '740-750 calories', '75-80 fat', '75-80 protein', '750-760 calories',
             '760-770 calories', '770-780 calories', '780-790 calories', '790-800 calories',
             '8-9 fiber', '80 fat', '80-85 protein', '80-90 calories', '80-90 carbs',
             '800-810 calories', '810-820 calories', '820-830 calories', '830-840 calories',
             '840-850 calories', '85-90 protein', '850-860 calories', '860-870 calories',
             '870-880 calories', '880-890 calories', '890-900 calories', '9-10 fiber',
             '90-100 calories', '90-100 carbs', '90-95 protein', '900-910 calories',
             '910-920 calories', '920-930 calories', '930-940 calories', '940-950 calories',
             '95-100 protein', '950-960 calories', '960-970 calories', '970-980 calories',
             '980-990 calories', '990-1000 calories', 'above 1200 calories', 'above 130 carbs',
             'above 140 protein', 'above 30 fiber', 'above 80 fat', 'no calories info',
             'no carbs info', 'no fat info', 'no fiber info', 'no protein info'],
 'tkn2tid': {'<pad>': 0,
             '0 calories': 1,
             '0 carbs': 2,
             '0 fat': 3,
             '0 fiber': 4,
             '0 protein': 5,
             '0-1 fiber': 6,
             '0-10 calories': 7,
             '0-10 carbs': 8,
             '0-5 fat': 9,
             '0-5 protein': 10,
             '1-2 fiber': 11,
             '10-11 fiber': 12,
             '10-15 fat': 13,
             '10-15 protein': 14,
             '10-20 calories': 15,
             '10-20 carbs': 16,
             '100-105 protein': 17,
             '100-110 calories': 18,
             '100-110 carbs': 19,
             '1000-1010 calories': 20,
             '1010-1020 calories': 21,
             '1020-1030 calories': 22,
             '1030-1040 calories': 23,
             '1040-1050 calories': 24,
             '105-110 protein': 25,
             '1050-1060 calories': 26,
             '1060-1070 calories': 27,
             '1070-1080 calories': 28,
             '1080-1090 calories': 29,
             '1090-1100 calories': 30,
             '11-12 fiber': 31,
             '110-115 protein': 32,
             '110-120 calories': 33,
             '110-120 carbs': 34,
             '1100-1110 calories': 35,
             '1110-1120 calories': 36,
             '1120-1130 calories': 37,
             '1130-1140 calories': 38,
             '1140-1150 calories': 39,
             '115-120 protein': 40,
             '1150-1160 calories': 41,
             '1160-1170 calories': 42,
             '1170-1180 calories': 43,
             '1180-1190 calories': 44,
             '1190-1200 calories': 45,
             '12-13 fiber': 46,
             '120-125 protein': 47,
             '120-130 calories': 48,
             '120-130 carbs': 49,
             '1200 calories': 50,
             '125-130 protein': 51,
             '13-14 fiber': 52,
             '130 carbs': 53,
             '130-135 protein': 54,
             '130-140 calories': 55,
             '135-140 protein': 56,
             '14-15 fiber': 57,
             '140 protein': 58,
             '140-150 calories': 59,
             '15-16 fiber': 60,
             '15-20 fat': 61,
             '15-20 protein': 62,
             '150-160 calories': 63,
             '16-17 fiber': 64,
             '160-170 calories': 65,
             '17-18 fiber': 66,
             '170-180 calories': 67,
             '18-19 fiber': 68,
             '180-190 calories': 69,
             '19-20 fiber': 70,
             '190-200 calories': 71,
             '2-3 fiber': 72,
             '20-21 fiber': 73,
             '20-25 fat': 74,
             '20-25 protein': 75,
             '20-30 calories': 76,
             '20-30 carbs': 77,
             '200-210 calories': 78,
             '21-22 fiber': 79,
             '210-220 calories': 80,
             '22-23 fiber': 81,
             '220-230 calories': 82,
             '23-24 fiber': 83,
             '230-240 calories': 84,
             '24-25 fiber': 85,
             '240-250 calories': 86,
             '25-26 fiber': 87,
             '25-30 fat': 88,
             '25-30 protein': 89,
             '250-260 calories': 90,
             '26-27 fiber': 91,
             '260-270 calories': 92,
             '27-28 fiber': 93,
             '270-280 calories': 94,
             '28-29 fiber': 95,
             '280-290 calories': 96,
             '29-30 fiber': 97,
             '290-300 calories': 98,
             '3-4 fiber': 99,
             '30 fiber': 100,
             '30-35 fat': 101,
             '30-35 protein': 102,
             '30-40 calories': 103,
             '30-40 carbs': 104,
             '300-310 calories': 105,
             '310-320 calories': 106,
             '320-330 calories': 107,
             '330-340 calories': 108,
             '340-350 calories': 109,
             '35-40 fat': 110,
             '35-40 protein': 111,
             '350-360 calories': 112,
             '360-370 calories': 113,
             '370-380 calories': 114,
             '380-390 calories': 115,
             '390-400 calories': 116,
             '4-5 fiber': 117,
             '40-45 fat': 118,
             '40-45 protein': 119,
             '40-50 calories': 120,
             '40-50 carbs': 121,
             '400-410 calories': 122,
             '410-420 calories': 123,
             '420-430 calories': 124,
             '430-440 calories': 125,
             '440-450 calories': 126,
             '45-50 fat': 127,
             '45-50 protein': 128,
             '450-460 calories': 129,
             '460-470 calories': 130,
             '470-480 calories': 131,
             '480-490 calories': 132,
             '490-500 calories': 133,
             '5-10 fat': 134,
             '5-10 protein': 135,
             '5-6 fiber': 136,
             '50-55 fat': 137,
             '50-55 protein': 138,
             '50-60 calories': 139,
             '50-60 carbs': 140,
             '500-510 calories': 141,
             '510-520 calories': 142,
             '520-530 calories': 143,
             '530-540 calories': 144,
             '540-550 calories': 145,
             '55-60 fat': 146,
             '55-60 protein': 147,
             '550-560 calories': 148,
             '560-570 calories': 149,
             '570-580 calories': 150,
             '580-590 calories': 151,
             '590-600 calories': 152,
             '6-7 fiber': 153,
             '60-65 fat': 154,
             '60-65 protein': 155,
             '60-70 calories': 156,
             '60-70 carbs': 157,
             '600-610 calories': 158,
             '610-620 calories': 159,
             '620-630 calories': 160,
             '630-640 calories': 161,
             '640-650 calories': 162,
             '65-70 fat': 163,
             '65-70 protein': 164,
             '650-660 calories': 165,
             '660-670 calories': 166,
             '670-680 calories': 167,
             '680-690 calories': 168,
             '690-700 calories': 169,
             '7-8 fiber': 170,
             '70-75 fat': 171,
             '70-75 protein': 172,
             '70-80 calories': 173,
             '70-80 carbs': 174,
             '700-710 calories': 175,
             '710-720 calories': 176,
             '720-730 calories': 177,
             '730-740 calories': 178,
             '740-750 calories': 179,
             '75-80 fat': 180,
             '75-80 protein': 181,
             '750-760 calories': 182,
             '760-770 calories': 183,
             '770-780 calories': 184,
             '780-790 calories': 185,
             '790-800 calories': 186,
             '8-9 fiber': 187,
             '80 fat': 188,
             '80-85 protein': 189,
             '80-90 calories': 190,
             '80-90 carbs': 191,
             '800-810 calories': 192,
             '810-820 calories': 193,
             '820-830 calories': 194,
             '830-840 calories': 195,
             '840-850 calories': 196,
             '85-90 protein': 197,
             '850-860 calories': 198,
             '860-870 calories': 199,
             '870-880 calories': 200,
             '880-890 calories': 201,
             '890-900 calories': 202,
             '9-10 fiber': 203,
             '90-100 calories': 204,
             '90-100 carbs': 205,
             '90-95 protein': 206,
             '900-910 calories': 207,
             '910-920 calories': 208,
             '920-930 calories': 209,
             '930-940 calories': 210,
             '940-950 calories': 211,
             '95-100 protein': 212,
             '950-960 calories': 213,
             '960-970 calories': 214,
             '970-980 calories': 215,
             '980-990 calories': 216,
             '990-1000 calories': 217,
             'above 1200 calories': 218,
             'above 130 carbs': 219,
             'above 140 protein': 220,
             'above 30 fiber': 221,
             'above 80 fat': 222,
             'no calories info': 223,
             'no carbs info': 224,
             'no fat info': 225,
             'no fiber info': 226,
             'no protein info': 227}}

def RecFeat_Tokenizer_fn(rec, Attr_to_AttrConfig):

    d = {}
    for attr, AttrConfig in Attr_to_AttrConfig.items():
        Max = AttrConfig['Max']
        Min = AttrConfig['Min']

        Length = len(str(int(Max)))


        INTERVAL = AttrConfig['INTERVAL']
        if pd.isnull(rec.get(attr, None)) :
            d[f"no {attr.lower()} info"] = 1
        elif float(rec[attr]) == Max:
            d[ f"{Max} {attr.lower()}"] = 1
        elif float(rec[attr]) == Min:
            d[ f"{Min} {attr.lower()}"] = 1
        elif float(rec[attr]) > Max:
            d[ f"above {Max} {attr.lower()}"] = 1
        elif float(rec[attr]) < Min:
            pass 
            # d[ f"below {Min} {attr.lower()}"] = 1
        else:
            lower_bound = int((float(rec[attr]) // INTERVAL) * INTERVAL)
            upper_bound = int(lower_bound + INTERVAL)

            # Calculate the proportion of value within the interval
            # proportion = (float(rec[attr]) - lower_bound) / INTERVAL
            # proportion = round((float(rec[attr]) - lower_bound) / INTERVAL, 4)
            # Construct the keys
            key1 = f"{str(lower_bound)}-{str(upper_bound)} {attr.lower()}"
            # key2 = f"{key1}Itv"
            # Add them to the dictionary with appropriate weights
            d[key1] = 1
            # d[key2] = proportion

    # output = self.post_process_recfeat(d)   
    tkn = list(d.keys())
    # wgt = list(d.values())
    output = {'tkn': tkn, # 'wgt': wgt
              }
    return output


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

    ObsDT = case_example['ObsDT']
    tkn2idx = COVocab['tkn2tid']

    if df is not None and len(df) > 0:
        tid_all = []
        fiveminutes_all = []
        for idx, rec in df.iterrows():
            DT_s = rec['DT_s']
            # print(DT_s)
            # print(ObsDT)

            time_delta = DT_s - ObsDT
            minutes = time_delta.total_seconds() / 60
            fiveminute_index = int(minutes / 5)
            # print(f"Time difference in minutes: {minutes}")
            # print(f'The 5 minute index is: {fiveminute_index}')

            d = RecFeat_Tokenizer_fn(rec, Attr_to_AttrConfig)
            # print(d)
            tkn = d['tkn']
            # print(tkn)

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
	"Attr_to_AttrConfig": Attr_to_AttrConfig,
	"COVocab": COVocab,
	"RecFeat_Tokenizer_fn": RecFeat_Tokenizer_fn,
	"fn_CaseFn": fn_CaseFn
}