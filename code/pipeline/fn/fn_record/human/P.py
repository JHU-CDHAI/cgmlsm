import os

import pandas as pd

import numpy as np

OneHuman_Args = {'HumanName': 'P', 'HumanID': 'PID', 'RawHumanID': 'PatientID', 'HumanIDLength': 10}

Excluded_RawNameList = ['Patient', 'QuestionResponse', 'PatientBloodGlucoseTargets', 'Rx', 'PatientObservationSummary', 'PatientTargetSegment', 'TDC']

def get_RawHumanID_from_dfRawColumns(dfRawColumns):
    RawHumanID_selected = None 
    if 'PatientID' in dfRawColumns: 
        RawHumanID_selected = 'PatientID'  
    return RawHumanID_selected


MetaDict = {
	"OneHuman_Args": OneHuman_Args,
	"Excluded_RawNameList": Excluded_RawNameList,
	"get_RawHumanID_from_dfRawColumns": get_RawHumanID_from_dfRawColumns
}