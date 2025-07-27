import pandas as pd


class DictToObj:
    def __init__(self, **dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])


def find_timelist_index(dates, DT):
    low, high = 0, len(dates) - 1
    if DT < dates[0]:
        # return -1  # DT is smaller than the first date in the list
        return 0     # DT is smaller than the first date in the list
    if DT > dates[-1]:
        # return len(dates)  # DT is larger or equal to the last date in the list
        return len(dates)    # DT is larger or equal to the last date in the list

    while low <= high:
        mid = (low + high) // 2
        if dates[mid] < DT:
            low = mid + 1
        else:
            high = mid - 1
    return low

def parse_ROName(ROName):
    element_list = ROName.split('.')
    Letter_to_Full = {'h': 'HumanName', 'r': 'RecordName', 'f': 'RecFeatName', 'c': 'CkpdName'}
    RONameInfo = {Letter_to_Full[i[0]]: i[1:] for i in element_list}

    if 'RecFeatName' in RONameInfo:
        RONameInfo['RecFeatName'] = RONameInfo['RecordName'] + '-' + RONameInfo['RecFeatName']

    return RONameInfo 

def get_RONameToROInfo(ROName_list, cohort_record_base, Ckpd_to_CkpdObsConfig):
    
    # update ROName_list from RecFeat.
    # ROName_rec_list = []
    # for ROName in ROName_list:
    #     if '.f' in ROName:
    #         ROName_rec = '.'.join([i for i in ROName.split('.') if i[0] != 'f']) 
    #         ROName_rec_list.append(ROName_rec)
    # ROName_list = list(set(ROName_list + ROName_rec_list))


    ROName_to_ROInfo = {}
    for ROName in ROName_list:
        ROInfo = {}

        element_list = ROName.split('.')
        Letter_to_Full = {'h': 'HumanName', 'r': 'RecordName', 'f': 'RecFeatName', 'c': 'CkpdName'}
        ROInfo = {Letter_to_Full[i[0]]: i[1:] for i in element_list}

        HumanName = ROInfo['HumanName']
        
        human = cohort_record_base.Name_to_HRF[HumanName] 
        ROInfo['human'] = DictToObj(OneHuman_Args = human.OneHuman_Args,
                                    **human.OneHuman_Args)


        RecordName = ROInfo['RecordName']
        if (HumanName, RecordName) not in cohort_record_base.Name_to_HRF:
            record = None 
            ROInfo['record'] = None
        else:
            record = cohort_record_base.Name_to_HRF[(HumanName, RecordName)]
            ROInfo['record'] = DictToObj(OneRecord_Args = record.OneRecord_Args,
                                        **record.OneRecord_Args)  

        if 'RecFeatName' in ROInfo:
            ROInfo['RecFeatName'] = ROInfo['RecordName'] + '-' + ROInfo['RecFeatName']
            RecFeatName = ROInfo['RecFeatName']
            
            if (HumanName, RecordName, RecFeatName) not in cohort_record_base.Name_to_HRF:
                recfeat = None 
                ROInfo['recfeat'] = None
            else:
                recfeat = cohort_record_base.Name_to_HRF[(HumanName, RecordName, RecFeatName)]
                ROInfo['recfeat'] = DictToObj(idx2tkn = recfeat.idx2tkn, 
                                              OneRecFeat_Args = recfeat.OneRecFeat_Args,
                                              **recfeat.OneRecFeat_Args) 
        
        if 'CkpdName' in ROInfo:
            CkpdName = ROInfo['CkpdName']
            ROInfo['CkpdInfo'] = Ckpd_to_CkpdObsConfig[CkpdName] 
        ROName_to_ROInfo[ROName] = ROInfo

    return ROName_to_ROInfo


def get_RONameToROData_for_OneCaseExample(case_example, 
                                          ROName_to_ROInfo, 
                                          HRFDirectory, 
                                          RO_to_Cache, 
                                          RCKPD_to_Cache, 
                                          caseset,
                                          ):
    
    # ---- in the special case, you may have different ObsDT. 
    # ---- for current version, we only focus on one ObsDT.
    
    ROName_to_ROData = {}

    # ---------- Cache Part -------------
    ROName_to_HumanROid_ToCal = {}
    ObsDTName = caseset.ObsDTName # case_config['ObsDTName']
    ObsDTValue = case_example[ObsDTName]
    if type(ObsDTValue) == int:
        ObsDTValue = pd.Timestamp(ObsDTValue)
    # print(case_example)

    for ROName, ROInfo in ROName_to_ROInfo.items():

        # -------------------------- get_ROid
        # 1. get HUMANid and ROid
        HumanName = ROInfo['HumanName']
        human = ROInfo['human']
        HumanID = human.HumanID
        HumanIDValue = case_example[HumanID]
        HUMANid = (HumanName, HumanIDValue)
        ROid = (ROName, ObsDTValue) # to ROName
        # -------------------------- get_ROid

        # -------- check whether RO_to_Cache has ROData or not. 
        if HUMANid not in RO_to_Cache:
            RO_to_Cache[HUMANid] = {}

        if ROid in RO_to_Cache[HUMANid]:
            ROData = RO_to_Cache[HUMANid][ROid]
            ROName_to_ROData[ROName] = ROData
        else:
            ROName_to_HumanROid_ToCal[ROName] = HUMANid, ROid

    for ROName, HumanROid in ROName_to_HumanROid_ToCal.items():
        ROInfo = ROName_to_ROInfo[ROName]
        HUMANid, ROid = HumanROid
        
        # -------- method 2: calculate it from the RFInfo for this human.
        RFInfo = HRFDirectory[(HumanName, HumanIDValue)]
        # print(RFInfo)

        # -------- get ds_Rec
        RecordName = ROInfo['RecordName']
        ds_RecAttr = RFInfo[RecordName]
        if 'RecFeatName' in ROInfo:
            RecFeatName = ROInfo['RecFeatName']
            ds_RecFeat = RFInfo[(RecordName, RecFeatName)]
            ds_Rec = ds_RecFeat
        else:
            ds_Rec = ds_RecAttr


        # -------- whether Ckpd is used
        if 'CkpdName' in ROInfo and ds_Rec is not None:
            CkpdName = ROInfo['CkpdName']

            # you will have a long long key.
            RCKPDid = (RecordName, ObsDTValue, CkpdName)
            # --------- get idx_s, idx_e --------

            if HUMANid not in RCKPD_to_Cache:
                RCKPD_to_Cache[HUMANid] = {}

            if RCKPDid in RCKPD_to_Cache[HUMANid]: # TODO
                # ------- method 1: check from RCkpd_to_Cache
                idx_s, idx_e = RCKPD_to_Cache[HUMANid][RCKPDid]

            else:
                # ------- method 2: calculate from begining.
                CkpdInfo = ROInfo['CkpdInfo']
                dates = RFInfo[(RecordName, 'dates')]
                # ----------------
                # RecDT = 'DT_s'
                # assert len(dates) == len(ds_RecAttr)
                # assert dates[0] == ds_RecAttr[0][RecDT].isoformat()
                # assert dates[-1] == ds_RecAttr[-1][RecDT].isoformat()
                # ----------------
                DistStartToPredDT = CkpdInfo['DistStartToPredDT']
                DistEndToPredDT   = CkpdInfo['DistEndToPredDT']
                TimeUnit = CkpdInfo['TimeUnit']

                # print('ObsDTValue', ObsDTValue)
                DT_s = (ObsDTValue + pd.to_timedelta(DistStartToPredDT, unit = TimeUnit)).isoformat()
                DT_e = (ObsDTValue + pd.to_timedelta(DistEndToPredDT,   unit = TimeUnit)).isoformat()  
                # print(DT_s, DT_e)  
                # print(dates)
                idx_s = find_timelist_index(dates, DT_s)
                idx_e = find_timelist_index(dates, DT_e)
                RCKPD_to_Cache[HUMANid][RCKPDid] = (idx_s, idx_e)
            
            if type(ds_Rec) == pd.DataFrame:
                ROData = ds_Rec.iloc[idx_s:idx_e] if idx_s != idx_e else None
            else:
                ROData = ds_Rec.select(range(idx_s, idx_e)) if idx_s != idx_e else None # R_{ij}^{ckpd, name, fld}
        else:
            ROData = ds_Rec

        RO_to_Cache[HUMANid][ROid] = ROData
        ROName_to_ROData[ROName] = ROData

    return ROName_to_ROData

