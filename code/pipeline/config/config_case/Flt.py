FltName_to_FltArgs = {
    'FltBasicDemo': [
        ('Gender', '!=', 'U')
    ], 
    
    # 'FltMiniAfBfCGMRecNum':[
    #     ('co.Bf24h_CGM_rn:recnum', '>=', 289),
    #     ('co.Af2h_CGM_rn:recnum',  '>=', 24),
    # ],

    # 'FltMiniAfBfCGM_FoodBf6h_Keep0.25':[
    #     ('co.Bf24h_CGM_info:recnum', '>=', 289),
    #     ('co.Af2h_CGM_info:recnum',  '>=', 24),
    #     ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
    #     ('co.Af2h_CGM_info:ModePercent',   '<=', 0.7),

    #     ('co.Bf24H_FoodInfo:MinToNow', '<=', 6 * 60),
    #     ('co.Bf24H_FoodInfo:recnum', '>=', 1),

    #     ('_keep_ratio', '<=', 0.25),
    # ],



    'FltMiniAfBfCGM_Diet5MinBf6h_3ph':[
        ('co.Bf24h_CGM_info:recnum', '>=', 289),
        ('co.Af2h_CGM_info:recnum',  '>=', 24),
        ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
        ('co.Af2h_CGM_info:ModePercent',   '<=', 0.7),

        ('co.Bf24H_Diet5MinInfo:MinToNow', '<=', 6 * 60),
        ('co.Bf24H_Diet5MinInfo:recnum', '>=', 1),

        ('_keep_ratio', '<=', 3/12), # 3ph: 3 per 12 5-minutes in one hour. 
    ],


    'FltMiniAfBfCGM_Diet5MinBf5min':[
        ('co.Bf24h_CGM_info:recnum', '>=', 289),
        ('co.Af2h_CGM_info:recnum',  '>=', 24),
        ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
        ('co.Af2h_CGM_info:ModePercent',   '<=', 0.7),

        ('co.Bf24H_Diet5MinInfo:MinToNow', '<=', 3),
        ('co.Bf24H_Diet5MinInfo:recnum', '>=', 1),

        # ('_keep_ratio', '<=', 0.25),
    ],


    'FltMiniBf24h_WholeDay_WithFood':[
        ('co.Bf24h_CGM_info:recnum', '>=', 289),
        ('co.Af2h_CGM_info:recnum',  '>=', 24),
        ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
        ('co.Af2h_CGM_info:ModePercent',   '<=', 0.7),

        # ('co.Bf24H_Diet5MinInfo:MinToNow', '<=', 3),
        ('co.Bf24H_Diet5MinInfo:recnum', '>=', 1),
        ('ObsDT', 'contains', '00:00:00'),
        # ('_keep_ratio', '<=', 0.25),
    ],





    'FltMiniAfBfCGM_FoodBf6h_3ph':[
        ('co.Bf24h_CGM_info:recnum', '>=', 289),
        ('co.Af2h_CGM_info:recnum',  '>=', 24),
        ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
        ('co.Af2h_CGM_info:ModePercent',   '<=', 0.7),

        ('co.Bf24H_FoodInfo:MinToNow', '<=', 6 * 60),
        ('co.Bf24H_FoodInfo:recnum', '>=', 1),

        ('_keep_ratio', '<=', 3/12),
    ],


    'FltMiniAfBfCGM_FoodBf5min':[
        ('co.Bf24h_CGM_info:recnum', '>=', 289),
        ('co.Af2h_CGM_info:recnum',  '>=', 24),
        ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
        ('co.Af2h_CGM_info:ModePercent',   '<=', 0.4),

        ('co.Bf24H_FoodInfo:MinToNow', '<=', 5),
        ('co.Bf24H_FoodInfo:recnum', '>=', 1),

        # ('_keep_ratio', '<=', 0.25),
    ],


    # 'FltMiniAfBfCGM_FoodBf5min':[
    #     ('co.Bf24h_CGM_info:recnum', '>=', 289),
    #     ('co.Af2h_CGM_info:recnum',  '>=', 24),
    #     ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
    #     ('co.Af2h_CGM_info:ModePercent',   '<=', 0.4),

    #     ('co.Bf24H_FoodInfo:MinToNow', '<=', 5),
    #     ('co.Bf24H_FoodInfo:recnum', '>=', 1),

    #     # ('_keep_ratio', '<=', 0.25),
    # ],
    


    'FltMiniAfBfCGM_32h_Keep24in288':[
        ('co.Bf24h_CGM_info:recnum', '>=', 289),
        ('co.Af2h_CGM_info:recnum',  '>=', 24),
        ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
        ('co.Af2h_CGM_info:ModePercent',   '<=', 0.4),
        ('_keep_ratio', '<=', 24/288),
    ],

    'FltMiniBfAfCGMRecInfo':[
        ('co.Bf24h_CGM_info:recnum', '>=', 289),
        ('co.Af2h_CGM_info:recnum',  '>=', 24),
        ('co.Bf24h_CGM_info:ModePercent',  '<=', 0.4),
        ('co.Af2h_CGM_info:ModePercent',   '<=', 0.4),
        # ('co.Af2h_CGM_info:recnum',  '>=', 24),
    ],


    'FltWithBf24hAf2HAf2Ht8H-MEDAL-OR': [
        ['co.Bf24H_MedAdmin_recnum:recnum', '>=', 1],
        # ['co.Bf24H_ImpMed_recnum:recnum', '>=' , 1],
        ['co.Bf24H_Cmt_recnum:recnum', '>=', 1],
        ['co.Bf24H_Lesson_recnum:recnum', '>=', 1],
        ['co.Bf24H_Food_recnum:recnum', '>=', 1],
        ['co.Bf24H_Carb_recnum:recnum', '>=', 1],
        ['co.Bf24H_Exercise_recnum:recnum', '>=', 1],
        ['co.Bf24H_Sleep_recnum:recnum', '>=', 1],
        ['co.Bf24H_Step_recnum:recnum', '>=', 1],
        ['co.Bf24H_Weight_recnum:recnum', '>=', 1],
        ['co.Bf24H_BP_recnum:recnum', '>=', 1],


        ['co.Af2H_MedAdmin_recnum:recnum', '>=', 1],
        # ['co.Af2H_ImpMed_recnum:recnum', '>=' , 1],
        ['co.Af2H_Cmt_recnum:recnum', '>=', 1],
        ['co.Af2H_Lesson_recnum:recnum', '>=', 1],
        ['co.Af2H_Food_recnum:recnum', '>=', 1],
        ['co.Af2H_Carb_recnum:recnum', '>=', 1],
        ['co.Af2H_Exercise_recnum:recnum', '>=', 1],
        ['co.Af2H_Sleep_recnum:recnum', '>=', 1],
        ['co.Af2H_Step_recnum:recnum', '>=', 1],
        ['co.Af2H_Weight_recnum:recnum', '>=', 1],
        ['co.Af2H_BP_recnum:recnum', '>=', 1],

        ['co.Af2Ht8H_MedAdmin_recnum:recnum', '>=', 1],
        # ['co.Af2H_ImpMed_recnum:recnum', '>=' , 1],
        ['co.Af2Ht8H_Cmt_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_Lesson_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_Food_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_Carb_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_Exercise_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_Sleep_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_Step_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_Weight_recnum:recnum', '>=', 1],
        ['co.Af2Ht8H_BP_recnum:recnum', '>=', 1],

    ],

    'FltWithBf24hFood': [
        ['co.Bf24H_Food_recnum:recnum', '>=', 1],
        ['co.Bf24H_Carb_recnum:recnum', '>=', 1],
    ],
}