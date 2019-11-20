#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 23:23:49 2019

@author: stevenalsheimer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:11:32 2019

@author: stevenalsheimer
"""

import numpy as np
import pandas as pd
import time
from scipy import linalg
import os
import copy
start_time = time.time()

import matplotlib as plt
pd.set_option('display.max_columns', 500)

###Open the left file ###
DFL = pd.read_csv('gh17_reducedR.ptx', sep = ' ', header = None,names=["x", "y", "z", "i"])
row_num = DFL.loc[0,'x']
col_num = DFL.loc[1,'x']

num_points = row_num*col_num
DFL_prime = pd.DataFrame(columns=["x", "y", "z", "i"])

DFL = DFL[10:(int(num_points))]
#DFL = DFL[10:18000]

DFL_shaved = pd.DataFrame(DFL)
indexNames = DFL_shaved[DFL_shaved['x'] == 0 ].index
DFL_shaved.drop(indexNames , inplace=True)
DFL_shaved = DFL_shaved.reset_index(drop=False)

###Open right File####
DFR = pd.read_csv('gh23_reduced.ptx', sep = ' ', header = None,names=["x", "y", "z", "i"])
row_num2 = DFR.loc[0,'x']
col_num2 = DFR.loc[1,'x']

num_points2 = row_num2*col_num2
#DF2 = pd.DataFrame(columns=["x", "y", "z", "i"])

DFR = DFR[10:(int(num_points))]
#DFR = DFR[10:18000]
###Find 1000 points in DFL and closest points in DFR####
left_training_data = DFL_shaved.sample(n=1)#currently set to 10 points
left_training_data = left_training_data.reset_index(drop=False)
S_L_data = left_training_data.copy(deep=True)
#print(left_training_data)   
Threshold_E = 0.12
E_p = 0
iterations = 0
E = 3000
convergenceE = pd.DataFrame(columns=['i','E'])
#while DiffE > Threshold_E:
for i in range(10):
    for i in range(len(left_training_data)):####Find Corresponding Points per iteration here
        x = left_training_data.loc[i,'x']
        y = left_training_data.loc[i,'y']
        z = left_training_data.loc[i,'z']
        dist_best = 2500000
        start_time = time.time()
        for d in range(10,20):
            x2 = DFR.loc[d,'x']
            y2 = DFR.loc[d,'y']
            z2 = DFR.loc[d,'z']
            if x2 == 0 and y2 == 0 and z2 == 0:
                continue
            else:
                dist = (x2-x)**2+(y2-y)**2+(z2-z)**2
                if dist < dist_best:
                    dist_best = dist
                    left_training_data.loc[i,'dist'] = dist
                    left_training_data.loc[i,'xr'] = x2
                    left_training_data.loc[i,'yr'] = y2
                    left_training_data.loc[i,'zr'] = z2
        print("--- %s seconds ---" % (time.time() - start_time))  














