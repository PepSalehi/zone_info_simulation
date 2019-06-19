#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:04:34 2019

@author: peyman
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import geopandas as gpd


zone_shp = gpd.read_file("./Data/taxi_zones/taxi_zones.shp")

report = m.get_service_rate_per_zone()
report.loc[:,'Zone_id'] = report.index 
merged = zone_shp.merge(report, left_on='LocationID', right_on='Zone_id')


ax = merged.dropna().plot(column="LOS", cmap='Blues', figsize=(12, 16),
                   scheme='equal_interval', k=9, legend=True)


ax = merged.dropna().plot(column="total", cmap='Blues', figsize=(12, 16),
                   scheme='equal_interval', k=9, legend=True)
#ax = merged.dropna().plot(column="Zone_id", cmap='Blues', figsize=(12, 16))