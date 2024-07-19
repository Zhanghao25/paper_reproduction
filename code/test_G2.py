# -*- coding: utf-8 -*-

import pulp as pl
import itertools
import numpy as np
import pandas as pd

df_s9 = pd.read_csv('df_s9.csv')
regions = list(df_s9["Regions"].unique())
crops = list(df_s9["Crop"].unique())

yield_per_ha = {}
net_profit = {}
harvested_area = {}
green_water = {}
blue_water = {}
ghg_emissions = {}
nitrogen_use = {}
phosphorus_use = {}
potassium_use = {}
pesticides_use = {}

for region in regions:
    region_data = df_s9[df_s9["Regions"] == region].set_index("Crop")
    yield_per_ha[region] = region_data["Yield (kg/ha)"].to_dict()
    net_profit[region] = region_data["Net Profit (US$/ha)"].to_dict()
    harvested_area[region] = df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"].sum()
    green_water[region] = region_data["Green Water (m3/ha)"].to_dict()
    blue_water[region] = region_data["Blue Water (m3/ha)"].to_dict()
    ghg_emissions[region] = region_data["GHGs (kg eq CO2/ha)"].to_dict()
    nitrogen_use[region] = region_data["Nitrogen (kg N/ha)"].to_dict()
    phosphorus_use[region] = region_data["Phosphorus (kg P2O5/ha)"].to_dict()
    potassium_use[region] = region_data["Potash (kg K2O/ha)"].to_dict()
    pesticides_use[region] = region_data["Pesticides (kg/ha)"].to_dict()
    
    
# 定义线性规划问题
lp2 = pl.LpProblem("Maximize_Farmers_Income_G2", pl.LpMaximize)
x = pl.LpVariable.dicts("Proportion", (regions, crops), lowBound=0, upBound=1.0)

# 目标函数不变:仍为最大化农民收入
lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * net_profit[region][crop]
                 for region in regions for crop in crops if crop in yield_per_ha[region])
# 约束条件1：生产量约束
for crop in crops:
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * yield_per_ha[region][crop]
                    for region in regions if crop in yield_per_ha[region]) >= df_s9[df_s9["Crop"] == crop]["Total Production (kg)"].sum()
# 约束条件2：收入约束
for region in regions:
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * net_profit[region][crop]
                    for crop in crops if crop in net_profit[region]) >= df_s9[df_s9["Regions"] == region]["Total income (US$)"].sum()
# 约束条件3：每个地区x的和为1
for region in regions:
    lp2 += pl.lpSum(x[region][crop] for crop in crops if crop in yield_per_ha[region]) <= 1

# 计算每个指标的全国总使用量
total_green_water = sum(df_s9["Green Water (m3/ha)"] * df_s9["Harvested Area (ha)"])
total_blue_water = sum(df_s9["Blue Water (m3/ha)"] * df_s9["Harvested Area (ha)"])
total_ghg_emissions = sum(df_s9["GHGs (kg eq CO2/ha)"] * df_s9["Harvested Area (ha)"])
total_nitrogen_use = sum(df_s9["Nitrogen (kg N/ha)"] * df_s9["Harvested Area (ha)"])
total_phosphorus_use = sum(df_s9["Phosphorus (kg P2O5/ha)"] * df_s9["Harvested Area (ha)"])
total_potassium_use = sum(df_s9["Potash (kg K2O/ha)"] * df_s9["Harvested Area (ha)"])
total_pesticides_use = sum(df_s9["Pesticides (kg/ha)"] * df_s9["Harvested Area (ha)"])

lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * green_water[region][crop]
                for region in regions for crop in crops if crop in green_water[region]) <= total_green_water, "Total_Green_Water"
lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * blue_water[region][crop]
                for region in regions for crop in crops if crop in blue_water[region]) <= total_blue_water, "Total_Blue_Water"
lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * ghg_emissions[region][crop]
                for region in regions for crop in crops if crop in ghg_emissions[region]) <= total_ghg_emissions, "Total_GHGs"
lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * nitrogen_use[region][crop]
                for region in regions for crop in crops if crop in nitrogen_use[region]) <= total_nitrogen_use, "Total_Nitrogen_Use"
lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * phosphorus_use[region][crop]
                for region in regions for crop in crops if crop in phosphorus_use[region]) <= total_phosphorus_use, "Total_Phosphorus_Use"
lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * potassium_use[region][crop]
                for region in regions for crop in crops if crop in potassium_use[region]) <= total_potassium_use, "Total_Potassium_Use"
lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * pesticides_use[region][crop]
                for region in regions for crop in crops if crop in pesticides_use[region]) <= total_pesticides_use, "Total_Pesticides_Use"


for region in regions:
    # 计算原始区域级别的总量
    original_income = df_s9[df_s9["Regions"] == region]["Total income (US$)"].sum()
    original_green_water = sum(df_s9[df_s9["Regions"] == region]["Green Water (m3/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
    original_blue_water = sum(df_s9[df_s9["Regions"] == region]["Blue Water (m3/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
    original_ghg_emissions = sum(df_s9[df_s9["Regions"] == region]["GHGs (kg eq CO2/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
    original_nitrogen_use = sum(df_s9[df_s9["Regions"] == region]["Nitrogen (kg N/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
    original_phosphorus_use = sum(df_s9[df_s9["Regions"] == region]["Phosphorus (kg P2O5/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
    original_potassium_use = sum(df_s9[df_s9["Regions"] == region]["Potash (kg K2O/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
    original_pesticides_use = sum(df_s9[df_s9["Regions"] == region]["Pesticides (kg/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])

    # 添加区域级别的约束条件
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * net_profit[region][crop]
                    for crop in crops if crop in net_profit[region]) >= original_income, f"Income_Constraint_{region}"
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * green_water[region][crop]
                    for crop in crops if crop in green_water[region]) <= original_green_water, f"Green_Water_Constraint_{region}"
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * blue_water[region][crop]
                    for crop in crops if crop in blue_water[region]) <= original_blue_water, f"Blue_Water_Constraint_{region}"
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * ghg_emissions[region][crop]
                    for crop in crops if crop in ghg_emissions[region]) <= original_ghg_emissions, f"GHG_Constraint_{region}"
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * nitrogen_use[region][crop]
                    for crop in crops if crop in nitrogen_use[region]) <= original_nitrogen_use, f"Nitrogen_Constraint_{region}"
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * phosphorus_use[region][crop]
                    for crop in crops if crop in phosphorus_use[region]) <= original_phosphorus_use, f"Phosphorus_Constraint_{region}"
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * potassium_use[region][crop]
                    for crop in crops if crop in potassium_use[region]) <= original_potassium_use, f"Potassium_Constraint_{region}"
    lp2 += pl.lpSum(x[region][crop] * harvested_area[region] * pesticides_use[region][crop]
                    for crop in crops if crop in pesticides_use[region]) <= original_pesticides_use, f"Pesticides_Constraint_{region}"


lp2.solve()




print("Status:", pl.LpStatus[lp2.status])
for name, constraint in lp2.constraints.items():
    lhs_value = pl.value(constraint)
    print(f"{name}: {lhs_value} Satisfied = {constraint.valid()}")