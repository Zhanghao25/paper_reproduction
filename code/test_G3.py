import pulp as pl
import itertools
import numpy as np
import pandas as pd

df_s9 = pd.read_csv('df_s9.csv')
regions = list(df_s9["Regions"].unique())
crops = list(df_s9["Crop"].unique())

# 定义字典来存储各个指标的数据
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

# 定义权重组合
weight_combinations = list(itertools.product([0, 1], repeat=7))

# 定义指标数据的列表
indicators = [blue_water, green_water, ghg_emissions, phosphorus_use, nitrogen_use, potassium_use, pesticides_use]
indicator_columns = ["Blue Water (m3/ha)", "Green Water (m3/ha)", "GHGs (kg eq CO2/ha)", "Phosphorus (kg P2O5/ha)", "Nitrogen (kg N/ha)", "Potash (kg K2O/ha)", "Pesticides (kg/ha)"]

# 存储结果
all_results = []

# 计算原始指标总量（用于计算改进值）
original_totals = []
for column in indicator_columns:
    original_total = sum(df_s9[column] * df_s9["Harvested Area (ha)"])
    original_totals.append(original_total)

for weights in weight_combinations:
    # 创建线性规划问题
    lp = pl.LpProblem("MultiObjectiveOptimization", pl.LpMinimize)

    # 定义决策变量
    x = pl.LpVariable.dicts("Proportion", (regions, crops), lowBound=0, upBound=1.0)

    # 构建目标函数
    objective = pl.lpSum(weights[i] * pl.lpSum(
        x[region][crop] * harvested_area[region] * indicators[i][region][crop]
        for region in regions for crop in crops if crop in indicators[i][region])
        for i in range(7))
    lp += objective

    # 添加约束条件
    # 约束条件1：生产量约束
    for crop in crops:
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * yield_per_ha[region][crop]
                        for region in regions if crop in yield_per_ha[region]) >= df_s9[df_s9["Crop"] == crop]["Total Production (kg)"].sum(), f"Production_{crop}"

    # 约束条件2：收入约束
    for region in regions:
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * net_profit[region][crop]
                        for crop in crops if crop in net_profit[region]) >= df_s9[df_s9["Regions"] == region]["Total income (US$)"].sum(), f"Income_{region}"

    # 约束条件3：每个地区x的和为1
    for region in regions:
        lp += pl.lpSum(x[region][crop] for crop in crops if crop in yield_per_ha[region]) <= 1, f"Region_{region}_Total"

    # 总指标约束
    lp += pl.lpSum(x[region][crop] * harvested_area[region] * blue_water[region][crop]
                    for region in regions for crop in crops if crop in blue_water[region]) <= sum(df_s9["Blue Water (m3/ha)"] * df_s9["Harvested Area (ha)"]), "Total_Blue_Water"
    lp += pl.lpSum(x[region][crop] * harvested_area[region] * green_water[region][crop]
                    for region in regions for crop in crops if crop in green_water[region]) <= sum(df_s9["Green Water (m3/ha)"] * df_s9["Harvested Area (ha)"]), "Total_Green_Water"
    lp += pl.lpSum(x[region][crop] * harvested_area[region] * ghg_emissions[region][crop]
                    for region in regions for crop in crops if crop in ghg_emissions[region]) <= sum(df_s9["GHGs (kg eq CO2/ha)"] * df_s9["Harvested Area (ha)"]), "Total_GHGs"
    lp += pl.lpSum(x[region][crop] * harvested_area[region] * phosphorus_use[region][crop]
                    for region in regions for crop in crops if crop in phosphorus_use[region]) <= sum(df_s9["Phosphorus (kg P2O5/ha)"] * df_s9["Harvested Area (ha)"]), "Total_Phosphorus_Use"
    lp += pl.lpSum(x[region][crop] * harvested_area[region] * nitrogen_use[region][crop]
                    for region in regions for crop in crops if crop in nitrogen_use[region]) <= sum(df_s9["Nitrogen (kg N/ha)"] * df_s9["Harvested Area (ha)"]), "Total_Nitrogen_Use"
    lp += pl.lpSum(x[region][crop] * harvested_area[region] * potassium_use[region][crop]
                    for region in regions for crop in crops if crop in potassium_use[region]) <= sum(df_s9["Potash (kg K2O/ha)"] * df_s9["Harvested Area (ha)"]), "Total_Potassium_Use"
    lp += pl.lpSum(x[region][crop] * harvested_area[region] * pesticides_use[region][crop]
                    for region in regions for crop in crops if crop in pesticides_use[region]) <= sum(df_s9["Pesticides (kg/ha)"] * df_s9["Harvested Area (ha)"]), "Total_Pesticides_Use"

    for region in regions:
        # 计算原始区域级别的总量
        original_income = df_s9[df_s9["Regions"] == region]["Total income (US$)"].sum()
        original_blue_water = sum(df_s9[df_s9["Regions"] == region]["Blue Water (m3/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
        original_green_water = sum(df_s9[df_s9["Regions"] == region]["Green Water (m3/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
        original_ghg_emissions = sum(df_s9[df_s9["Regions"] == region]["GHGs (kg eq CO2/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
        original_nitrogen_use = sum(df_s9[df_s9["Regions"] == region]["Nitrogen (kg N/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
        original_phosphorus_use = sum(df_s9[df_s9["Regions"] == region]["Phosphorus (kg P2O5/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
        original_potassium_use = sum(df_s9[df_s9["Regions"] == region]["Potash (kg K2O/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])
        original_pesticides_use = sum(df_s9[df_s9["Regions"] == region]["Pesticides (kg/ha)"] * df_s9[df_s9["Regions"] == region]["Harvested Area (ha)"])

        # 添加区域级别的约束条件
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * net_profit[region][crop]
                        for crop in crops if crop in net_profit[region]) >= original_income, f"Income_Constraint_{region}"
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * blue_water[region][crop]
                        for crop in crops if crop in blue_water[region]) <= original_blue_water, f"Blue_Water_Constraint_{region}"
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * green_water[region][crop]
                        for crop in crops if crop in green_water[region]) <= original_green_water, f"Green_Water_Constraint_{region}"
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * ghg_emissions[region][crop]
                        for crop in crops if crop in ghg_emissions[region]) <= original_ghg_emissions, f"GHG_Constraint_{region}"
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * nitrogen_use[region][crop]
                        for crop in crops if crop in nitrogen_use[region]) <= original_nitrogen_use, f"Nitrogen_Constraint_{region}"
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * phosphorus_use[region][crop]
                        for crop in crops if crop in phosphorus_use[region]) <= original_phosphorus_use, f"Phosphorus_Constraint_{region}"
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * potassium_use[region][crop]
                        for crop in crops if crop in potassium_use[region]) <= original_potassium_use, f"Potassium_Constraint_{region}"
        lp += pl.lpSum(x[region][crop] * harvested_area[region] * pesticides_use[region][crop]
                        for crop in crops if crop in pesticides_use[region]) <= original_pesticides_use, f"Pesticides_Constraint_{region}"

    # 求解问题
    lp.solve()

    # 计算每个指标的改进值和方差
    improvements = [1 - pl.value(pl.lpSum(
        x[region][crop] * harvested_area[region] * indicators[i][region][crop]
        for region in regions for crop in crops if crop in indicators[i][region])) / original_totals[i] for i in range(7)]
    avg_improvement = np.mean(improvements)
    var_improvement = np.var(improvements)

    # 保存当前权重组合和改进值
    result = {
        "weights": weights,
        "avg_improvement": avg_improvement,
        "var_improvement": var_improvement,
        "improvements": improvements,
        "objective_value": avg_improvement / var_improvement if var_improvement != 0 else 0
    }

    # 保存各个区域和作物的比例
    for region in regions:
        for crop in crops:
            result[f"{region}_{crop}_proportion"] = pl.value(x[region][crop])

    all_results.append(result)

# 将结果保存到数据框
df_results = pd.DataFrame(all_results)
print(df_results)
