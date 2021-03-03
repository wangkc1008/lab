"""
created by PyCharm
date: 2021/1/25
time: 10:05
user: wkc
"""

import pandas as pd
import numpy as np
import random

digital_path = './data/digital_data_20200118.csv'
text_path = './data/total_vector_20210116_095132.csv'

digital_data = pd.read_csv(digital_path, encoding='ANSI')
text_data = pd.read_csv(text_path)
print('数值数据', digital_data.shape)
print('文本数据', text_data.shape)

digital_data = digital_data[digital_data['失访'].isna()]

digital_data['身高（cm）'] = digital_data['身高（cm）'].replace(['平车', '轮椅', 'pc', '?', np.NAN], 0)
digital_data['体重（kg）'] = digital_data['体重（kg）'].replace(['平车', '轮椅', 'pc', '?', np.NAN], 0)
# digital_data['身高（cm）'] = digital_data['身高（cm）'].apply(float)
# digital_data['体重（kg）'] = digital_data['体重（kg）'].apply(float)

man_height_mean = round(digital_data[(digital_data['性别'] == 1) & (digital_data['身高（cm）'].astype('float') != 0)]['身高（cm）'].astype('float').mean())
woman_height_mean = round(digital_data[(digital_data['性别'] == 2) & (digital_data['身高（cm）'].astype('float') != 0)]['身高（cm）'].astype('float').mean())
man_weight_mean = round(digital_data[(digital_data['性别'] == 1) & (digital_data['体重（kg）'].astype('float') != 0)]['体重（kg）'].astype('float').mean(), 1)
woman_weight_mean = round(digital_data[(digital_data['性别'] == 2) & (digital_data['体重（kg）'].astype('float') != 0)]['体重（kg）'].astype('float').mean(), 1)

df_height = digital_data['身高（cm）']
man_height_index = (digital_data['性别'] == 1) & (digital_data['身高（cm）'].astype('float') == 0)
woman_height_index = (digital_data['性别'] == 2) & (digital_data['身高（cm）'].astype('float') == 0)
df_height[man_height_index] = df_height[man_height_index].replace([0], man_height_mean)
df_height[woman_height_index] = df_height[woman_height_index].replace([0], woman_height_mean)
digital_data['身高（cm）'] = df_height

df_weight = digital_data['体重（kg）']
man_weight_index = (digital_data['性别'] == 1) & (digital_data['体重（kg）'].astype('float') == 0)
woman_weight_index = (digital_data['性别'] == 2) & (digital_data['体重（kg）'].astype('float') == 0)
df_weight[man_weight_index] = df_weight[man_weight_index].replace([0], man_weight_mean)
df_weight[woman_weight_index] = df_weight[woman_weight_index].replace([0], woman_weight_mean)
digital_data['体重（kg）'] = df_weight

digital_data['cTnT'] = 0
cTnT = digital_data['cTnT']
cTnT[(digital_data['临床诊断'] == 2) | (digital_data['临床诊断'].isna())] = 1
digital_data['cTnT'] = cTnT

jigan = digital_data['肌酐（μmol/L）/88,4']
jigan[(digital_data['肌酐（μmol/L）/88,4'].isna()) | (digital_data['肌酐（μmol/L）/88,4'] == '？')] = 55
digital_data['肌酐（μmol/L）/88,4'] = jigan

digital_data['Killip分级'] = digital_data['Killip分级'].replace({'k2': 'K2', 'N1`': 'N1', 'Ｎ2～3': 'N2', 'NYHAⅠ': 'N1', 'NYHAⅡ': 'N2', 'NYHAⅢ': 'N3', 'NYHA I级': 'N1', 'NYHA II级': 'N2', 'k1': 'K1', ' N2': 'N2'})
Killip_index = sorted(list(set(digital_data['Killip分级'].values)))
index = 0
Killip_set = {}
for item in Killip_index:
    Killip_set[item] = index
    index += 1
digital_data['Killip分级'] = digital_data['Killip分级'].replace(Killip_set)

digital_data['危险级别'] = digital_data['危险级别'].replace({'低危': 0, '中危': 1, '高危': 2, '高危 ': 2})
digital_data['危险因素'] = digital_data['危险因素'].replace(['2.3', '23', '2,3'], 1)
digital_data['高血压病史'] = digital_data['高血压病史'].replace([np.NaN], 0)
digital_data['糖尿病病史'] = digital_data['糖尿病病史'].replace({np.NAN: 0, 3: 1, 9: 1})
digital_data['吸烟史'] = digital_data['吸烟史'].replace([np.NaN], 0)
digital_data['饮酒史'] = digital_data['饮酒史'].replace({np.NAN: 0, '偶有': 1, '适量': 1})
digital_data['心肌梗死史'] = digital_data['心肌梗死史'].replace({np.NAN: 0, '？': 0})
digital_data['血运重建史'] = digital_data['血运重建史'].replace({np.NAN: 0, '溶栓': 4, '1、2': 2})
digital_data['心力衰竭史'] = digital_data['心力衰竭史'].replace({np.NAN: 0, '1、3': 1})
digital_data['脑血管病史'] = digital_data['脑血管病史'].replace({np.NAN: 0})
digital_data['呼吸系统病史'] = digital_data['呼吸系统病史'].replace({np.NAN: 0})
digital_data['肾功能不全'] = digital_data['肾功能不全'].replace({np.NAN: 0, '尿酸高10+年': 1})
heart_family_history = digital_data['早发心血管疾病家族史'].copy()
heart_family_history[~(digital_data['早发心血管疾病家族史'] == '0') & ~(digital_data['早发心血管疾病家族史'] == '1') & ~(digital_data['早发心血管疾病家族史'].isna())] = 1
digital_data['早发心血管疾病家族史'] = heart_family_history
digital_data['早发心血管疾病家族史'] = digital_data['早发心血管疾病家族史'].replace({np.NAN: 0})
digital_data['心律失常病史'] = digital_data['心律失常病史'].replace({np.NAN: 0})

digital_data['CK-MB'] = digital_data['CK-MB'].replace({np.NAN: 0})
digital_data['NT-poBNP(pg/mL)'] = digital_data['NT-poBNP(pg/mL)'].replace({np.NAN: 0})
digital_data['NT-poBNP'] = 0
NT = digital_data['NT-poBNP']
NT[(digital_data['NT-poBNP(pg/mL)'] > 450) & (digital_data['年龄'] < 50)] = 1
NT[(digital_data['NT-poBNP(pg/mL)'] > 900) & (digital_data['年龄'] > 50) & (digital_data['年龄'] < 75)] = 1
NT[(digital_data['NT-poBNP(pg/mL)'] > 1800) & (digital_data['年龄'] < 75)] = 1
digital_data['NT-poBNP'] = NT

for key, value in digital_data['TC(mmol/L)'].isna().items():
    if value:
        digital_data['TC(mmol/L)'].loc[key] = round(random.uniform(3, 5.2), 1)

digital_data['TG(mmol/L)'] = digital_data['TG(mmol/L)'].replace({' ': np.NAN, '1,99': 1.99})
for key, value in digital_data['TG(mmol/L)'].isna().items():
    if value:
        digital_data['TG(mmol/L)'].loc[key] = round(random.uniform(0.3, 1.7), 1) 
        # print(digital_data['TG(mmol/L)'].loc[key])

for key, value in digital_data['LDL-C(mmol/L)'].isna().items():
    if value:
        digital_data['LDL-C(mmol/L)'].loc[key] = round(random.uniform(1, 3.37), 2) 
        # print(digital_data['LDL-C(mmol/L)'].loc[key])

digital_data['FPG(mmol/L)'] = digital_data['FPG(mmol/L)'].replace({'6，48': 6.48})
for key, value in digital_data['FPG(mmol/L)'].isna().items():
    if value:
        digital_data['FPG(mmol/L)'].loc[key] = round(random.uniform(3.9, 6.1), 1) 
        # print(digital_data['FPG(mmol/L)'].loc[key])

digital_data['BUN(mmol/L)'] = digital_data['BUN(mmol/L)'].replace({'4,2': 4.2})

for key, value in digital_data['BUN(mmol/L)'].isna().items():
    if value:
        digital_data['BUN(mmol/L)'].loc[key] = round(random.uniform(2.3, 7.8), 1) 
        # print(digital_data['BUN(mmol/L)'].loc[key])

for key, value in digital_data['CR(μmol/L)'].isna().items():
    if value:
        digital_data['CR(μmol/L)'].loc[key] = int(random.uniform(62, 115))
        # print(digital_data['CR(μmol/L)'].loc[key])

for key, value in digital_data[digital_data['性别'] == 1]['HB(g/L)'].isna().items():
    if value:
        digital_data['HB(g/L)'].loc[key] = int(random.uniform(120, 160)) 
        # print(digital_data['HB(g/L)'].loc[key])
for key, value in digital_data[digital_data['性别'] == 2]['HB(g/L)'].isna().items():
    if value:
        digital_data['HB(g/L)'].loc[key] = int(random.uniform(110, 150)) 
        # print(digital_data['HB(g/L)'].loc[key])

for key, value in digital_data[(digital_data['凝血酶原时间(S)'] == '正常') | (digital_data['凝血酶原时间(S)'].isna())]['凝血酶原时间(S)'].items():
    if value:
        digital_data['凝血酶原时间(S)'].loc[key] = round(random.uniform(8.8, 13.8), 1) 
        # print(digital_data['凝血酶原时间(S)'].loc[key])

for key, value in digital_data[(digital_data['纤维蛋白原(g/L)'] == '正常') | (digital_data['纤维蛋白原(g/L)'].isna())]['纤维蛋白原(g/L)'].items():
    if value:
        digital_data['纤维蛋白原(g/L)'].loc[key] = round(random.uniform(2, 4), 2) 
        # print(digital_data['纤维蛋白原(g/L)'].loc[key])

digital_data['ST-T动态改变'] = digital_data['ST-T动态改变'].replace(['0(短暂抬高）', '0（短暂抬高）', '1（短暂抬高）', '1（一过抬高）'], 1)
digital_data['ST-T动态改变'] = digital_data['ST-T动态改变'].replace(np.nan, 0)
digital_data['ST-T动态改变'] = digital_data['ST-T动态改变'].apply(int)

digital_data['LV(mm)'] = digital_data['LV(mm)'].replace('中下段内径52mm', 52)
for key, value in digital_data['LV(mm)'].isna().items():
    if value:
        digital_data['LV(mm)'].loc[key] = int(random.uniform(45, 50))
        # print(digital_data['LV(mm)'].loc[key])

for key, value in digital_data['LVEF（%）'].isna().items():
    if value:
        digital_data['LVEF（%）'].loc[key] = round(random.uniform(0.5, 0.7), 2)
        # print(digital_data['LVEF（%）'].loc[key])

digital_data['时间是否推迟'] = digital_data['时间是否推迟'].replace(np.nan, 0)
digital_data['ACEI/ARB'] = digital_data['ACEI/ARB'].replace(np.nan, 0)
digital_data['他汀类'] = digital_data['他汀类'].replace({'1.2': 1, '21': 2, '氟伐他汀': 1, '辛伐他汀': 1, '匹伐他汀': 1})
digital_data['他汀类'] = digital_data['他汀类'].apply(float)
digital_data['CCB'] = digital_data['CCB'].replace(np.nan, 0)


df_flag = digital_data[['已故', '心源性死亡', '新发/再发心肌梗死']]
df_flag = df_flag[['已故', '心源性死亡', '新发/再发心肌梗死']].replace(['否', '否，脑出血', '否，脑卒中', '否，癌症', '否，肿瘤', '无'], np.NaN)
df_flag['label'] = 0
label = df_flag['label']
label[~(df_flag['已故'].isna()) | ~(df_flag['心源性死亡'].isna()) | ~(df_flag['新发/再发心肌梗死'].isna())] = 1
df_flag['label'] = label
digital_data['label'] = df_flag['label']

digital_data.drop(columns=['编号', '入院日期', 'BMI', 'MYO(ng/ml)', 'CRP(mg/L)', 'HbA1c(%)', 'APOB(g/L)', '失访', 'CK-MB(ng/ml)', '已故', '心源性死亡', '新发/再发心肌梗死', 'cTnT(ng/l)', '颈动脉斑块(有无斑块)左侧', '颈动脉斑块(有无斑块)右侧', '颈动脉斑块(有无斑块)双侧', '临床诊断', 'NT-poBNP(pg/mL)', '致命性心律失常', '心脏骤停', 'MI机械并发症', '拜阿司匹林'], inplace=True)
digital_data.to_csv('./data/total_digital_20210125.csv', index=False)

text_data.rename(columns={'0': '姓名'}, inplace=True)
total_data = pd.merge(text_data, digital_data, on='姓名', how='inner')
total_data.drop(columns='姓名', inplace=True)
total_data.to_csv('./data/total_data_20210125.csv', index=False)


