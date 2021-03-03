"""
created by PyCharm
date: 2021/1/3
time: 16:58
user: wkc
"""

import pandas as pd

from_path = './最终版收集数据 2020.12.30.xlsx'
to_path = './digital_data_20210118.csv'


def data_process(sheet_name):
    data = data_origin[sheet_name]
    print('sheet: %s; size: %s' % (sheet_name, data.shape))
    # 人口学特征 入院时grace评分
    data_1 = special_type_data_process(sheet_name, 0, 21)
    data_1 = data_1.drop(columns=['得分'])

    # 既往史
    data_2 = pd.DataFrame()
    disease_history_dict = {'高血压病史': 22, '糖尿病病史': 26, '吸烟史': 30, '饮酒史': 35, '心肌梗死史': 40, '血运重建史': 42,
                            '心力衰竭史': 43, '脑血管病史': 44, '呼吸系统病史': 47, '肾功能不全': 50, '早发心血管疾病家族史': 54,
                            '心律失常病史': 55, '颈动脉斑块(有无斑块)左侧': 58, '颈动脉斑块(有无斑块)右侧': 59,
                            '颈动脉斑块(有无斑块)双侧': 60, '临床诊断': 61}
    for disease, column in disease_history_dict.items():
        data_2[disease] = data.iloc[:, column]
    data_2 = data_2.drop(index=[0, 1, 2])

    # 入院后查的化验指标
    data_3 = data.iloc[:, [i for i in range(67, 83)]]
    drugs = data_3.iloc[[1, 2]].values[0]
    units = data_3.iloc[[1, 2]].values[1]
    new_columns = []
    for i in range(len(drugs)):
        new_columns.append(drugs[i].strip() + '(' + units[i].strip() + ')')
    data_3.columns = new_columns
    data_3 = data_3.drop(index=[0, 1, 2])

    # 心电图 心脏超声
    data_4 = special_type_data_process(sheet_name, 94, 100)

    # 时间是否推迟
    data_5 = data.iloc[:, [i for i in range(103, 104)]]
    data_5 = data_5.drop(index=[0, 1, 2])

    # 出院带药
    data_6 = special_type_data_process(sheet_name, 104, 113)

    # 随访预后
    data_7 = pd.DataFrame()
    after_recover_dict = {'失访': 121, '已故': 123, '心源性死亡': 125, '新发/再发心肌梗死': 133}
    for item, column in after_recover_dict.items():
        data_7[item] = data.iloc[:, column]
    data_7 = data_7.drop(index=[0, 1, 2])

    data_sheet = pd.concat([data_1, data_2, data_3, data_4, data_5, data_6, data_7], axis=1)
    print('sheet: %s; 处理后size: %s' % (sheet_name, data_sheet.shape))

    return data_sheet


def special_type_data_process(sheet_name, start, end):
    data = data_origin[sheet_name]
    data_partial = data.iloc[:, [i for i in range(start, end)]]
    data_partial.columns = data_partial.iloc[1].values
    data_partial = data_partial.drop(index=[0, 1, 2])
    return data_partial


if __name__ == '__main__':
    data_origin = pd.read_excel(from_path, sheet_name=None)
    data_total = pd.DataFrame()
    for key, values in data_origin.items():
        data_sheet = data_process(key)
        # data_sheet.to_csv('./' + key + '.csv')
        data_total = pd.concat([data_total, data_sheet], axis=0)

    print(data_total.shape)
    data_total.to_csv(to_path, index=False)
