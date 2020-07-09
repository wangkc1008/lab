"""
created by PyCharm
date: 2020/7/9
time: 15:14
user: wkc
"""

from testPostgreSQL import PostgreSQLOperate
import numpy
import pandas
import csv

hadm_id_file = './hadm_id_20200709'
admission_information_file = './admission_information_20200709'
demographics_file = './demographics_20200709'
treatment_information_file = './treatment_information_20200709'
lab_value_file = './lab_value_20200709'
chart_value_file = './chart_value_20200709'
item_id_list = [50903,50905,50906,50904,50907,51000,50861,50878,3728,50862,51464,781,50912,50927,50954,1539,220650,50963,227444,50910,50911,50909,50945,851,227429,46362,44441,1522,44855,1523,44711,1535,41956,1536,3808,837,226534,3744,3745,1529,226537,813,3761,220545,226540,227017,226761,226762,814,220228,1127,861,4200,1542,227062,227063,220051,227242,8368,8446,225310,8445,8364,228151,8440,8555,224643,8441,220180,227243,51,484,225309,482,6,220050,228152,442,6701,224167,455,220046,618,220210]


def write_to_file(filename, label, res):
    with open(filename, mode='w+', encoding='utf-8', newline='') as f:
        write = csv.writer(f, dialect='excel')
        write.writerow(label)
        for row in res:
            write.writerow(row)
    f.close()


class DataProcess:

    def __init__(self):
        self.post = PostgreSQLOperate('127.0.0.1', 5432)
        self.post.connect('postgres', 'postgres', 'mimic')

    def get_hadm_id(self):
        """
        获取hadm_id
        :return:
        """
        res = self.post.select("select distinct(hadm_id),subject_id from diagnoses_icd where icd9_code like '410%' ")
        hadm_id_str = ','.join([str(row[0]) for row in res])
        subject_id_str = ','.join([str(row[1]) for row in res])
        write_to_file(hadm_id_file, ['hadm_id', 'subject_id'], res)
        print('hadm_id 写入完成: %d' % len(res))
        return hadm_id_str, subject_id_str

    def get_admission_information(self, hadm_id_str):
        """
        获取住院信息
        :param hadm_id_list:
        :return:
        """
        res = self.post.select("select hadm_id,admittime,dischtime,deathtime,discharge_location,religion,marital_status,ethnicity,diagnosis from admissions where hadm_id in (" + hadm_id_str + ")")
        write_to_file(admission_information_file, ['hadm_id', 'admittime', 'dischtime','deathtime','discharge_location','religion','marital_status','ethnicity','diagnosis'], res)
        print('admission_information 写入完成: %d' % len(res))
        return res

    def get_demographics(self, subject_id_str):
        """
        获取人口统计学信息
        :param hadm_id_str:
        :return:
        """
        res = self.post.select("select subject_id,gender,dob,dod,expire_flag from patients where subject_id in (" + subject_id_str + ")")
        write_to_file(demographics_file, ['subject_id', 'gender', 'dob','dod','expire_flag'], res)
        print('demographics 写入完成: %d' % len(res))
        return res

    def get_treatment_information(self, hadm_id_str):
        """
        获取治疗信息
        :param hadm_id_str:
        :return:
        """
        res = self.post.select("select hadm_id,description from drgcodes where hadm_id in (" + hadm_id_str + ")")
        write_to_file(treatment_information_file, ['hadm_id', 'descriptin'], res)
        print('treatment_information 写入完成: %d' % len(res))
        return res

    def get_lab_values(self, hadm_id_str):
        """
        获取检查信息
        :param hadm_id_str:
        :return:
        """
        res = self.post.select("select hadm_id,itemid,value from labevents where hadm_id in (" + hadm_id_str + ") and itemid in (" + ','.join([str(item) for item in item_id_list]) + ");")
        write_to_file(lab_value_file, ['hadm_id', 'itemid', 'value'], res)
        print('lab_value 写入完成: %d' % len(res))
        return res

    def get_chart_values(self, hadm_id_str):
        """
        获取检查信息
        :param hadm_id_str:
        :return:
        """
        res = self.post.select("select hadm_id,itemid,value from chartevents where hadm_id in (" + hadm_id_str + ") and itemid in (" + ','.join([str(item) for item in item_id_list]) + ");")
        write_to_file(chart_value_file, ['hadm_id', 'itemid', 'value'], res)
        print('chart_value 写入完成: %d' % len(res))
        return res


if __name__ == '__main__':
    dp_obj = DataProcess()
    hadm_id_str, subject_id_str = dp_obj.get_hadm_id()
    admission_information = dp_obj.get_admission_information(hadm_id_str)
    demographics = dp_obj.get_demographics(subject_id_str)
    treatment_information = dp_obj.get_treatment_information(hadm_id_str)
    lab_values = dp_obj.get_lab_values(hadm_id_str)
    chart_values = dp_obj.get_chart_values(hadm_id_str)