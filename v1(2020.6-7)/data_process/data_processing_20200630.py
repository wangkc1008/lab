"""
created by PyCharm
date: 2020/6/14
time: 17:50
user: wkc
"""

from testPostgreSQL import PostgreSQLOperate
import csv


def write_to_file(filename, label, res):
    with open(filename, mode='w+', encoding='utf-8', newline='') as f:
        write = csv.writer(f, dialect='excel')
        write.writerow(label)
        for row in res:
            write.writerow(row)
    f.close()

if __name__ == '__main__':
    post = PostgreSQLOperate('127.0.0.1', 5432)
    post.connect('postgres', 'postgres', 'mimic')

    # 病人基本信息
    res1 = post.select("select subject_id,gender,dod,dob,expire_flag from patients where subject_id in (select d.subject_id from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id);")
    filename_1 = "./data_20200630_1.csv"
    print('%s 数据长度：%d' % (filename_1 , len(res1)))
    write_to_file(filename_1, ['subject_id', 'gender', 'dod', 'dob', 'expire_flag'], res1)
    print('%s 文件写入完成' % filename_1)

    # 病人婚姻状况
    res2 = post.select("select a.subject_id,a.hadm_id,marital_status,diagnosis,ethnicity,admittime from admissions a where a.hadm_id in (select distinct(d.hadm_id) from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id);")
    filename_2 = "./data_20200630_2.csv"
    print('%s 数据长度：%d' % (filename_2, len(res2)))
    write_to_file(filename_2, ['subject_id', 'hadm_id', 'marital_status', 'diagnosis', 'ethnicity', 'admittime'], res2)
    print('%s 文件写入完成' % filename_2)

    # 病人chart_events
    res3 = post.select("select subject_id,hadm_id,itemid,value from chartevents where hadm_id in (select distinct(d.hadm_id) from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id) and itemid in (227007,220046,227243,51,484,225309,482,6,220050,442,6701,224167,455,220051,227242,8368,8446,225310,8445,8364,8440,8555,224643,8441,220180,789,3748,1524,220603,850,3811,1540,225693,220624,1127,861,4200,1542,227062,227063,220546,51256,6256,227457,225678,824,806,50963,851,793,1526,14090,228240,50912,781,853,829,3792,227,226535,837,3803,220645,1536,228389,116,430,431,227008,51006,51277,644,50868,827,50970,51301,8480,212,432,51237,1087,51274,470,50863,50931,617,824,787,786,51491,813,833,50882,8441,50893,51221,762,51244,51279,814,198,51222,51249,50862,87,227411,814);")  # 你就用这个就行 下面的insert和update和delete你先不要用
    filename_3 = "./data_20200630_3.csv"
    print('%s 数据长度：%d' % (filename_3, len(res3)))
    write_to_file(filename_3, ['subject_id', 'hadm_id', 'itemid', 'value'], res3)
    print('%s 文件写入完成' % filename_3)

    # 病人lab_events
    res4 = post.select("select subject_id,hadm_id,itemid,value,valuenum,valueuom from labevents where hadm_id in (select distinct(d.hadm_id) from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id) and itemid in (227007,220046,227243,51,484,225309,482,6,220050,442,6701,224167,455,220051,227242,8368,8446,225310,8445,8364,8440,8555,224643,8441,220180,789,3748,1524,220603,850,3811,1540,225693,220624,1127,861,4200,1542,227062,227063,220546,51256,6256,227457,225678,824,806,50963,851,793,1526,14090,228240,50912,781,853,829,3792,227,226535,837,3803,220645,1536,228389,116,430,431,227008,51006,51277,644,50868,827,50970,51301,8480,212,432,51237,1087,51274,470,50863,50931,617,824,787,786,51491,813,833,50882,8441,50893,51221,762,51244,51279,814,198,51222,51249,50862,87,227411,814);")  # 你就用这个就行 下面的insert和update和delete你先不要用
    filename_4 = "./data_20200630_4.csv"
    print('%s 数据长度：%d' % (filename_4, len(res4)))
    write_to_file(filename_4, ['subject_id', 'hadm_id', 'itemid', 'value', 'valuenum', 'valueuom'], res4)
    print('%s 文件写入完成' % filename_4)

    # 病人体重
    res5 = post.select("select subject_id,hadm_id,max(patientweight) from inputevents_mv where hadm_id in (select distinct(d.hadm_id) from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id) group by subject_id,hadm_id;")
    filename_5 = "./data_20200630_5.csv"
    print('%s 数据长度：%d' % (filename_5, len(res5)))
    write_to_file(filename_5, ['subject_id', 'hadm_id', 'patientweight'], res5)
    print('%s 文件写入完成' % filename_5)

    post.close()
