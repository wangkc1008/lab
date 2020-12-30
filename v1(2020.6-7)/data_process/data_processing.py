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
    res1 = post.select("select subject_id,gender,dod,dob,expire_flag from patients where subject_id in (select d.subject_id from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id);")  # 你就用这个就行 下面的insert和update和delete你先不要用
    filename_1 = "./data_1.csv"
    print('%s 数据长度：%d' % (filename_1 , len(res1)))
    write_to_file(filename_1, ['subject_id', 'gender', 'dod', 'dob', 'expire_flag'], res1)
    print('%s 文件写入完成' % filename_1)

    # 病人婚姻状况
    res2 = post.select("select a.subject_id,a.hadm_id,marital_status,diagnosis from admissions a where a.hadm_id in (select distinct(d.hadm_id) from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id);")  # 你就用这个就行 下面的insert和update和delete你先不要用
    filename_2 = "./data_2.csv"
    print('%s 数据长度：%d' % (filename_2, len(res2)))
    write_to_file(filename_2, ['subject_id', 'hadm_id', 'marital_status', 'diagnosis'], res2)
    print('%s 文件写入完成' % filename_2)

    # 病人chart_events
    res3 = post.select("select subject_id,hadm_id,c.itemid,value,d.label from chartevents c inner join d_items d on c.hadm_id in (select distinct(d.hadm_id) from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id) and c.itemid = d.itemid;")  # 你就用这个就行 下面的insert和update和delete你先不要用
    filename_3 = "./data_3.csv"
    print('%s 数据长度：%d' % (filename_3, len(res3)))
    write_to_file(filename_3, ['subject_id', 'hadm_id', 'itemid', 'value', 'label'], res3)
    print('%s 文件写入完成' % filename_3)

    # 病人lab_events
    res4 = post.select("select l.subject_id,l.hadm_id,l.itemid,l.value,l.valuenum,l.valueuom,d.label,d.fluid from labevents l inner join d_labitems d on l.hadm_id in (select distinct(d.hadm_id) from diagnoses_icd d inner join (select a.subject_id,a.hadm_id from admissions a inner join (select subject_id,min(admittime) as admittime from admissions group by subject_id) as t on a.subject_id = t.subject_id and a.admittime = t.admittime) as t2 on d.icd9_code like '410%' and d.icd9_code not like '410%2' and d.subject_id = t2.subject_id and d.hadm_id = t2.hadm_id) and l.itemid = d.itemid;")  # 你就用这个就行 下面的insert和update和delete你先不要用
    filename_4 = "./data_4.csv"
    print('%s 数据长度：%d' % (filename_4, len(res4)))
    write_to_file(filename_4, ['subject_id', 'hadm_id', 'itemid', 'value', 'valuenum', 'valueuom', 'label', 'fluid'], res4)
    print('%s 文件写入完成' % filename_4)

    post.close()
