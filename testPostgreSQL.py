"""
created by PyCharm
date: 2020/6/9
time: 19:50
user: wkc
"""

import csv
import psycopg2


class PostgreSQLOperate:

    def __init__(self, host='127.0.0.1', port=5432):
        """
        初始化数据库连接类
        """
        self.__host = host
        self.__port = port
        self.post_conn = None
        self.res = None

    def connect(self, user, password, db_name):
        """
        连接数据库
        """
        self.post_conn = psycopg2.connect(dbname=db_name, user=user, password=password,
                                          port=self.__port, host=self.__host, client_encoding="UTF-8")

    def select(self, sql):
        """
        执行select 等sql语句
        :param sql:
        :return: 返回查询结果
        """
        cursor = self.post_conn.cursor()
        cursor.execute(sql)
        self.res = cursor.fetchall()
        return self.res

    def execute(self, sql, value):
        """
        执行insert、update、delete sql语句
        :param sql:
        :param value: 写入数据
        :return: 返回insert_id
        """
        cursor = self.post_conn.cursor()
        try:
            cursor.execute(sql, value)
            self.post_conn.commit()
        except Exception as e:
            self.post_conn.rollback()
            print('error:%s' % e)

        return True

    def close(self):
        self.post_conn.close()


if __name__ == '__main__':
    post = PostgreSQLOperate('127.0.0.1', 5432)
    post.connect('postgres', 'postgres', 'mimic')
    # res = post.select("select c.subject_id,c.itemid,d.label from chartevents as c inner join d_items as d on c.subject_id in (select subject_id from diagnoses_icd where icd9_code like '410%' and icd9_code not like '410%2') and c.itemid = d.itemid limit 1000000")  # 你就用这个就行 下面的insert和update和delete你先不要用
    res = post.select("select subject_id,drug_type,drug,drug_name_poe,drug_name_generic from prescriptions where subject_id in (select subject_id from diagnoses_icd where icd9_code like '410%' and icd9_code not like '410%2');")  # 你就用这个就行 下面的insert和update和delete你先不要用
    # for row in res:
    #     print(type(row))  # 每一行的数据格式是元组，看你怎么用了
    #     break
    # res = post.execute("insert into t_test(id, username) values(%s, %s)", ['456', 'huhaha'])
    # res = post.execute("update t_test set username = %s where id = %s", ['wanghaha', 123])
    # res = post.execute("delete from t_test", [])
    print(len(res))
    post.close()
    data = []
    filename = "./data_03.csv"
    with open(filename, mode='w+', encoding='utf-8', newline='') as f:
        write = csv.writer(f, dialect='excel')
        write.writerow(['subject_id', 'drug_type', 'drug', 'drug_name_poe', 'drug_name_generic'])
        # write.writerow(['subject_id', 'item_id', 'label'])
        for row in res:
            write.writerow(row)
    # print(res)