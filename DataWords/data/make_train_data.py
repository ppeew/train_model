import random

import pymysql
import jieba
import json
import csv


def WordsRandom():
    # 数据库连接
    conn = pymysql.connect(host='10.3.0.247', user='dhtest', password='xpFH9PkrKFrIRPtr3UHnW', db='rangeidc')
    cursor = conn.cursor()

    # 查询地址数据
    cursor.execute("SELECT addr_id, name, parent_id, report, life FROM address WHERE life=1")
    address_data = cursor.fetchall()

    # 打开 CSV 文件进行写入
    with open('train_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')

        # 生成并写入训练数据
        for addr in address_data:
            addr_id, name, parent_id, report, life = addr

            # 根据地址生成描述性句子
            input_text = f"查询{name}的报告状态和存活状态，以及父节点信息。"

            # 分词
            # segments = jieba.lcut(input_text)

            # 将分词结果转换为字符串，方便写入 CSV
            # segments_str = " ".join(segments)

            # 打包为 JSON 格式
            address_info = json.dumps({
                'addr_id': addr_id,
                'name': name,
                'parent_id': parent_id,
                'report': report,
                'life': life
            }, ensure_ascii=False)

            # 写入数据到 CSV
            writer.writerow([input_text, address_info])

    # 关闭连接
    cursor.close()
    conn.close()

    print("训练数据生成完毕，已保存至 'train_data.csv'")


WordsRandom()
