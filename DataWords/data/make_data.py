import random
import time

import pandas as pd

gardens = [i for i in range(1, 2, 1)]
floors = [i for i in range(1, 2, 1)]
rooms = []
for floor in floors:
    start = 101 + ((floor - 1) * 10)
    end = start + 3
    rooms.extend(range(start, end))
device_names = ["热通道", "空调配电箱", "列头柜", "二次泵", "空调", "电池"]
device_ids = [i for i in range(9000, 10000, 1000)]
device_infos = ["温湿度", "温度", "cpu占用率", "内存使用率"]

# 创建ID与label的映射
label_map = {}
label_map2 = {}
index = 0

# 生成训练数据
train_data = []
test_data = []
input_word = []
target = []
for garden in gardens:
    for floor in floors:
        for room in rooms:
            for device_name in device_names:
                for device_id in device_ids:
                    for device_info in device_infos:
                        for j in range(1, 7, 1):
                            for i in range(2, j * 2 + 1, 1):
                                # 生成全排列
                                # TODO 同一个结果有多种生成方式 查询园区(1)(2)楼的(3)机房,设备名为(4),设备编号为(5)的(6)信息情况
                                input2 = "查询(1)(2)(3)(4)(5)(6)"
                                positions = [1, 2, 3, 4, 5, 6]
                                # 需要替换的内容
                                selected_positions = random.sample(range(1, 7, 1), j)
                                reversed_positions = [i for i in positions if i not in selected_positions]
                                for pos in positions:
                                    if pos in selected_positions:
                                        data = ""
                                        if pos == 1:
                                            data = "园区" + str(garden) + ","
                                        elif pos == 2:
                                            data = str(floor) + "楼" + ","
                                        elif pos == 3:
                                            data = str(room) + "机房" + ","
                                        elif pos == 4:
                                            data = "设备名为" + str(device_name) + ","
                                        elif pos == 5:
                                            data = "设备编号为" + str(device_id) + ","
                                        elif pos == 6:
                                            data = str(device_info) + "信息情况"
                                        input2 = input2.replace(f"({pos})", data)
                                    elif pos in reversed_positions:
                                        input2 = input2.replace(f"({pos})", "")
                                t2 = "M{}-{}F-{}机房-{}-{}#{}".format(garden, floor, room, device_name, device_id,
                                                                      device_info)
                                # csv target只存储ID
                                poss = index
                                if t2 in label_map2:
                                    # 如果存在改元素
                                    poss = label_map2[t2]

                                else:
                                    label_map2[t2] = poss
                                    index = index + 1

                                label_map[poss] = t2
                                input_word.append(input2)
                                target.append(poss)

                                # 抽取一个作为测试集
                                if i == 2 * j and j == 6:
                                    test_data.append((input2, poss))
                                else:
                                    train_data.append((input2, poss))

df = pd.DataFrame(train_data)
df.to_csv("train_data.txt", sep="\t", encoding='utf-8', index=False, header=False)

df2 = pd.DataFrame(test_data)
df2.to_csv("test_data.txt", sep="\t", encoding='utf-8', index=False, header=False)

# 随机抽取训练集10%作为测试集（但是不合理，应该重新造数据）
# test_data = []
# random_ids = [random.randint(0, len(input_word) - 1) for _ in range(len(input_word) // 10)]
# for id in random_ids:
#     test_data.append((input_word[id], target[id]))
# df2 = pd.DataFrame(test_data)
# df2.to_csv("test_data_bak.csv", encoding='utf-8', index=False, header=False)

# 造测试集（合理造法,但是又得存改产生的新类型target到class）
# random.seed(time.time())
# garden2= random.randint(1,1000)
# floor2=random.randint(1,100)
# room2=random.randint(1,100)
# device_name2 = random.choice(["交换机", "路由器", "防火墙", "负载均衡器", "服务器", "存储设备"])
# device_id2 = random.randint(100000, 999999)
# device_info2=random.choice(["内核版本", "时区", "内网IP","首选DNS服务器","MAC地址","电源电压"])
#
# test_data = []
# for i in range(len(input_word)//10):
#     input2 = "查询(1)(2)(3)(4)(5)(6)"
#     positions = [1, 2, 3, 4, 5, 6]
#     # 需要替换的内容
#     selected_positions = random.sample(range(1, 7, 1), 3)
#     reversed_positions = [i for i in positions if i not in selected_positions]
#     for pos in positions:
#         if pos in selected_positions:
#             data = ""
#             if pos == 1:
#                 data = "园区"+str(garden2)+","
#             elif pos == 2:
#                 data = str(floor2)+"楼"+","
#             elif pos == 3:
#                 data = str(room2)+"机房"+","
#             elif pos == 4:
#                 data = "设备名为"+str(device_name2)+","
#             elif pos == 5:
#                 data = "设备编号为"+str(device_id2)+","
#             elif pos == 6:
#                 data = str(device_info2)+"信息情况"
#             input2 = input2.replace(f"({pos})", data)
#         elif pos in reversed_positions:
#             input2 = input2.replace(f"({pos})", "")
#     t2 = "M{}-{}F-{}机房-{}-{}#{}".format(garden2, floor2, room2, device_name2, device_id2, device_info2)
#     test_data.append((input2, label_map2[t2]))
# df2 = pd.DataFrame(test_data)
# df2.to_csv("test_data_bak.csv", encoding='utf-8', index=False, header=False)

# 存储类型.txt
class_data = [value for key, value in sorted(label_map.items())]
df3 = pd.DataFrame(class_data)
df3.to_csv("class.txt", encoding='utf-8', sep="\t", index=False, header=False)
