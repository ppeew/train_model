import random
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

# 生成训练数据
train_data = []
input_word = []
target = []
for garden in gardens:
    for floor in floors:
        for room in rooms:
            for device_name in device_names:
                for device_id in device_ids:
                    for device_info in device_infos:
                        for i in range(20):
                            # TODO 同一个结果有多种生成方式
                            input2 = "查询(1)(2)(3)(4)(5)(6)"
                            positions = [1, 2, 3, 4, 5, 6]
                            # 需要替换的内容
                            selected_positions = random.sample(range(1, 7, 1), 5)
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
                            input_word.append(input2)
                            target.append(t2)
                            train_data.append((input2, t2))

df = pd.DataFrame(train_data)
df.to_csv("train_data_bak2.csv", encoding='utf-8', index=False, header=False)

test_data = []
random_ids = [random.randint(0, len(input_word) - 1) for _ in range(len(input_word) // 10)]
for id in random_ids:
    test_data.append((input_word[id], target[id]))
df2 = pd.DataFrame(test_data)
df2.to_csv("test_data_bak2.csv", encoding='utf-8', index=False, header=False)
