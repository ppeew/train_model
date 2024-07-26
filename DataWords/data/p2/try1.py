import random
import pandas as pd

original_code = """
查询园区?1?2楼的?3机房,设备名为?4,设备编号为?5的?6信息情况
"""

positions=[1,2,3,4,5,6]

# 需要替换的内容
replacement_content = "###"

# 随机选择3个位置进行替换
selected_positions = random.sample(range(1,7,1), 3)
reversed_positions = [i for i in positions if i not in selected_positions]
# 将原始代码中的3个位置替换为需要的内容
for pos in selected_positions:
    original_code = original_code.replace(f"?{pos}", replacement_content)
for pos in reversed_positions:
    original_code = original_code.replace(f"?{pos}", "")

print(selected_positions)
print(reversed_positions)
# 打印修改后的代码
print(original_code)

