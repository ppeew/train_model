# import os
# os.environ["SYNONYMS_DL_LICENSE"] = "YOUR_LICENSE"
# _licenseid = os.environ.get("SYNONYMS_DL_LICENSE", None)
# print("SYNONYMS_DL_LICENSE=", _licenseid)

import synonyms

# synonyms.describe()
#
# synonyms.display("能力")
#
# print(synonyms.nearby("人脸"))
#
# print(synonyms.display("人脸"))

print(synonyms.seg("查询园区1,101机房,设备名为热通道,设备编号为9000,温湿度信息情况"),sep='\n')

# TODO 在原来基础上（已经能够生成完整词句，但是不符合自然语言输入规律）-----> 提取其中的动词，替换，并将新文本作为输入的训练集
# TODO 在原来的输出上（M1-1F-101机房-热通道-9000#cpu占用率）-------> M1-1F-101机房-热通道-9000#cpu占用率（宾语）~~~查询设备使用情况（动作）~~~~
