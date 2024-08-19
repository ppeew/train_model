import json

import requests
import base64

requests.packages.urllib3.disable_warnings()

def ConvertWords(text):
    headers = {}
    data = {
        "text": text,
        "locale": "zh",
    }
    json_data = json.dumps(data)
    response0 = requests.request("POST", "https://wordcount.com/api/rewrite", headers=headers, data=json_data,
                                 verify=False)
    obj = json.loads(response0.content)
    return obj["content"]

words="AI，机房101，园区3的1号冰箱的用电情况怎么样，说给我听听"
for _ in range(10):
  words=ConvertWords(words)
  print(words)
