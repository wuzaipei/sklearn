import pymongo
import pandas as pd

# 创建连接
client = pymongo.MongoClient('mongodb://localhost:27017/')
# 连接数据库
db = client['lianjia']

# 选择集合
room_price = db['fangyuanbigsz']

def read_mongodb_data():
    room_info = []
    for item in room_price.find():
        room_info.append(item)
    # 全部把这些数据转为pandas类型数据。
    room_data = pd.DataFrame(room_info,columns=list(room_info[0].keys()))
    return room_data