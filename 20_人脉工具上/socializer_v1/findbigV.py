# 导入环境变量管理库
from dotenv import load_dotenv
load_dotenv()

# 导入所取的库
import re
from agents.weibo_agent import lookup_V

if __name__ == "__main__":

    # 拿到UID
    response_UID = lookup_V(flower_type = "牡丹" )

    # 抽取UID里面的数字
    UID = re.findall(r'\d+', response_UID)[0]
    print("这位鲜花大V的微博ID是", UID)



