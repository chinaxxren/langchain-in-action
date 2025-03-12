
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

import os  # 添加 os 模块导入

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

# 创建文本补全请求
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",  # 使用 GPT-3.5 turbo instruct 模型
    temperature=0.5,                  # 控制输出的随机性
    max_tokens=100,                   # 限制输出长度
    prompt="请给我的花店起个名"       # 输入提示
)

# strip() 方法用于去除文本前后的空白字符（包括空格、换行符、制表符等）
print(response.choices[0].text.strip())