from dotenv import load_dotenv  # 用于加载环境变量
import pandas as pd
from langchain_openai import ChatOpenAI  # 改用 ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate  # 更新导入路径

load_dotenv()  # 加载 .env 文件中的环境变量

model = ChatOpenAI(model_name='gpt-4')

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 定义我们想要接收的数据格式
class FlowerDescription(BaseModel):
    flower_type: str = Field(description="鲜花的种类")
    price: int = Field(description="鲜花的价格")
    description: str = Field(description="鲜花的描述文案")
    reason: str = Field(description="为什么要这样写这个文案")

# ------Part 3
# 创建输出解析器
output_parser = PydanticOutputParser(pydantic_object=FlowerDescription)

# 获取输出格式指示
format_instructions = output_parser.get_format_instructions()
print("输出格式：",format_instructions)
print("*" * 60)

# 创建提示模板
prompt_template = """
您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower} ，您能提供一个吸引人的简短中文描述吗？
{format_instructions}
请确保输出是一个有效的 JSON 对象，包含所有必需的字段。
"""

# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template, partial_variables={"format_instructions": format_instructions}) 

# 打印提示
print("提示：", prompt)
print("*" * 60)

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower=flower, price=price)
    
    # 获取模型的输出
    output = model.invoke(input).content  # 添加 .content 来获取实际内容
    print(f"原始输出: {output}")

