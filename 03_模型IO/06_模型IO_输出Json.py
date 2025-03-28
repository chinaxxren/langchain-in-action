from dotenv import load_dotenv  # 用于加载环境变量
# 导入LangChain中的提示模板
from langchain.prompts import PromptTemplate
# 导入结构化输出解析器和ResponseSchema
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import json

load_dotenv()  # 加载 .env 文件中的环境变量

# 创建提示模板
prompt_template = """您是一位专业的鲜花店文案撰写员。
对于售价为 {price} 元的 {flower_name} ，您能提供一个吸引人的简短描述吗？
{format_instructions}"""

# 通过LangChain调用模型
from langchain_openai import OpenAI
# 创建模型实例
model = OpenAI(model_name='gpt-3.5-turbo-instruct')

# 定义我们想要接收的响应模式
response_schemas = [
    ResponseSchema(name="description", description="鲜花的描述文案"),
    ResponseSchema(name="reason", description="问什么要这样写这个文案")
]
# 创建输出解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式指示
format_instructions = output_parser.get_format_instructions()
# 根据模板创建提示，同时在提示中加入输出解析器的说明
prompt = PromptTemplate.from_template(prompt_template, 
                partial_variables={"format_instructions": format_instructions}) 

# 数据准备
flowers = ["玫瑰", "百合", "康乃馨"]
prices = ["50", "30", "20"]

# 创建一个列表用于存储所有结果
results = []

for flower, price in zip(flowers, prices):
    # 根据提示准备模型的输入
    input = prompt.format(flower_name=flower, price=price)

    # 获取模型的输出
    output = model.invoke(input)
    
    # 解析模型的输出
    parsed_output = output_parser.parse(output)

    # 添加flower和price信息
    parsed_output['flower'] = flower
    parsed_output['price'] = price

    # 将结果添加到列表中
    results.append(parsed_output)

# json.dumps(obj, **kwargs)
# ### 主要参数说明
# 1. **obj**: 要序列化的 Python 对象（如字典、列表等）
# 2. **ensure_ascii**: 默认为 True，设为 False 时允许输出非 ASCII 字符（如中文）
# 3. **indent**: 缩进格式化，设置缩进空格数
# 4. **sort_keys**: 是否按键排序，默认为 False
# 5. **separators**: 自定义分隔符，默认为 (', ', ': ')

# 只保留控制台输出部分
print(json.dumps(results, ensure_ascii=False, indent=2))