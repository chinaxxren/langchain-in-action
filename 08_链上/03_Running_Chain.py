from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from typing import List, Dict  # 导入类型注解所需的类型

# 加载环境变量
load_dotenv()

# 设置提示模板
# input_variables: 定义模板中需要替换的变量
# template: 定义提示模板的具体内容
prompt = PromptTemplate(
    input_variables=["flower", "season"],
    template="{flower}在{season}的花语是?"
)

# 初始化大模型
# temperature=0: 使输出更加确定性，降低随机性
llm = OpenAI(temperature=0)

# 创建链
# 使用 | 运算符将提示模板和模型连接成链
chain = prompt | llm

# 1. 使用 invoke 方法处理单个输入
# invoke 方法接受一个字典作为输入，字典的键需要与模板中的变量名匹配
response = chain.invoke({
    'flower': "玫瑰",
    'season': "夏季"
})
print(response)
print("*" * 60)  # 分隔线

# 展示 invoke 方法的另一个示例
result = chain.invoke({
    'flower': "玫瑰",
    'season': "夏季"
})
print(result)
print("*" * 60)

# 2. 准备批量处理的输入数据
# 使用类型注解指定列表中包含字典
input_list: List[Dict[str, str]] = [
    {"flower": "玫瑰", 'season': "夏季"},
    {"flower": "百合", 'season': "春季"},
    {"flower": "郁金香", 'season': "秋季"}
]

# 3. 使用 batch 方法批量处理多个输入
# batch 方法接受一个输入列表，返回对应的输出列表
results = chain.batch(input_list)
print(results)