'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''

from dotenv import load_dotenv
load_dotenv()

# 导入所需库
from langchain import PromptTemplate, OpenAI, LLMChain

# 设置提示模板
prompt = PromptTemplate(
    input_variables=["flower", "season"],
    template="{flower}在{season}的花语是?"
)

# 初始化大模型
llm = OpenAI(temperature=0)

# 初始化链
llm_chain = LLMChain(llm=llm, prompt=prompt)

print("*" * 60)

# 1，调用链，- 返回完整的响应对象，包含输出和其他元数据
response = llm_chain({
    'flower': "玫瑰",
    'season': "夏季"
})
print(response)
print("*" * 60)
#{'flower': '玫瑰', 'season': '夏季', 'text': '\n\n夏季的花语是热情、热爱、幸福和欢乐。'}

# 2，run方法，- 只返回生成的文本结果
result = llm_chain.run({
    'flower': "玫瑰",
    'season': "夏季"
})
print(result)
print("*" * 60)
#夏季的花语是: 热情、欢乐、爱情、美好、繁荣、生机

# 3，predict方法
result = llm_chain.predict(flower="玫瑰", season="夏季")
print(result)
print("*" * 60)
#夏季的花语是热情、浪漫、繁荣和美丽。

# 4，apply方法允许您针对输入列表运行链
input_list = [
    {"flower": "玫瑰", 'season': "夏季"},
    {"flower": "百合", 'season': "春季"},
    {"flower": "郁金香", 'season': "秋季"}
]
# 返回结果列表- 适合需要处理多个相似查询的场景
result = llm_chain.apply(input_list)
print(result)
print("*" * 60)

# generate方法，- 适合需要详细跟踪和分析的场景
result = llm_chain.generate(input_list)
print(result)