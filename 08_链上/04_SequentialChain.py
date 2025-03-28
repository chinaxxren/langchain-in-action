from dotenv import load_dotenv
from langchain_openai import OpenAI  # OpenAI 模型接口
from langchain_core.prompts import PromptTemplate  # 提示模板
from langchain_core.runnables import RunnablePassthrough  # 数据流转换器
from operator import itemgetter  # 用于从字典中获取值

# RunnablePassthrough 详细解释
# `RunnablePassthrough` 是 LangChain 中一个重要的数据流工具，主要用于在链式操作中传递和转换数据。
# 它是一个函数，接受一个输入，并返回一个输出。

load_dotenv()  # 加载环境变量

# 初始化 OpenAI 模型
# temperature: 控制输出的随机性，值越大随机性越高
llm = OpenAI(temperature=.7)

# 第一个链：生成鲜花的介绍
# 使用 PromptTemplate 创建模板，定义输入变量和提示文本
introduction_prompt = PromptTemplate(
    template="""
你是一个植物学家。给定花的名称和类型，你需要为这种花写一个200字左右的介绍。
花名: {name}
颜色: {color}
植物学家: 这是关于上述花的介绍:""",
    input_variables=["name", "color"]  # 定义需要填充的变量
)
# 使用管道操作符连接提示模板和语言模型
introduction_chain = introduction_prompt | llm

# 第二个链：根据介绍写评论
review_prompt = PromptTemplate(
    template="""
你是一位鲜花评论家。给定一种花的介绍，你需要为这种花写一篇200字左右的评论。
鲜花介绍:
{introduction}
花评人对上述花的评论:""",
    input_variables=["introduction"]  # 只需要花的介绍作为输入
)
review_chain = review_prompt | llm

# 第三个链：生成社交媒体文案
social_prompt = PromptTemplate(
    template="""
你是一家花店的社交媒体经理。给定一种花的介绍和评论，你需要为这种花写一篇社交媒体的帖子，300字左右。
鲜花介绍:
{introduction}
花评人对上述花的评论:
{review}
社交媒体帖子:""",
    input_variables=["introduction", "review"]  # 需要花的介绍和评论作为输入
)
social_chain = social_prompt | llm

# 组合链：使用 RunnablePassthrough 和字典组合多个链
chain = (
    # RunnablePassthrough 用于传递原始输入到后续步骤
    RunnablePassthrough()
    # 第一步：获取名称和颜色，生成介绍
    | {
        "name": itemgetter("name"),  # 从输入中获取花名
        "color": itemgetter("color"),  # 从输入中获取颜色
        "introduction": introduction_chain,  # 生成介绍
    }
    # 第二步：保留原始信息，添加评论
    | {
        "introduction": itemgetter("introduction"),  # 传递介绍
        "review": review_chain,  # 生成评论
        "name": itemgetter("name"),  # 继续传递花名
        "color": itemgetter("color"),  # 继续传递颜色
    }
    # 第三步：生成社交媒体文案
    | {
        "introduction": itemgetter("introduction"),  # 保留介绍
        "review": itemgetter("review"),  # 保留评论
        "social_post": social_chain,  # 生成社交媒体文案
    }
)

# 运行组合链并打印结果
result = chain.invoke({
    "name": "玫瑰",
    "color": "黑色"
})

# 按照不同部分打印结果
print("\n=== 鲜花介绍 ===")
print(result["introduction"])
print("\n=== 评论 ===")
print(result["review"])
print("\n=== 社交媒体文案 ===")
print(result["social_post"])
