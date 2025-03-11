'''欢迎来到LangChain实战课
https://time.geekbang.org/column/intro/100617601
作者 黄佳'''
from dotenv import load_dotenv
import os
load_dotenv()

# 导入必要的库
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

# 初始化HF LLM
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    max_new_tokens=200  # 添加 token 限制，设置为小于 250 的值
)

# 创建简单的question-answering提示模板
template = """Question: {question}
              Answer: """

# 创建Prompt          
prompt = PromptTemplate(template=template, input_variables=["question"])

# 创建运行链
chain = prompt | llm | StrOutputParser()

# 准备问题
question = "Rose is which type of flower?"

# 调用模型并返回结果
print(chain.invoke({"question": question}))