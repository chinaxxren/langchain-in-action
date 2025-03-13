# 导入所需的库
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    PyPDFLoader,    # PDF文档加载器
    Docx2txtLoader, # Word文档加载器
    TextLoader      # 文本文档加载器
)

from dotenv import load_dotenv  # 环境变量管理
load_dotenv()  # 加载环境变量，包含API密钥等敏感信息

class ChatbotWithRetrieval:
    def __init__(self, dir):
        # 第一步：文档加载
        # 支持多种格式文档的加载和处理
        base_dir = dir
        documents = []
        for file in os.listdir(base_dir): 
            file_path = os.path.join(base_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.docx') or file.endswith('.doc'):
                loader = Docx2txtLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith('.txt'):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        
        # 第二步：文本分割
        # 将长文本分割成较小的块，以便于向量化和检索
        # chunk_size=200：每个文本块的大小
        # chunk_overlap=0：文本块之间不重叠
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)
        
        # 第三步：向量数据库构建
        # 使用OpenAI的嵌入模型将文本转换为向量
        # 使用Qdrant进行向量存储和检索
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits,
            embedding=OpenAIEmbeddings(),  # 文本向量化模型
            location=":memory:",           # 内存存储模式
            collection_name="my_documents" # 集合名称
        )
        
        # 第四步：初始化大语言模型
        # 使用OpenAI的ChatGPT模型
        self.llm = ChatOpenAI()
        
        # 第五步：初始化对话记忆组件
        # 使用ConversationSummaryMemory保存对话历史
        # 可以帮助模型记住之前的对话内容
        self.memory = ConversationSummaryMemory(
            llm=self.llm,                # 使用的语言模型
            memory_key="chat_history",   # 记忆的键名
            return_messages=True         # 以消息形式返回历史
        )
        
        # 第六步：构建检索问答链
        # 将向量存储、语言模型和记忆组件组合成完整的问答系统
        retriever = self.vectorstore.as_retriever()  # 创建检索器
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm,           # 语言模型：处理自然语言理解和生成
            retriever=retriever, # 检索器：负责从向量库中找到相关文档
            memory=self.memory  # 记忆组件：管理对话历史
        )

    def chat_loop(self):
        """交互式对话循环"""
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break
            # 调用问答链处理用户输入
            response = self.qa.invoke({"question": user_input})
            print(f"Chatbot: {response['answer']}")

if __name__ == "__main__":
    # 指定文档目录并启动聊天机器人
    folder = "/Users/chinaxxren/AI/langchain-in-action/02_文档QA系统/OneFlower"
    bot = ChatbotWithRetrieval(folder)
    bot.chat_loop()
