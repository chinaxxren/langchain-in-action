# 导入所需的库
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# ChatBot类的实现
class ChatbotWithRetrieval:

    def __init__(self, dir):

        # 加载Documents
        base_dir = dir # 文档的存放目录
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
        
        # 文本的分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        all_splits = text_splitter.split_documents(documents)
        
        # 向量数据库
        self.vectorstore = Qdrant.from_documents(
            documents=all_splits, # 以分块的文档
            embedding=OpenAIEmbeddings(), # 用OpenAI的Embedding Model做嵌入
            location=":memory:",  # in-memory 存储
            collection_name="my_documents",) # 指定collection_name
        
        # 初始化LLM
        self.llm = ChatOpenAI()
        
        # 初始化Memory
        self.memory = ConversationSummaryMemory(
            llm=self.llm, 
            memory_key="chat_history", 
            return_messages=True
            )
        
        # 设置Retrieval Chain
        retriever = self.vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm, 
            retriever=retriever, 
            memory=self.memory
            )

    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break
            # 调用 Retrieval Chain  
            response = self.qa(user_input)
            print(f"Chatbot: {response['answer']}")

# Streamlit界面的创建
def main():
    st.title("易速鲜花聊天客服")
    folder = "/Users/chinaxxren/AI/langchain-in-action/02_文档QA系统/OneFlower"
    
    # 初始化聊天历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 初始化机器人
    if "bot" not in st.session_state:
        st.session_state.bot = ChatbotWithRetrieval(folder)

    # 显示聊天历史
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 聊天输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 获取机器人响应
        with st.chat_message("assistant"):
            response = st.session_state.bot.qa.invoke({"question": prompt})
            st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if __name__ == "__main__":
    main()

