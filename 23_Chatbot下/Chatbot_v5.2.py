# 导入所需的库
import os
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from dotenv import load_dotenv
load_dotenv()

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
        # 初始化对话历史
        self.conversation_history = ""

        # 设置Retrieval Chain
        retriever = self.vectorstore.as_retriever()
        self.qa = ConversationalRetrievalChain.from_llm(
            self.llm, 
            retriever=retriever, 
            memory=self.memory
            )

    def get_response(self, user_input):  # 这是为 Gradio 创建的新函数
        response = self.qa(user_input)
        # 更新对话历史
        self.conversation_history += f"你: {user_input}\nChatbot: {response['answer']}\n"
        return self.conversation_history

if __name__ == "__main__":
    folder = "/Users/chinaxxren/AI/langchain-in-action/02_文档QA系统/OneFlower"
    bot = ChatbotWithRetrieval(folder)

    # 使用更简单的 Gradio 配置
    interface = gr.Interface(
        fn=bot.get_response,
        inputs=gr.Textbox(lines=2, placeholder="请输入您的问题..."),
        outputs=gr.Textbox(lines=10, label="对话历史"),
        title="易速鲜花智能客服",
        description="请输入问题，然后点击提交。",
        allow_flagging="never",
        cache_examples=False
    )
    
    # 使用基本配置启动
    interface.launch(
        server_port=7860,
        share=False,
        debug=True
    )