
from dotenv import load_dotenv  # 用于加载环境变量
load_dotenv()  # 加载 .env 文件中的环境变量

# 导入所需的库和模块
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 带记忆的聊天机器人类
class ChatbotWithMemory:
    def __init__(self):

        # 初始化LLM
        self.llm = ChatOpenAI()

        # 初始化Prompt
        self.prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    "你是一个花卉行家。你通常的回答不超过30字。"
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        
        # 初始化Memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # 使用 RunnableSequence 替代 LLMChain
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt
            | self.llm
        )

    # 与机器人交互的函数
    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break
            
            response = self.chain.invoke({"question": user_input})
            self.memory.save_context({"question": user_input}, {"output": response.content})
            print(f"Chatbot: {response.content}")

if __name__ == "__main__":
    # 启动Chatbot
    bot = ChatbotWithMemory()
    bot.chat_loop()
