# ... 前面的代码保持不变 ...

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
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 创建对话链
        self.chain = (
            RunnablePassthrough.assign(
                chat_history=lambda x: self.memory.load_memory_variables({})["chat_history"]
            )
            | self.prompt
            | self.llm
        )

    def chat_loop(self):
        print("Chatbot 已启动! 输入'exit'来退出程序。")
        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                print("再见!")
                break
            
            # 获取对话历史并传递给模型
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            response = self.chain.invoke({
                "question": user_input,
                "chat_history": chat_history
            })
            
            # 保存新的对话内容
            self.memory.save_context(
                {"input": user_input},
                {"output": response.content}
            )
            print(f"Chatbot: {response.content}")

if __name__ == "__main__":
    # 启动Chatbot
    bot = ChatbotWithMemory()
    bot.chat_loop()