# 这是一个基于 CAMEL (Communicative Agents for "Mind" 
# Exploration of Large Language Models) 框架的对话系统实现

# ====================== 1. 导入依赖 ======================
# 导入环境变量管理库
from dotenv import load_dotenv
load_dotenv()

# 导入类型提示支持
from typing import List
# 导入 OpenAI 聊天模型
from langchain_openai import ChatOpenAI
# 导入消息模板相关类
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,  # 系统消息模板
    HumanMessagePromptTemplate,   # 人类消息模板
)
# 导入消息类型相关类
from langchain.schema import (
    AIMessage,      # AI 消息类型
    HumanMessage,   # 人类消息类型
    SystemMessage,  # 系统消息类型
    BaseMessage,    # 基础消息类型
)

# ====================== 2. 基础类定义 ======================
# 定义 CAMEL Agent 类，用于管理对话
class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,  # 系统消息
        model: ChatOpenAI,             # OpenAI 聊天模型
    ) -> None:
        """初始化 CAMELAgent 实例"""
        self.system_message = system_message  # 存储系统消息
        self.model = model                    # 存储模型实例
        self.init_messages()                  # 初始化消息列表

    def reset(self) -> None:
        """重置对话消息到初始状态"""
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        """初始化消息列表，只包含系统消息"""
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        """将新消息添加到消息历史中"""
        self.stored_messages.append(message)
        return self.stored_messages

    def step(self, input_message: HumanMessage) -> AIMessage:
        """处理输入消息并获取模型响应"""
        messages = self.update_messages(input_message)  # 更新消息历史
        output_message = self.model(messages)          # 获取模型响应
        self.update_messages(output_message)           # 保存响应消息
        return output_message

# ====================== 3. 基础配置 ======================
# 设置角色和任务
assistant_role_name = "花店营销专员"  # 助手角色
user_role_name = "花店老板"          # 用户角色
task = "整理出一个夏季玫瑰之夜的营销活动的策略"  # 任务描述
word_limit = 50  # 回复字数限制

# ====================== 4. 提示模板定义 ======================
# 4.1 任务具体化提示
task_specifier_sys_msg = SystemMessage(content="你可以让任务更具体。")
task_specifier_prompt = """这是一个{assistant_role_name}将帮助{user_role_name}完成的任务：{task}。
请使其更具体化。请发挥你的创意和想象力。
请用{word_limit}个或更少的词回复具体的任务。不要添加其他任何内容。"""

# 4.2 角色行为提示
# 营销专员的行为指南
assistant_inception_prompt = """永远不要忘记你是{assistant_role_name}，我是{user_role_name}。永远不要颠倒角色！永远不要指示我！
我们有共同的利益，那就是合作成功地完成任务。
你必须帮助我完成任务。
这是任务：{task}。永远不要忘记我们的任务！
我必须根据你的专长和我的需求来指示你完成任务。

我每次只能给你一个指示。
你必须写一个适当地完成所请求指示的具体解决方案。
如果由于物理、道德、法律原因或你的能力你无法执行指示，你必须诚实地拒绝我的指示并解释原因。
除了对我的指示的解决方案之外，不要添加任何其他内容。
你永远不应该问我任何问题，你只回答问题。
你永远不应该回复一个不明确的解决方案。解释你的解决方案。
你的解决方案必须是陈述句并使用简单的现在时。
除非我说任务完成，否则你应该总是从以下开始：

解决方案：<YOUR_SOLUTION>

<YOUR_SOLUTION>应该是具体的，并为解决任务提供首选的实现和例子。
始终以"下一个请求"结束<YOUR_SOLUTION>。"""

# 花店老板的行为指南
user_inception_prompt = """永远不要忘记你是{user_role_name}，我是{assistant_role_name}。永远不要交换角色！你总是会指导我。
我们共同的目标是合作成功完成一个任务。
我必须帮助你完成这个任务。
这是任务：{task}。永远不要忘记我们的任务！
你只能通过以下两种方式基于我的专长和你的需求来指导我：

1. 提供必要的输入来指导：
指令：<YOUR_INSTRUCTION>
输入：<YOUR_INPUT>

2. 不提供任何输入来指导：
指令：<YOUR_INSTRUCTION>
输入：无

"指令"描述了一个任务或问题。与其配对的"输入"为请求的"指令"提供了进一步的背景或信息。

你必须一次给我一个指令。
我必须写一个适当地完成请求指令的回复。
如果由于物理、道德、法律原因或我的能力而无法执行你的指令，我必须诚实地拒绝你的指令并解释原因。
你应该指导我，而不是问我问题。
现在你必须开始按照上述两种方式指导我。
除了你的指令和可选的相应输入之外，不要添加任何其他内容！
继续给我指令和必要的输入，直到你认为任务已经完成。
当任务完成时，你只需回复一个单词<CAMEL_TASK_DONE>。
除非我的回答已经解决了你的任务，否则永远不要说<CAMEL_TASK_DONE>。"""

# ====================== 5. 辅助函数 ======================
# 根据预设的角色和任务提示生成系统消息
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    # 创建营销专员的系统消息模板
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    
    # 格式化营销专员的系统消息
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    # 创建花店老板的系统消息模板
    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )

    # 格式化花店老板的系统消息
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    return assistant_sys_msg, user_sys_msg

# ====================== 6. 任务具体化处理 ======================
# 创建任务具体化的消息模板
task_specifier_template = HumanMessagePromptTemplate.from_template(
    template=task_specifier_prompt
)

# 创建任务具体化的代理实例
task_specify_agent = CAMELAgent(
    task_specifier_sys_msg, 
    ChatOpenAI(model_name='gpt-4', temperature=1.0)  # 使用 GPT-4 模型
)

# 格式化任务具体化消息
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    word_limit=word_limit,
)[0]

# ====================== 7. 主执行流程 ======================
if __name__ == "__main__":
    # 获取具体化后的任务描述
    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    print(f"Specified task: {specified_task_msg.content}")
    specified_task = specified_task_msg.content

    # 生成两个角色的系统消息
    assistant_sys_msg, user_sys_msg = get_sys_msgs(
        assistant_role_name, user_role_name, specified_task
    )

    # 创建两个角色的 Agent 实例，temperature=0.2 表示较低的随机性
    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(temperature=0.2))
    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(temperature=0.2))

    # 重置两个 Agent 的对话历史
    assistant_agent.reset()
    user_agent.reset()

    # 创建初始对话消息
    assistant_msg = HumanMessage(
        content=(
            f"{user_sys_msg.content}。"  # 添加系统消息内容
            "现在开始逐一给我介绍。"     # 开始指令
            "只回复指令和输入。"         # 限制回复格式
        )
    )

    # 初始化用户消息并获取第一次响应
    user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
    user_msg = assistant_agent.step(user_msg)

    # 打印原始任务和具体化后的任务
    print(f"Original task prompt:\n{task}\n")
    print(f"Specified task prompt:\n{specified_task}\n")

    # 开始对话循环，最多进行 30 轮对话
    chat_turn_limit, n = 30, 0
    while n < chat_turn_limit:
        n += 1
        # 花店老板（用户）发送指令
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content)
        print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")

        # 营销专员（助手）回复方案
        assistant_ai_msg = assistant_agent.step(user_msg)
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
        
        # 如果用户表示任务完成，则结束对话
        if "<CAMEL_TASK_DONE>" in user_msg.content:
            break