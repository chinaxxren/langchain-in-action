
# 导入环境变量管理库并加载环境变量
from dotenv import load_dotenv
load_dotenv()  # 从.env文件加载环境变量到程序中

# 导入所需的库
from typing import List  # 用于类型提示
# 导入OpenAI聊天模型
from langchain_openai import ChatOpenAI
# 导入消息模板相关类
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,  # 用于创建系统消息的模板
    HumanMessagePromptTemplate,   # 用于创建人类消息的模板
)
# 导入消息类型相关类
from langchain.schema import (
    AIMessage,      # AI生成的消息类型
    HumanMessage,   # 人类输入的消息类型
    SystemMessage,  # 系统指令的消息类型
    BaseMessage,    # 所有消息类型的基类
)

# 定义CAMELAgent类，用于管理与语言模型的交互
class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,  # 初始化时的系统指令
        model: ChatOpenAI,              # 使用的语言模型
    ) -> None:
        self.system_message = system_message  # 存储系统消息
        self.model = model                    # 存储模型实例
        self.init_messages()                  # 初始化消息历史

    def reset(self) -> None:
        """重置对话消息到初始状态"""
        self.init_messages()  # 调用初始化方法
        return self.stored_messages  # 返回重置后的消息列表

    def init_messages(self) -> None:
        """初始化对话消息列表，只包含系统消息"""
        self.stored_messages = [self.system_message]  # 创建只包含系统消息的列表

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        """将新消息添加到对话历史中"""
        self.stored_messages.append(message)  # 添加新消息
        return self.stored_messages  # 返回更新后的消息列表

    def step(self, input_message: HumanMessage) -> AIMessage:
        """执行一轮对话，获取模型响应"""
        messages = self.update_messages(input_message)  # 添加输入消息到历史

        # 将 self.model(messages) 改为 self.model.invoke(messages)
        output_message = self.model.invoke(messages)  # 调用模型获取响应
        self.update_messages(output_message)   # 将响应添加到历史

        return output_message  # 返回模型响应
    
# 设置对话角色和任务
assistant_role_name = "花店营销专员"  # 助手角色名称
user_role_name = "花店老板"          # 用户角色名称
task = "整理出一个夏季玫瑰之夜的营销活动的策略"  # 任务描述
word_limit = 50  # 任务具体化的字数限制

########################################################################################3

# 定义任务具体化的系统提示
task_specifier_sys_msg = SystemMessage(content="你可以让任务更具体。")  # 创建系统消息
# 创建任务具体化的代理实例
task_specify_agent = CAMELAgent(
    task_specifier_sys_msg,  # 使用任务具体化系统消息
    ChatOpenAI(model_name = 'gpt-4', temperature=1.0)  # 使用GPT-4模型，高创造性设置
)

########################################################################################3

# 定义任务具体化的提示模板
task_specifier_prompt = """这是一个{assistant_role_name}将帮助{user_role_name}完成的任务：{task}。
请使其更具体化。请发挥你的创意和想象力。
请用{word_limit}个或更少的词回复具体的任务。不要添加其他任何内容。"""

# 创建任务具体化的消息模板
task_specifier_template = HumanMessagePromptTemplate.from_template(
    template=task_specifier_prompt  # 使用上面定义的提示模板
)

# 格式化任务具体化消息
task_specifier_msg = task_specifier_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
    word_limit=word_limit,
)[0]  # 获取格式化后的第一条消息

# 获取具体化后的任务描述
specified_task_msg = task_specify_agent.step(task_specifier_msg)  # 发送消息获取响应
print(f"Specified task: {specified_task_msg.content}")  # 打印具体化后的任务
specified_task = specified_task_msg.content  # 存储具体化后的任务文本

########################################################################################3

# 定义营销专员(助手)的行为指南
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

# 定义花店老板(用户)的行为指南
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


# 根据预设的角色和任务提示生成系统消息
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    # 创建营销专员的系统消息模板
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt  # 使用营销专员的行为指南
    )
    # 格式化营销专员的系统消息
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]  # 获取格式化后的第一条消息

    # 创建花店老板的系统消息模板
    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt  # 使用花店老板的行为指南
    )
    # 格式化花店老板的系统消息
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]  # 获取格式化后的第一条消息

    return assistant_sys_msg, user_sys_msg  # 返回两个角色的系统消息

# 生成两个角色的系统消息(注意这里是系统消息，不是人类消息)
assistant_sys_msg, user_sys_msg = get_sys_msgs(
    assistant_role_name, user_role_name, specified_task  # 使用具体化后的任务
)

# 创建营销专员的代理实例
assistant_agent = CAMELAgent(
    assistant_sys_msg,  # 使用营销专员的系统消息
    ChatOpenAI(temperature=0.2)  # 使用较低的随机性，确保回答的一致性
)
# 创建花店老板的代理实例
user_agent = CAMELAgent(
    user_sys_msg,  # 使用花店老板的系统消息
    ChatOpenAI(temperature=0.2)  # 使用较低的随机性，确保回答的一致性
)

# 重置两个代理的对话历史
assistant_agent.reset()
user_agent.reset()

# 初始化对话互动，创建发送给营销专员的初始消息
assistant_msg = HumanMessage(
    content=(
        f"{user_sys_msg.content}。"  # 添加花店老板的系统消息内容
        "现在开始逐一给我介绍。"     # 开始指令
        "只回复指令和输入。"         # 限制回复格式
    )
)

# 初始化用户消息并获取第一次响应
user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")  # 创建包含营销专员系统消息的人类消息
user_msg = assistant_agent.step(user_msg)  # 获取营销专员的首次响应

# 打印原始任务和具体化后的任务
print(f"Original task prompt:\n{task}\n")
print(f"Specified task prompt:\n{specified_task}\n")

# 模拟对话交互，直到达到对话轮次上限或任务完成
chat_turn_limit, n = 30, 0  # 设置最大对话轮次为30
while n < chat_turn_limit:
    n += 1  # 增加对话轮次计数
    # 花店老板(用户)发送指令
    user_ai_msg = user_agent.step(assistant_msg)  # 花店老板处理营销专员的消息
    user_msg = HumanMessage(content=user_ai_msg.content)  # 创建人类消息
    print(f"AI User {n} ({user_role_name}):\n\n{user_msg.content}\n\n")  # 打印花店老板的消息

    # 营销专员(助手)回复方案
    assistant_ai_msg = assistant_agent.step(user_msg)  # 营销专员处理花店老板的消息
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)  # 创建人类消息
    print(f"AI Assistant {n} ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")  # 打印营销专员的消息
    
    # 检查是否任务完成
    if "<CAMEL_TASK_DONE>" in user_msg.content:  # 如果花店老板表示任务完成
        break  # 结束对话循环