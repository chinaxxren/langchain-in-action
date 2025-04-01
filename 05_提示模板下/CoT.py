# 使用 Chain of Thought (CoT) 思维链方法的花店 AI 助手示例，主要包含以下几个部分：

# 1. 系统角色设定：
# - 定义为花店电商公司的 AI 助手
# - 目标是帮助客户选择合适的花卉
# 2. 思维链模板设计：
# - 包含详细的思考步骤：理解需求 → 思考问题 → 给出推荐

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# 设置环境变量和API密钥
from dotenv import load_dotenv  # 用于加载环境变量

# 创建聊天模型
from langchain_openai import ChatOpenAI  # 更新导入路径

load_dotenv()  # 加载 .env 文件中的环境变量

llm = ChatOpenAI(temperature=0)

# 设定 AI 的角色和目标
role_template = (
    "你是一个为花店电商公司工作的AI助手, 你的目标是帮助客户根据他们的喜好做出明智的决定"
)

# CoT 的关键部分，AI 解释推理过程，并加入一些先前的对话示例（Few-Shot Learning）
cot_template = """
作为一个为花店电商公司工作的AI助手，我的目标是帮助客户根据他们的喜好做出明智的决定。 

我会按部就班的思考，先理解客户的需求，然后考虑各种鲜花的涵义，最后根据这个需求，给出我的推荐。
同时，我也会向客户解释我这样推荐的原因。

示例 1:
  人类：我想找一种象征爱情的花。
  AI：首先，我理解你正在寻找一种可以象征爱情的花。
  在许多文化中，红玫瑰被视为爱情的象征，这是因为
  它们的红色通常与热情和浓烈的感情联系在一起。因此，
  考虑到这一点，我会推荐红玫瑰。红玫瑰不仅能够象征爱情，
  同时也可以传达出强烈的感情，这是你在寻找的。

示例 2:
  人类：我想要一些独特和奇特的花。
  AI：从你的需求中，我理解你想要的是独一无二
  和引人注目的花朵。兰花是一种非常独特并且颜色鲜艳的花，
  它们在世界上的许多地方都被视为奢侈品和美的象征。因此，
  我建议你考虑兰花。选择兰花可以满足你对独特和奇特的要求，
  而且，兰花的美丽和它们所代表的力量和奢侈也可能会吸引你。
"""

# 设定 AI 的角色
system_prompt_role = SystemMessagePromptTemplate.from_template(role_template)
# 设定 AI 的目标
system_prompt_cot = SystemMessagePromptTemplate.from_template(cot_template)

# 用户的询问
human_template = "{human_input}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 将以上所有信息结合为一个聊天提示
chat_prompt = ChatPromptTemplate.from_messages(
    [
        system_prompt_role,  # 角色定义
        system_prompt_cot,  # 思维链模板
        human_prompt,  # 用户输入
    ]
)

# 生成提示
prompt = chat_prompt.format_prompt(
    human_input="我想为我的女朋友购买一些花。她喜欢粉色和紫色。你有什么建议吗?"
).to_messages()

# 接收用户的询问，返回回答结果
response = llm.invoke(prompt)  # 使用 invoke 方法替代直接调用
print(response.content)  # 使用 content 属性获取响应内容
