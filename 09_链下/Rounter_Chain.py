# graph TD
#     A[用户问题] --> B[路由器]
#     B --> C[花卉护理专家]
#     B --> D[插花大师]
#     B --> E[默认处理链]
# 花卉护理问题
# "如何为玫瑰浇水？"  -> flower_care 专家
# 花艺设计问题
# "如何为婚礼场地装饰花朵？" -> flower_decoration 专家
# 其他问题
# "如何区分阿豆和罗豆？" -> default_chain 默认链

from dotenv import load_dotenv

# 加载环境变量，通常用于加载API密钥等敏感信息
load_dotenv()

# 定义两个场景的模板
# 花卉护理模板
flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}
"""

# 花卉装饰模板
flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}
"""

# 构建提示信息列表，每个元素包含一个键、描述和模板
prompt_infos = [
    {
        "key": "flower_care",
        "description": "适合回答关于鲜花护理的问题",
        "template": flower_care_template,
    },
    {
        "key": "flower_decoration",
        "description": "适合回答关于鲜花装饰的问题",
        "template": flower_deco_template,
    },
]

# 初始化OpenAI语言模型
from langchain_openai import OpenAI

llm = OpenAI()

# 构建目标链映射
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

chain_map = {}

# 遍历提示信息列表，为每个提示创建一个LLMChain
for info in prompt_infos:
    # 创建PromptTemplate对象
    prompt = PromptTemplate(template=info["template"], input_variables=["input"])
    print("目标提示:\n", prompt)

    # 创建LLMChain对象，并将其添加到chain_map中
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    chain_map[info["key"]] = chain

# 构建路由链
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import (
    MULTI_PROMPT_ROUTER_TEMPLATE as RounterTemplate,
)

# 创建目的地列表，用于路由提示
destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
router_template = RounterTemplate.format(destinations="\n".join(destinations))
print("路由模板:\n", router_template)

# 创建路由提示模板
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
print("路由提示:\n", router_prompt)

# 创建LLMRouterChain对象
router_chain = LLMRouterChain.from_llm(llm, router_prompt, verbose=True)

# 构建默认链，用于处理无法路由的问题
from langchain.chains import ConversationChain

default_chain = ConversationChain(llm=llm, output_key="text", verbose=True)

# 构建多提示链
from langchain.chains.router import MultiPromptChain

# 创建MultiPromptChain对象，包含路由链、目标链映射和默认链
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=default_chain,
    verbose=True,
)

# 测试1：询问如何为玫瑰浇水
print(chain.invoke("如何为玫瑰浇水？"))

# 测试2：询问如何为婚礼场地装饰花朵
print(chain.invoke("如何为婚礼场地装饰花朵？"))

# 测试3：询问如何区分阿豆和罗豆（这个问题将被默认链处理）
print(chain.invoke("如何区分阿豆和罗豆？"))
