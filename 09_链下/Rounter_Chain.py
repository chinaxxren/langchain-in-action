import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import RouterOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv()

# 构建两个场景的模板
flower_care_template = """
你是一个经验丰富的园丁，擅长解答关于养花育花的问题。
下面是需要你来回答的问题:
{input}
"""

flower_deco_template = """
你是一位网红插花大师，擅长解答关于鲜花装饰的问题。
下面是需要你来回答的问题:
{input}
"""

# 构建提示信息
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
    }
]

# 初始化语言模型
llm = OpenAI()

# 创建链的映射
chains = {}
for info in prompt_infos:
    prompt = PromptTemplate(
        template=info['template'],
        input_variables=["input"]
    )
    chain = prompt | llm
    chains[info["key"]] = chain

# 创建路由器链
destinations = [f"{p['key']}: {p['description']}" for p in prompt_infos]
router_template = """Given a raw text input to a language model select the model prompt best suited for the input.
You will be given the names of the available prompts and a description of what the prompt is best suited for.

<< PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT >>
Return the name of the most suitable prompt from the available prompts above. If none are suitable, return "DEFAULT".
"""

router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    partial_variables={"destinations": "\n".join(destinations)}
)

router_chain = router_prompt | llm

# 创建默认链
default_prompt = PromptTemplate(
    template="Question: {input}\nAnswer:",
    input_variables=["input"]
)
default_chain = default_prompt | llm

# 组合链
def route_and_run(input_text: str):
    # 确定路由
    destination = router_chain.invoke({"input": input_text})
    destination = destination.strip().lower()
    
    # 选择合适的链
    if destination in chains:
        return chains[destination].invoke({"input": input_text})
    return default_chain.invoke({"input": input_text})

# 测试代码
test_inputs = [
    "如何为玫瑰浇水？",
    "如何为婚礼场地装饰花朵？",
    "如何区分阿豆和罗豆？"
]

for test_input in test_inputs:
    print(f"\n问题: {test_input}")
    print("回答:", route_and_run(test_input))
    print("-" * 50)