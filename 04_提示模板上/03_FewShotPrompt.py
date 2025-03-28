from dotenv import load_dotenv  # 用于加载环境变量
from langchain_openai import OpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector.semantic_similarity import SemanticSimilarityExampleSelector  # 更新导入路径
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

load_dotenv()  # 加载 .env 文件中的环境变量

# 1. 创建一些示例
samples = [

  {
    "flower_type": "玫瑰",
    "occasion": "爱情",
    "ad_copy": "玫瑰，浪漫的象征，是你向心爱的人表达爱意的最佳选择。"
  },
  {
    "flower_type": "康乃馨",
    "occasion": "母亲节",
    "ad_copy": "康乃馨代表着母爱的纯洁与伟大，是母亲节赠送给母亲的完美礼物。"
  },
  {
    "flower_type": "百合",
    "occasion": "庆祝",
    "ad_copy": "百合象征着纯洁与高雅，是你庆祝特殊时刻的理想选择。"
  },
  {
    "flower_type": "向日葵",
    "occasion": "鼓励",
    "ad_copy": "向日葵象征着坚韧和乐观，是你鼓励亲朋好友的最好方式。"
  }
]

# 2. 创建一个提示模板
prompt_sample = PromptTemplate(input_variables=["flower_type", "occasion", "ad_copy"], 
                               template="鲜花类型: {flower_type}\n场合: {occasion}\n文案: {ad_copy}")
print(prompt_sample.format(**samples[1]))
print("*0" * 30)

# 3. 创建一个FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    examples=samples,
    example_prompt=prompt_sample,
    suffix="鲜花类型: {flower_type}\n场合: {occasion}",
    input_variables=["flower_type", "occasion"]
)
print(prompt.format(flower_type="野玫瑰", occasion="爱情"))
print("*1" * 30)

# 4. 把提示传递给大模型
model = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.7
)
result = model.invoke(prompt.format(flower_type="野玫瑰", occasion="爱情"))
print(result)
print("*2" * 30)

# 5. 使用示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    samples,
    OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    ),
    FAISS,
    k=1
)


# 创建一个使用示例选择器的FewShotPromptTemplate对象
prompt = FewShotPromptTemplate(
    example_selector=example_selector, 
    example_prompt=prompt_sample, 
    suffix="鲜花类型: {flower_type}\n场合: {occasion}", #意思是提交给大模型时候这段话显示在最后
    example_prompt=prompt_sample,
    input_variables=["flower_type", "occasion"]
)
print(prompt.format(flower_type="红玫瑰", occasion="爱情"))

### suffix 的特点

# 1. 它是模型最终需要回答的问题
# 2. 它会出现在所有示例之后
# 3. 它可以包含变量（使用 `{变量名}` 格式）
# 4. 这些变量必须在 `input_variables` 中声明

# 所以，FewShotPromptTemplate中的`suffix` 实际上就是在告诉模型："看完了上面的例子后，请针对这个新情况（suffix中描述的）给出回答"。



