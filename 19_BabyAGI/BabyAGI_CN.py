# 这是一个基于 BabyAGI 框架的任务规划和执行系统，主要包含以下几个核心部分：

# 1. 核心组件：
# - TaskCreationChain : 负责生成新任务
# - TaskPrioritizationChain : 负责任务优先级排序
# - ExecutionChain : 负责执行具体任务
# - BabyAGI : 主控制器，协调各个组件工作
# # 初始化
# OBJECTIVE = "分析一下北京市今天的气候情况，写出鲜花储存策略。"
# max_iterations = 6  # 最大迭代次数

# # 循环执行
# while 任务未完成 and 未达到最大迭代次数:
#     1. 获取并执行当前任务
#     2. 存储任务结果
#     3. 基于结果创建新任务
#     4. 对任务列表重新排序
# 3. 关键特性：
# - 使用向量存储（FAISS）保存任务结果
# - 自动任务分解和优先级管理
# - 任务执行过程可追踪
# - 支持最大迭代次数限制
# 4. 技术栈：
# - LangChain 框架
# - OpenAI API
# - FAISS 向量数据库
# - Pydantic 数据模型

# 导入环境变量管理库
from dotenv import load_dotenv
from collections import deque
from typing import Dict, List, Optional, Any
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings  # 更新
from langchain_openai import OpenAI  # 更新
from langchain.chains.base import Chain
from langchain_community.vectorstores import FAISS  # 更新
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore  # 更新
from langchain.llms.base import BaseLLM  # 添加BaseLLM导入
from pydantic import Field, BaseModel  # 添加Field和BaseModel导入
from langchain.vectorstores.base import VectorStore  # 添加VectorStore导入

load_dotenv()

# 定义嵌入模型
embeddings_model = OpenAIEmbeddings()
# 初始化向量存储
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
# 修复FAISS初始化方式
vectorstore = FAISS(
    embedding_function=embeddings_model,
    index=index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={},
)


# 任务生成链
class TaskCreationChain(LLMChain):
    """负责生成任务的链"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """从LLM获取响应解析器"""
        task_creation_template = (
            "You are a task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: {objective},"
            " The last completed task has the result: {result}."
            " This result was based on this task description: {task_description}."
            " These are incomplete tasks: {incomplete_tasks}."
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as an array."
            "The maximum length of the array is three"  # 限制任务数量,我添加的
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# 任务优先级链
class TaskPrioritizationChain(LLMChain):
    """负责任务优先级排序的链"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """从LLM获取响应解析器"""
        task_prioritization_template = (
            "You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: {task_names}."
            " Consider the ultimate objective of your team: {objective}."
            " Do not remove any tasks. Return the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=["task_names", "next_task_id", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# 任务执行链
class ExecutionChain(LLMChain):
    """负责执行任务的链"""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        """从LLM获取响应解析器"""
        execution_template = (
            "You are an AI who performs one task based on the following objective: {objective}."
            " Take into account these previously completed tasks: {context}."
            " Your task: {task}."
            " Response:"
        )
        prompt = PromptTemplate(
            template=execution_template,
            input_variables=["objective", "context", "task"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


# 根据当前任务的执行结果生成新任务
# - 将未完成任务转换为字符串
# - 调用任务创建链生成新任务
# - 处理响应并返回新任务列表
def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """获取下一个任务。"""

    # 将未完成的任务列表转换为逗号分隔的字符串
    incomplete_tasks = ", ".join(task_list)

    # 调用任务创建链，根据当前任务结果生成新任务
    response = task_creation_chain.invoke(
        {
            "result": result,  # 当前任务的执行结果
            "task_description": task_description,  # 当前任务的描述
            "incomplete_tasks": incomplete_tasks,  # 未完成的任务列表
            "objective": objective,
        }  # 总体目标
    )
    # 从响应中提取文本内容
    response_text = response.get("text", "")
    # 将响应文本按行分割，每行代表一个新任务
    new_tasks = response_text.split("\n")
    # 过滤空行并将每个任务转换为字典格式返回
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


# 对任务列表进行优先级排序
# - 提取任务名称
# - 调用任务优先级链进行排序
# - 解析响应，处理任务ID和名称
# - 返回排序后的任务列表
def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """对任务进行优先级排序。"""
    # 提取所有任务的名称
    task_names = [t["task_name"] for t in task_list]
    # 计算下一个任务ID
    next_task_id = int(this_task_id) + 1
    # 调用任务优先级链，对任务进行重新排序
    response = task_prioritization_chain.invoke(
        {
            "task_names": task_names,  # 所有任务名称列表
            "next_task_id": next_task_id,  # 下一个任务ID
            "objective": objective,
        }  # 总体目标
    )
    # 从响应中提取文本内容
    response_text = response.get("text", "")
    # 将响应文本按行分割，每行代表一个优先级排序后的任务
    new_tasks = response_text.split("\n")
    # 初始化优先级排序后的任务列表
    prioritized_task_list = []
    # 处理每个任务字符串
    for task_string in new_tasks:
        # 跳过空行
        if not task_string.strip():
            continue
        # 去除首尾空白字符
        task_string = task_string.strip()
        # 按第一个点分割，分离任务ID和任务名称
        task_parts = task_string.strip().split(".", 1)
        # 确保分割后有两部分
        if len(task_parts) == 2:
            # 提取任务ID并去除空白字符
            task_id = task_parts[0].strip()
            # 移除任务ID中的#号
            task_id = task_id.replace("#", "")
            # 确保任务ID不为空，如果为空则使用默认值
            if not task_id:
                task_id = str(next_task_id)
                next_task_id += 1
            # 提取任务名称并去除空白字符
            task_name = task_parts[1].strip()
            # 将任务添加到优先级排序后的列表中
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    # 返回优先级排序后的任务列表
    return prioritized_task_list


# 从向量存储中获取与查询最相关的前k个任务
def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """获取与查询最相关的前k个任务。"""
    # 使用向量存储的相似度搜索功能
    results = vectorstore.similarity_search_with_score(query, k=k)
    # 如果没有找到相关任务，返回空列表
    if not results:
        return []
    # 按相似度得分排序并提取任务信息
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    # 返回任务描述列表
    return [str(item.metadata["task"]) for item in sorted_results]


# 执行具体任务的函数
def execute_task(
    vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
) -> str:
    """执行单个任务。"""
    # 获取与当前目标相关的历史任务作为上下文
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    # 调用执行链完成任务
    response = execution_chain.invoke(
        {
            "objective": objective,  # 总体目标
            "context": context,  # 相关历史任务上下文
            "task": task,
        }  # 当前需要执行的任务
    )
    # 返回任务执行结果
    return response.get("text", "")


# BabyAGI类的辅助方法
def add_task(self, task: Dict):
    """添加新任务到任务列表。"""
    self.task_list.append(task)


def print_task_list(self):
    """打印当前任务列表。"""
    print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
    for t in self.task_list:
        print(str(t["task_id"]) + ": " + t["task_name"])


def print_next_task(self, task: Dict):
    """打印即将执行的任务。"""
    print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
    print(str(task["task_id"]) + ": " + task["task_name"])


def print_task_result(self, result: str):
    """打印任务执行结果。"""
    print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
    print(result)


# BabyAGI 主类
class BabyAGI(Chain, BaseModel):
    """BabyAGI代理的控制器模型"""

    # 任务队列，存储所有待执行的任务
    task_list: deque = Field(default_factory=deque)

    # 任务创建链，负责根据已完成任务的结果生成新任务
    task_creation_chain: TaskCreationChain = Field(...)

    # 任务优先级链，负责对任务列表进行优先级排序
    task_prioritization_chain: TaskPrioritizationChain = Field(...)

    # 任务执行链，负责执行具体任务
    execution_chain: ExecutionChain = Field(...)

    # 任务ID计数器，用于为新任务分配唯一ID
    task_id_counter: int = Field(1)

    # 向量存储，用于存储任务结果并支持相似性搜索
    vectorstore: VectorStore = Field(init=False)

    # 最大迭代次数，控制系统运行的轮数，防止无限循环
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        # 从输入中获取目标任务
        objective = inputs["objective"]
        # 获取第一个任务，如果没有指定则默认为"Make a todo list"
        first_task = inputs.get("first_task", "Make a todo list")
        # 将第一个任务添加到任务列表中
        self.add_task({"task_id": 1, "task_name": first_task})
        # 初始化迭代计数器
        num_iters = 0
        # 开始主循环
        while True:
            # 检查任务列表是否有任务
            if self.task_list:
                # 打印当前任务列表
                self.print_task_list()

                # 步骤1: 获取优先级最高的任务（队列头部的任务）
                task = self.task_list.popleft()
                # 打印即将执行的任务
                self.print_next_task(task)

                # 步骤2: 执行任务
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                # 确保task_id是整数，处理可能的格式问题
                try:
                    # 移除任务ID中的#号并转换为整数
                    task_id_str = str(task["task_id"]).replace("#", "")
                    # 如果ID为空则使用当前计数器值
                    this_task_id = (
                        int(task_id_str) if task_id_str else self.task_id_counter
                    )
                except ValueError:
                    # 如果转换失败，使用当前计数器值
                    this_task_id = self.task_id_counter

                # 打印任务执行结果
                self.print_task_result(result)

                # 步骤3: 将结果存储到向量数据库中
                result_id = f"result_{task['task_id']}_{num_iters}"
                # 添加文本到向量存储，包含任务元数据
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # 步骤4: 根据执行结果创建新任务
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )

                # 为每个新任务分配ID并添加到任务列表
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)

                # 步骤5: 对任务列表重新排序
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
                
            # 增加迭代计数
            num_iters += 1
            # 检查是否达到最大迭代次数
            if self.max_iterations is not None and num_iters == self.max_iterations:
                # 打印任务结束信息
                print(
                    "\033[91m\033[1m" + "\n*****TASK ENDING*****\n" + "\033[0m\033[0m"
                )
                # 跳出循环
                break

        # 返回空字典
        return {}

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vectorstore: VectorStore, verbose: bool = False, **kwargs
    ) -> "BabyAGI":
        """初始化 BabyAGI 控制器。"""
        # 创建任务生成链实例，用于根据已完成任务的结果生成新任务
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)

        # 创建任务优先级链实例，用于对任务列表进行优先级排序
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )

        # 创建任务执行链实例，用于执行具体任务
        execution_chain = ExecutionChain.from_llm(llm, verbose=verbose)

        # 返回初始化后的 BabyAGI 实例
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=execution_chain,
            vectorstore=vectorstore,
            **kwargs,
        )


# 主执行部分
if __name__ == "__main__":
    # 定义系统的总体目标
    # OBJECTIVE = "分析一下北京市今天的气候情况，写出鲜花储存策略。全部用中文回答。"
    OBJECTIVE = "写出鲜花储存策略。全部用中文回答。"

    # 初始化 OpenAI 语言模型，temperature=0 表示输出最确定的结果
    llm = OpenAI(temperature=0)

    # 设置是否输出详细日志
    verbose = False

    # 设置最大迭代次数，控制系统运行的轮数
    max_iterations: Optional[int] = 6

    # 创建 BabyAGI 实例，初始化所有必要的组件
    baby_agi = BabyAGI.from_llm(
        llm=llm,  # 语言模型
        vectorstore=vectorstore,  # 向量存储
        verbose=verbose,  # 是否输出详细日志
        max_iterations=max_iterations,  # 最大迭代次数
    )

    # 启动系统，开始执行任务
    baby_agi.invoke({"objective": OBJECTIVE})
