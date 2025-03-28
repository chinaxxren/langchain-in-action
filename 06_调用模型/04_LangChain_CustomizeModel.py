from typing import Optional, List, Mapping, Any
from langchain.llms.base import LLM  # 导入LangChain基础LLM类
from langchain_community.llms import Ollama  # 导入Ollama支持
from pydantic import Field  # 导入Pydantic的Field用于类型验证

# 自定义的LLM类，继承自LangChain的基础LLM类
class CustomLLM(LLM):
    # 使用 Pydantic Field 定义类属性，支持类型检查和默认值
    model_name: str = Field(default="llama3.2")  # 模型名称，默认使用llama3.2
    ollama: Optional[Ollama] = Field(default=None)  # Ollama客户端实例

    def __init__(self, model_name: str = "llama3.2", **kwargs):
        """初始化CustomLLM实例
        Args:
            model_name: 要使用的Ollama模型名称
            kwargs: 其他参数
        """
        super().__init__()
        super().__init__(model_name=model_name, **kwargs)
        # 初始化Ollama客户端，设置服务地址和模型
        self.ollama = Ollama(
            base_url="http://localhost:11434",  # Ollama服务地址
            model=self.model_name  # 使用指定的模型
        )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """执行实际的模型调用
        Args:
            prompt: 输入的提示文本
            stop: 停止词列表（可选）
        Returns:
            str: 模型生成的回复
        """
        response = self.ollama.invoke(
            f"Q: {prompt}\nA: ",  # 格式化提示文本
            stop=stop  # 传入停止词
        )
        return response.strip()  # 去除首尾空白字符

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """返回模型的标识参数
        Returns:
            包含模型名称的字典
        """
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识
        Returns:
            自定义的LLM类型名称
        """
        return "custom_ollama"

# 测试不同模型的效果
def test_models():
    """测试不同Ollama模型的回答效果"""
    # 测试 llama3.2 模型
    llm_llama = CustomLLM("llama3.2")
    result_llama = llm_llama("昨天有一个客户抱怨他买了花给女朋友之后，两天花就枯了，你说作为客服我应该怎么解释？")
    print("=== Llama3.2 的回答 ===")
    print(result_llama)
    print("\n")

    # 测试 Mistral 模型
    llm_mistral = CustomLLM("MHKetbi/Mistral-Small3.1-24B-Instruct-2503:q8_0")
    result_mistral = llm_mistral("昨天有一个客户抱怨他买了花给女朋友之后，两天花就枯了，你说作为客服我应该怎么解释？")
    print("=== Mistral 的回答 ===")
    print(result_mistral)

# 主程序入口
if __name__ == "__main__":
    test_models()