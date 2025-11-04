from typing import Optional
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI  # OpenAI和LM Studio均使用OpenAI兼容接口


def get_llm(
    llm_type: str,
    model: str,
    temperature: float = 0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> OllamaLLM | ChatOpenAI:
    """
    通用LLM实例化函数，支持Ollama、LM Studio、OpenAI
    
    参数说明：
    - llm_type: 模型类型，可选值："ollama"、"lm_studio"、"openai"
    - model: 模型名称（如"qwen2.5:14b"、"gpt-3.5-turbo"、"mistral"等）
    - temperature: 生成温度（0表示确定性输出）
    - api_key: API密钥（仅OpenAI需要，LM Studio可留空或填任意值）
    - base_url: 接口地址（各平台默认地址不同，可自定义）
    """
    if llm_type == "ollama":
        # Ollama默认地址：http://localhost:11434
        return OllamaLLM(
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature
        )
    
    elif llm_type == "lm_studio":
        # LM Studio兼容OpenAI接口，默认地址：http://localhost:1234/v1
        # 无需真实API密钥，可填任意字符串（如"lm-studio"）
        return ChatOpenAI(
            model=model,
            base_url=base_url or "http://localhost:1234/v1",
            api_key=api_key or "lm-studio",  # LM Studio无需真实密钥
            temperature=temperature
        )
    
    elif llm_type == "openai":
        # OpenAI官方地址：https://api.openai.com/v1
        if not api_key:
            raise ValueError("OpenAI需要传入有效的api_key")
        return ChatOpenAI(
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"不支持的llm_type：{llm_type}，可选值：'ollama'、'lm_studio'、'openai'")