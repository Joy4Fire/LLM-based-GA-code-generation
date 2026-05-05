from typing import Optional


def get_llm(
    llm_type: str,
    model: str,
    temperature: float = 0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 120,
    max_retries: int = 2,
):
    if llm_type == "ollama":
        from langchain_ollama import OllamaLLM

        return OllamaLLM(
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
        )

    if llm_type == "lm_studio":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            base_url=base_url or "http://localhost:1234/v1",
            api_key=api_key or "lm-studio",
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    if llm_type == "openai":
        from langchain_openai import ChatOpenAI

        if not api_key:
            raise ValueError("OpenAI mode requires api_key.")
        return ChatOpenAI(
            model=model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    raise ValueError(
        f"Unsupported llm_type: {llm_type}. Expected one of: 'ollama', 'lm_studio', 'openai'."
    )
