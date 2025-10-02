import re
import time
import torch
from accelerate.commands.config.config_args import cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
import os
import logging
from utils import save_to_llama_format
import ollama
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError
from openai import OpenAI
import sys
from typing import List, Dict, Tuple, Any, Optional
sys.stdout.reconfigure(encoding='utf-8')


# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Commander:
    def __init__(self, api_key, model_name="gpt-4.1", api_version="2024-12-01-preview", data_file="commander_data.json", cache_dir="./jsons/"):
        """
        Initializes the Commander class to interact with the GPT API.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.api_version = None
        self.base_url = "https://genai.hkbu.edu.hk/api/v0/rest"
        self.data_file = os.path.join(cache_dir, data_file)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Ensure data file exists
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w") as f:
                json.dump([], f)


    def _submit(self, conversation):
        """
        Sends the conversation to the GPT API and retrieves the response.
        """
        url = f"{self.base_url}/deployments/{self.model_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        payload = {
            "messages": conversation,
            "temperature": 0.7,
            "max_tokens": 512,
            "top_p": 1,
            "stream": False
        }

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json(), 200
        else:
            return {"error": f"API Request Failed: {response.status_code}, {response.text}"}, response.status_code

    def chat(self, prompt, instruction=""):
        conversation = []
        if instruction.strip():
            conversation.append({"role": "system", "content": "You are a drug discovery assistant. Your answer will be used as input for other models. Please be concise, short(at most 20 words) and clear. ALL in English."})
        conversation.append({"role": "user", "content": prompt})

        max_retries = 5
        retry_count = 0


        while retry_count < max_retries:
            response, status_code = self._submit(conversation)
            print("=====================\n"+response+"\n=============================")
            if status_code == 200:
                if "choices" in response and response["choices"]:
                    save_to_llama_format(instruction, prompt, response["choices"][0]["message"]["content"], self.data_file)
                    return response["choices"][0]["message"]["content"]
                else:
                    return response.get("error", "❌ Error: Invalid response format.")
            else:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Request failed. Retrying in 3 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(3)
        raise RuntimeError("❌ Maximum retry attempts reached. Unable to get a valid response from the API.")


class DeepSeekOllama:
    def __init__(self, model_name="deepseek-r1:14b", data_file="commander_data.json", cache_dir="./jsons/"):
        """
        Initializes the Commander class for local inference using Ollama and DeepSeek-R1-14B.

        Args:
            model_name (str): Name of the locally available DeepSeek model in Ollama.
            data_file (str): Filepath for storing conversation data.
        """
        self.model_name = model_name
        self.data_file = cache_dir+data_file

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Ensure data file exists
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w") as f:
                json.dump([], f)

    def chat(self, prompt, instruction="",no_need_think=True):
        """
        Generates a response using DeepSeek-R1-14B via Ollama.

        Args:
            prompt (str): User input message.
            instruction (str): Optional instruction to guide the model.

        Returns:
            str: Generated response from DeepSeek-R1-14B.
        """
        input_text = f"{instruction}\n{prompt}".strip()

        response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": input_text}])

        if "message" in response:
            response_text = response["message"]["content"]
            save_to_llama_format(instruction, prompt, response_text, self.data_file)
        else:
            response_text = "Error: No response from model."
        if no_need_think:
            response_text = re.sub(r"<think>.*?</think>\n?", "", response_text, flags=re.DOTALL)
        return response_text


class Deepseek_official:
    def __init__(self, key="deepseek-r1:14b", data_file="commander_data.json", cache_dir="./jsons/"):
        """
        Initializes the Commander class for local inference using Ollama and DeepSeek-R1-14B.

        Args:
            model_name (str): Name of the locally available DeepSeek model in Ollama.
            data_file (str): Filepath for storing conversation data.
        """
        self.data_file = cache_dir+data_file
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Ensure data file exists
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w") as f:
                json.dump([], f)

    def chat(self, prompt, instruction="", no_need_think=True):
        """
        Generates a response using DeepSeek-R1-14B via Ollama.

        Args:
            prompt (str): User input message.
            instruction (str): Optional instruction to guide the model.

        Returns:
            str: Generated response from DeepSeek-R1-14B.
        """
        
        input_text = f"{instruction}\n{prompt}".strip()
        # Truncate input_text if it exceeds a certain max length (e.g., 4096 characters)
        max_input_length = 200
        if len(input_text) > max_input_length:
            input_text = input_text[:max_input_length]
        response = self.client.chat.completions.create(model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "You are a drug discovery assistant. Your answer will be used as input for other models. Please be concise, short(at most 20 words) and clear. ALL in English."},
                            {"role": "user", "content": input_text},
                            ],
                        stream=False
                    )
        # print(response)
        if response:
            response_text = response.choices[0].message.content
            save_to_llama_format(instruction, prompt, response_text, self.data_file)
        else:
            response_text = "Error: No response from model."
        return response_text

from typing import List, Dict, Optional, Union
from openai import AzureOpenAI

class CommanderAzure:
    """非流式，默认关闭推理（reasoning）。"""

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://22481087-5850-resource.cognitiveservices.azure.com/",
        deployment: str = "gpt-5-nano",                 # 你的 Azure 部署名
        api_version: str = "2024-12-01-preview",
        data_file: str = "commander_data.json",
        cache_dir: str = "./jsons/",
        max_completion_tokens: int = 160,               # 小预算：快
        reasoning_effort: str = "minimal",              # 'minimal' | 'low' | 'medium' | 'high'
        verbosity: str = "low",                         # 可选：'low' | 'medium' | 'high'
        retries: int = 1,                               # 极简重试
    ):
        if not api_key:
            raise ValueError("api_key is required.")
        # from pdb import set_trace; set_trace()
        if api_version == 'None' or api_version is None:
            self.client = OpenAI(
                            base_url=f"{endpoint}",
                            api_key=api_key
                        )
        else:
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
            )
        self.deployment = deployment
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.retries = retries

        os.makedirs(cache_dir, exist_ok=True)
        self.data_file = os.path.join(cache_dir, data_file)
        if not os.path.exists(self.data_file):
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False)

        # 极简输出约束（中英都兼容）
        self.system_prompt = (
            "Be concise. Answer in ≤ 20 words. Plain text only. No preface. No lists. No explanations."
        )

    def chat(
        self,
        prompt: str,
        instruction: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        return_raw: bool = False,
    ) -> Union[str, object]:
        if 'Phi' in self.deployment:
            prompt = prompt[:500]
        # 组装 messages（仅 system + user，避免 developer/system 混用）
        if messages is None:
            msgs = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}]
            if instruction.strip():
                # 若需要附加约束，仍作为 system，保持“只用一种角色提示”
                msgs.insert(1, {"role": "system", "content": instruction.strip()})
        else:
            msgs = messages

        # 按文档使用的最小参数集合
        if '4o' in self.deployment or 'gpt-4o' in self.deployment or 'gpt-5-chat' in self.deployment:
            payload = dict(
            model=self.deployment,
            messages=msgs,
            max_completion_tokens=self.max_completion_tokens,  # Chat Completions 用这个
        )
        else:
            payload = dict(
                model=self.deployment,
                messages=msgs,
                max_completion_tokens=self.max_completion_tokens,  # Chat Completions 用这个
                reasoning_effort=self.reasoning_effort,            # GPT-5 支持 'minimal'
            )
        # 可选的“更简洁”控制；若不被后端识别会在异常里移除
        if self.verbosity:
            payload["verbosity"] = self.verbosity

        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.client.chat.completions.create(**payload)
                if return_raw:
                    return resp
                content = resp.choices[0].message.content if resp and resp.choices else ""

                # 若被长度截断且仍无内容，放大一点点预算补打一枪
                if (not content) and getattr(resp.choices[0], "finish_reason", "") == "length":
                    payload["max_completion_tokens"] = int(self.max_completion_tokens * 2)
                    resp = self.client.chat.completions.create(**payload)
                    content = resp.choices[0].message.content if resp and resp.choices else ""

                save_to_llama_format(instruction, prompt, content, self.data_file)
                return content or ""
            except Exception as e:
                last_err = e
                # 兼容老 SDK/区域：若不认识 verbosity/effort 字段，则移除后再试
                if "verbosity" in payload:
                    payload.pop("verbosity", None)
                    continue
                if "reasoning_effort" in payload:
                    payload.pop("reasoning_effort", None)
                    continue
                break
        raise RuntimeError(f"Request failed: {last_err}")


if __name__ == "__main__":
    # 最小可运行示例（建议把密钥放到环境变量 AZURE_OPENAI_KEY）
    api_key = "9EyK9OgHh2bHMM97joKUuQO43Sfv3k3NX6ilhVGlqAa55geOt5noJQQJ99BIACHYHv6XJ3w3AAAAACOG5XOm"
    cmd = CommanderAzure(api_key=api_key)
    print("GPT:")
    print(cmd.chat("Propose specific structural modifications for next-gen molecules in 20 words. Ensure suggestions are actionable and clear without extra context. Critique: Modifying polar groups can enhance affinity while maintaining solubility and permeability.\nKnowledge: Optimize binding affinity and lipophilicity to enhance drug efficacy and improve pharmacokinetic properties."))