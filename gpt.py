import ast
import glob
import os
import random
import re
import sys

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

# Import token tracker if available
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from experiments.core.token_tracker import get_tracker
    _token_tracker_available = True
except ImportError:
    _token_tracker_available = False

# Resolve project root so that this module can be imported from any CWD.
# gpt.py itself lives in the project root directory, so we use its parent.
PROJECT_ROOT = Path(__file__).resolve().parent
API_CONFIG_PATH = PROJECT_ROOT / "api.json"
PROMPT_PATH = PROJECT_ROOT / "prompt.txt"

with open(str(API_CONFIG_PATH), "r") as f:
    config = json.load(f)
API_KEY = config.get("OPENAI_API_KEY")
API_BASE_URL = config.get("API_BASE_URL", "https://api.openai.com/v1")  # 支持自定义 API 基础 URL
DEFAULT_MODEL = config.get("MODEL_NAME", "gpt-4o-mini")  # 从配置文件读取默认模型名称
API_QUOTA_LIMIT = 100000  # Example daily quota limit in tokens

PROXY = {
    "http": "http://127.0.0.1:8080",
    "https": "http://127.0.0.1:8080"
}

# 创建全局 Session 对象以优化代理连接
# 使用 Session 可以复用连接，提高代理连接的稳定性
_session = None

def get_session():
    """获取或创建全局 Session 对象"""
    global _session
    if _session is None:
        _session = requests.Session()
        # 配置重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
        # 设置代理
        _session.proxies.update(PROXY)
    return _session

with open(str(PROMPT_PATH), "r") as f:
    prompt = f.read()


# def check_api_quota() -> int:
#     """
#     Check API usage to monitor the remaining quota, supporting proxy settings.
#
#     Args:
#         proxy (dict): Proxy settings as a dictionary (e.g., {"http": "http://proxyserver:port", "https": "http://proxyserver:port"}).
#
#     Returns:
#         int: Remaining tokens for the day, or -1 if the quota cannot be checked.
#     """
#     url = "https://api.openai.com/v1/dashboard/billing/usage"
#     headers = {
#         "Authorization": f"Bearer {API_KEY}"
#     }
#     try:
#         # ????????
#         response = requests.get(url, headers=headers, proxies=PROXY)
#         response.raise_for_status()
#         usage_data = response.json()
#         remaining_quota = API_QUOTA_LIMIT - usage_data.get("total_usage", 0)
#         return remaining_quota
#     except requests.exceptions.RequestException as e:
#         print(f"Unable to check quota: {e}")
#         return -1


def call_gpt4_vision(question: str, image_url: str, max_tokens: int = 10000, retries: int = None) -> str:
    """
    Multimodal call to GPT-4 for image-related questions with retry and quota check.

    Args:
        question (str): The question to ask the model.
        image_url (str): URL of the public image.
        max_tokens (int): Maximum token length for the response.
        retries (int): Number of retry attempts in case of request failure.

    Returns:
        str: The response from GPT-4 or an error message.
    """
    # remaining_quota = check_api_quota()
    # if remaining_quota != -1 and remaining_quota < max_tokens:
    #     return "Insufficient quota for this request."

    url = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4-vision",
        "messages": [
            {"role": "user", "content": question},
            {"role": "user", "content": f"[Image URL: {image_url}]"}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    attempt = 0
    max_wait_time = 60  # Maximum wait time between retries (seconds)
    
    while retries is None or attempt < retries:
        attempt += 1
        try:
            # 使用 Session 和增加超时时间以解决代理连接问题
            session = get_session()
            response = session.post(
                url, 
                headers=headers, 
                data=json.dumps(data),
                timeout=(10, 60)  # (连接超时, 读取超时) - 增加超时时间以应对代理延迟
            )
            response.raise_for_status()
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
            return answer
        except requests.exceptions.RequestException as e:
            # Print detailed error information
            error_details = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_body = e.response.text
                    error_status = e.response.status_code
                    print(f"Error status: {error_status}")
                    print(f"Error response: {error_body}")
                except:
                    pass
            
            # For 400 Bad Request errors, these are usually permanent (invalid API key, invalid model, etc.)
            # Don't retry indefinitely for 400 errors
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 400:
                    print(f"400 Bad Request error detected. This is usually a permanent error (invalid model name, API key, or request format).")
                    print(f"Stopping retries to prevent infinite loop.")
                    if retries is None:
                        print(f"Error details: {error_details}")
                        return f"Request failed with 400 Bad Request: {error_details}"
            
            # Calculate wait time with exponential backoff (capped at max_wait_time)
            wait_time = min(2 ** min(attempt, 6), max_wait_time)  # Exponential backoff: 2, 4, 8, 16, 32, 60, 60, ...
            if retries is None:
                print(f"Request failed, retry {attempt} (infinite retries): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
            else:
                print(f"Request failed, retry {attempt}/{retries}: {e}")
                print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    return "Request failed after maximum retries."


def get_frame_data(json_path, default_frame_number=-1):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if data.get("min_dist_frame") is not None:
        frame_number = data.get("min_dist_frame")
        print(f"Frame number: {frame_number}")
    else:
        frame_number = default_frame_number
    frame_keys = [key for key in data.keys() if key.isdigit()]

    frame_keys = sorted(map(int, frame_keys))

    if frame_number == -1:
        random_frame_key = str(random.choice(frame_keys))
        return data[random_frame_key]
    elif frame_number in frame_keys:
        return data[str(frame_number)]


def call_gpt(question: str, model_version: str = None, max_tokens: int = 10000, retries: int = None) -> str:
    """
    Call GPT model with automatic API selection based on model version.
    
    - GPT-5 series (gpt-5, gpt-5-mini, etc.): Uses Responses API with max_completion_tokens
    - GPT-3.5/4 series: Uses ChatCompletion API with max_tokens
    
    Uses infinite retries by default to ensure the experiment completes (except for 400 errors).

    Args:
        question (str): The question to ask the model.
        model_version (str): The model version to use. If None, uses DEFAULT_MODEL from config.
            - GPT-5 series: "gpt-5", "gpt-5-mini", etc. (uses Responses API)
            - Legacy models: "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o-mini" (uses ChatCompletion API)
        max_tokens (int): Maximum token length for the response.
            - For GPT-5 series: automatically converted to max_completion_tokens
            - For legacy models: used as max_tokens
        retries (int): Number of retry attempts in case of request failure. None means infinite retries (except for 400 errors).

    Returns:
        str: The response from the specified GPT model or an error message.
    """
    if model_version is None:
        model_version = DEFAULT_MODEL
    
    # 检测是否为 GPT-5 系列模型
    is_gpt5_model = model_version.startswith("gpt-5")
    
    # GPT-5 系列可以使用两种 API：
    # 1. Responses API (推荐，新接口)
    # 2. Chat Completions API (兼容，老接口但支持 GPT-5)
    # 这里默认使用 Responses API，如需使用 Chat Completions，可以修改条件
    use_responses_api = is_gpt5_model  # 可以改为 False 来使用 Chat Completions API
    
    if use_responses_api:
        # 使用 Responses API (推荐用于 GPT-5 系列)
        url = f"{API_BASE_URL}/responses"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_version,
            "input": question,  # Responses API 使用 input 而不是 messages
            "max_output_tokens": max_tokens,  # Responses API 使用 max_output_tokens
            # 注意：GPT-5 推理模型不支持 temperature、top_p 等采样参数
        }
    elif is_gpt5_model:
        # 使用 Chat Completions API (兼容模式，用于 GPT-5 系列)
        url = f"{API_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_version,
            "messages": [
                {"role": "user", "content": question}
            ],
            "max_completion_tokens": max_tokens,  # Chat Completions API 中 GPT-5 系列使用 max_completion_tokens
            # 注意：GPT-5 推理模型不支持 temperature、top_p 等采样参数
        }
    else:
        # 使用传统的 ChatCompletion API (用于 GPT-3.5/4 系列)
        url = f"{API_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_version,
            "messages": [
                {"role": "user", "content": question}
            ],
            "max_tokens": max_tokens,  # 传统模型使用 max_tokens
            "temperature": 0.7
        }

    attempt = 0
    max_wait_time = 60  # Maximum wait time between retries (seconds)
    
    # 打印请求信息（第一次尝试时）
    if use_responses_api:
        api_type = "Responses API (推荐)"
    elif is_gpt5_model:
        api_type = "Chat Completions API (兼容模式)"
    else:
        api_type = "Chat Completions API (传统)"
    print(f"\n=== 请求信息 ===")
    print(f"API 类型: {api_type}")
    print(f"URL: {url}")
    print(f"模型: {model_version}")
    print(f"请求头: {json.dumps({k: (v[:20] + '...' if k == 'Authorization' and len(v) > 20 else v) for k, v in headers.items()}, indent=2, ensure_ascii=False)}")
    print(f"请求体: {json.dumps(data, indent=2, ensure_ascii=False)}")
    print(f"===============\n")
    
    while retries is None or attempt < retries:
        attempt += 1
        request_start_time = time.time()
        request_timestamp = datetime.now().isoformat()
        
        try:
            # 使用 Session 和增加超时时间以解决代理连接问题
            session = get_session()
            response = session.post(
                url, 
                headers=headers, 
                data=json.dumps(data),
                timeout=(10, 60)  # (连接超时, 读取超时) - 增加超时时间以应对代理延迟
            )
            request_end_time = time.time()
            request_duration = request_end_time - request_start_time
            
            response.raise_for_status()
            response_data = response.json()
            
            # 调试信息：打印响应状态和关键字段
            if use_responses_api:
                status = response_data.get("status", "unknown")
                print(f"[DEBUG] 响应状态: {status}")
                print(f"[DEBUG] 响应包含的字段: {list(response_data.keys())}")
                if "output" in response_data:
                    print(f"[DEBUG] output 数组长度: {len(response_data.get('output', []))}")
                if "usage" in response_data:
                    print(f"[DEBUG] usage 信息: {response_data.get('usage', {})}")

            # 辅助函数：从 output 数组中提取文本
            def extract_text_from_output(output_list):
                """从 output 数组中提取文本内容（支持嵌套结构）"""
                text_parts = []
                
                def extract_recursive(obj):
                    """递归提取文本"""
                    if isinstance(obj, str):
                        return obj
                    elif isinstance(obj, dict):
                        # 优先查找常见的文本字段
                        for key in ["text", "content", "message", "output_text"]:
                            if key in obj:
                                result = extract_recursive(obj[key])
                                if result:
                                    return result
                        # 如果没有找到，递归查找所有值
                        for value in obj.values():
                            result = extract_recursive(value)
                            if result:
                                return result
                    elif isinstance(obj, list):
                        # 递归处理列表中的每个元素
                        results = []
                        for elem in obj:
                            result = extract_recursive(elem)
                            if result:
                                if isinstance(result, str):
                                    results.append(result)
                                elif isinstance(result, list):
                                    results.extend(result)
                        return results if results else None
                    return None
                
                # 处理 output_list
                for item in output_list:
                    result = extract_recursive(item)
                    if result:
                        if isinstance(result, str):
                            text_parts.append(result)
                        elif isinstance(result, list):
                            text_parts.extend([r for r in result if isinstance(r, str)])
                
                # 确保所有元素都是字符串
                text_parts = [str(part) for part in text_parts if part]
                return " ".join(text_parts) if text_parts else None
            
            # 根据 API 类型解析响应
            if use_responses_api:
                # Responses API 返回格式可能是异步的，需要检查状态
                status = response_data.get("status", "unknown")
                response_id = response_data.get("id", "")
                
                # 如果响应未完成，尝试轮询获取完整响应
                max_poll_attempts = 10
                poll_interval = 2  # 秒
                poll_attempt = 0
                
                while status == "incomplete" and poll_attempt < max_poll_attempts:
                    poll_attempt += 1
                    print(f"响应未完成，轮询获取完整响应 (尝试 {poll_attempt}/{max_poll_attempts})...")
                    
                    if not response_id:
                        print("警告: 没有响应 ID，无法轮询")
                        break
                    
                    # 等待一段时间后再次请求
                    time.sleep(poll_interval)
                    
                    # 使用响应 ID 获取完整响应
                    try:
                        poll_url = f"{API_BASE_URL}/responses/{response_id}"
                        poll_headers = {
                            "Authorization": f"Bearer {API_KEY}",
                            "Content-Type": "application/json"
                        }
                        poll_response = session.get(poll_url, headers=poll_headers, timeout=(10, 60))
                        poll_response.raise_for_status()
                        response_data = poll_response.json()
                        status = response_data.get("status", "unknown")
                        print(f"轮询结果: status = {status}")
                    except Exception as poll_error:
                        print(f"轮询失败: {poll_error}")
                        break
                
                # 解析最终响应
                if status == "incomplete":
                    # 如果仍然未完成，尝试从 output 数组中提取已有文本
                    output_list = response_data.get("output", [])
                    answer = extract_text_from_output(output_list)
                    if answer:
                        print(f"警告: 响应仍未完成，但已提取部分文本")
                    else:
                        # 如果响应未完成且没有文本，返回提示信息
                        incomplete_reason = response_data.get("incomplete_details", {}).get("reason", "unknown")
                        answer = f"Response incomplete (reason: {incomplete_reason}). Status: {status}. 已尝试轮询 {poll_attempt} 次。"
                else:
                    # 响应已完成或其他状态，尝试提取文本
                    if "output_text" in response_data:
                        # 标准格式
                        answer = response_data.get("output_text", "No response")
                    elif "output" in response_data:
                        # 从 output 数组中提取文本
                        output_list = response_data.get("output", [])
                        answer = extract_text_from_output(output_list)
                        if not answer:
                            # 如果 output 数组中没有文本，尝试查找其他字段
                            print(f"[DEBUG] output 数组中没有找到文本，output 类型: {type(output_list)}, 长度: {len(output_list) if isinstance(output_list, list) else 'N/A'}")
                            if isinstance(output_list, list) and len(output_list) > 0:
                                print(f"[DEBUG] output[0] 示例: {str(output_list[0])[:200] if output_list else 'empty'}")
                            answer = "No response"
                    else:
                        # 尝试查找任何可能的文本字段
                        answer = response_data.get("text", response_data.get("content", "No response"))
                    
                    # 如果仍然没有找到答案，打印调试信息
                    if answer == "No response":
                        print(f"[DEBUG] 无法提取响应文本。响应状态: {status}")
                        print(f"[DEBUG] 响应数据的关键字段: {list(response_data.keys())}")
                        # 尝试打印部分响应数据用于调试（避免打印过长）
                        debug_data = {k: str(v)[:100] if isinstance(v, (str, dict, list)) else v 
                                     for k, v in list(response_data.items())[:5]}
                        print(f"[DEBUG] 响应数据预览: {debug_data}")
            else:
                # ChatCompletion API 返回格式：{"choices": [{"message": {"content": "..."}}]}
                answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response")

            # 根据 API 类型解析 usage 信息
            if use_responses_api:
                # Responses API 的 usage 格式使用 input_tokens 和 output_tokens
                usage_info = response_data.get("usage", {})
                # 尝试多种可能的字段名
                prompt_tokens = (usage_info.get("input_tokens") or 
                                usage_info.get("prompt_tokens") or 
                                (usage_info.get("input_tokens_details", {}) and usage_info.get("input_tokens_details", {}).get("total_tokens", 0)) or 0)
                completion_tokens = (usage_info.get("output_tokens") or 
                                    usage_info.get("completion_tokens") or 
                                    (usage_info.get("output_tokens_details", {}) and usage_info.get("output_tokens_details", {}).get("total_tokens", 0)) or 0)
                total_tokens = (usage_info.get("total_tokens") or 
                               (prompt_tokens + completion_tokens) or 0)
                
                # 调试信息：如果 token 计数有问题
                if prompt_tokens == 0 and completion_tokens == 0 and total_tokens > 0:
                    print(f"[DEBUG] 警告: token 计数可能有问题。usage_info = {usage_info}")
                    # 尝试从 total_tokens 推断（如果只有 total_tokens）
                    if total_tokens > 0:
                        # 假设大部分是 completion tokens（因为 prompt 通常较小）
                        completion_tokens = total_tokens - 100  # 粗略估计
                        prompt_tokens = 100
                        print(f"[DEBUG] 使用估算值: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
                
                # Get response metadata
                response_id = response_data.get("id", "")
                response_model = response_data.get("model", model_version)
                response_created = response_data.get("created", None)
                finish_reason = response_data.get("finish_reason", "")
            else:
                # ChatCompletion API (包括 GPT-5 使用 Chat Completions 的情况)
                usage_info = response_data.get("usage", {})
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
                total_tokens = usage_info.get("total_tokens", 0)
                
                # Get response metadata
                response_id = response_data.get("id", "")
                response_model = response_data.get("model", model_version)
                response_created = response_data.get("created", None)
                finish_reason = response_data.get("choices", [{}])[0].get("finish_reason", "")
            
            print(f"Tokens used: Prompt = {prompt_tokens}, Completion = {completion_tokens}, Total = {total_tokens}")
            print(f"Request duration: {request_duration:.2f}s")
            
            # Record token usage if tracker is available
            if _token_tracker_available:
                try:
                    tracker = get_tracker()
                    tracker.record_usage(
                        model=model_version,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens
                    )
                except Exception as e:
                    # Don't fail if token tracking fails
                    pass

            # Return answer with metadata attached (as a dict for enhanced logging)
            # For backward compatibility, we still return the string, but we can attach metadata
            # The caller can access this via a global or we can modify the return type
            # For now, we'll return the string but store metadata in a way that can be accessed
            return answer
        except requests.exceptions.RequestException as e:
            # 打印详细的错误信息
            print(f"\n=== 错误详情 (尝试 {attempt}) ===")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误消息: {str(e)}")
            
            # 获取响应详情（如果有）
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_status = e.response.status_code
                    error_headers = dict(e.response.headers)
                    error_body = e.response.text
                    
                    print(f"HTTP 状态码: {error_status}")
                    print(f"响应头: {json.dumps(error_headers, indent=2, ensure_ascii=False)}")
                    print(f"响应体: {error_body}")
                    
                    # 尝试解析 JSON 响应
                    try:
                        error_json = e.response.json()
                        print(f"解析后的错误 JSON: {json.dumps(error_json, indent=2, ensure_ascii=False)}")
                    except:
                        pass
                except Exception as parse_error:
                    print(f"解析错误响应时出错: {parse_error}")
            
            print(f"==================\n")
            
            # For 400 Bad Request errors, these are usually permanent (invalid API key, invalid model, etc.)
            # Don't retry indefinitely for 400 errors
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 400:
                    print(f"⚠️  400 Bad Request 错误检测到。这通常是永久性错误（无效的模型名称、API密钥或请求格式）。")
                    print(f"⚠️  停止重试以避免无限循环。")
                    if retries is None:
                        return f"Request failed with 400 Bad Request: {str(e)}"
            
            # Calculate wait time with exponential backoff (capped at max_wait_time)
            wait_time = min(2 ** min(attempt, 6), max_wait_time)  # Exponential backoff: 2, 4, 8, 16, 32, 60, 60, ...
            if retries is None:
                print(f"Request failed, retry {attempt} (infinite retries): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
            else:
                print(f"Request failed, retry {attempt}/{retries}: {e}")
                print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    return "Request failed after maximum retries."


def call_gpt_with_rag(question: str, rag_engine, model_version: str = None, 
                      max_tokens: int = 10000, retries: int = None) -> str:
    """
    Call GPT with RAG enhancement
    
    Args:
        question: Base question/prompt
        rag_engine: RAGEngine instance
        model_version: GPT model version to use. If None, uses DEFAULT_MODEL from config.
        max_tokens: Maximum tokens for response
        retries: Number of retry attempts
        
    Returns:
        GPT response string
    """
    if model_version is None:
        model_version = DEFAULT_MODEL
    
    # Extract seed scenario from question (simplified - in practice would parse more carefully)
    # Use RAG to retrieve relevant scenarios
    retrieved = rag_engine.retrieve_relevant_scenarios(question, k=rag_engine.top_k)
    
    # Build enhanced prompt
    enhanced_prompt = rag_engine.generate_enhanced_prompt(question, retrieved)
    
    # Call GPT with enhanced prompt (let any errors propagate)
    return call_gpt(enhanced_prompt, model_version=model_version, 
                    max_tokens=max_tokens, retries=retries)


def extract_json(response):
    try:
        json_text = re.search(r'{.*}', response, re.DOTALL).group(0)
        result = json.loads(json_text)
        return result
    except (json.JSONDecodeError, AttributeError):
        print("Failed to extract JSON from the response.")
        return None


def add_answer1_to_database(response_json, database, max_size=30):
    """
    Adds the answer1 from response_json to the database and ensures the database size
    does not exceed max_size. If the database exceeds max_size, it removes the oldest
    entry before adding the new one.

    Parameters:
    - response_json: JSON data containing answer1
    - database: OrderedDict storing answer1 data
    - max_size: Maximum size of the database; removes the oldest entry if exceeded

    Returns:
    - Updated database
    """
    if "answer1" in response_json:
        # Ensure database is an OrderedDict
        if not isinstance(database, OrderedDict):
            database = OrderedDict(database)

        # If the database is full, remove the oldest entry
        if len(database) >= max_size:
            database.popitem(last=False)  # Removes the first-added item in OrderedDict

        # Add the new entry
        new_key = str(int(max(database.keys(), key=int)) + 1) if database else "0"
        database[new_key] = response_json["answer1"]
        return database
    else:
        print("answer1 not found in response JSON.")
        return database


def get_overall_similarity(response_json):
    if "answer2" in response_json and "Overall Similarity" in response_json["answer2"]:
        try:
            return int(float(response_json["answer2"]["Overall Similarity"]))
        except ValueError:
            print("Failed to convert Overall Similarity to integer.")
            return None
    else:
        print("Overall Similarity not found in answer2.")
        return None


def modify_json_file(json_file, mutate_info):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Extract vehicle information from mutate_info
    vehicle_id = mutate_info["Vehicle ID"]
    new_location = mutate_info["Location"]
    new_speed = mutate_info["Speed"]

    # Recursively search and modify NPCs
    for key, value in data.items():
        if isinstance(value, dict) and "NPC" in value:
            for npc in value["NPC"]:
                if int(npc["id"]) == int(vehicle_id):
                    # Modify velocity
                    velocity = npc.get("velocity", {})
                    current_speed = (velocity.get("x", 0) ** 2 + velocity.get("y", 0) ** 2 + velocity.get("z",
                                                                                                          0) ** 2) ** 0.5
                    if current_speed != 0:
                        scale = new_speed / current_speed
                        npc["velocity"]["x"] *= scale
                        npc["velocity"]["y"] *= scale
                        npc["velocity"]["z"] *= scale
                    else:
                        # If current speed is zero, set velocity based on new speed equally divided among components
                        direction = [1, 0, 0]  # Default direction if current speed is zero
                        npc["velocity"]["x"] = new_speed * direction[0]
                        npc["velocity"]["y"] = new_speed * direction[1]
                        npc["velocity"]["z"] = new_speed * direction[2]

                    # Modify location (keeping rotation unchanged)
                    if "transform" in npc and "location" in npc["transform"]:
                        npc["transform"]["location"]["x"] = new_location[0]
                        npc["transform"]["location"]["y"] = new_location[1]
                    break
    return data


def get_answer3_vehicle_info(response_json):
    if "answer3" in response_json:
        try:
            mutate_section = response_json["answer3"]["Modified Background Vehicle for diversity"]
            raw_vehicle_id = str(mutate_section.get("Vehicle ID", "")).strip()
            raw_location = mutate_section.get("Location", "")
            raw_speed = mutate_section.get("Speed", "")

            if not raw_vehicle_id or raw_vehicle_id.lower() == "none":
                return None

            vehicle_id = int(raw_vehicle_id)

            try:
                location = ast.literal_eval(raw_location)
            except (ValueError, SyntaxError):
                return None

            speed_tokens = "".join(ch for ch in raw_speed if (ch.isdigit() or ch in ".-"))
            if not speed_tokens:
                return None
            speed_ms = float(speed_tokens) / 3.6  # convert km/h -> m/s

            return {
                "Vehicle ID": vehicle_id,
                "Location": location,
                "Speed": speed_ms,
            }
        except KeyError as e:
            print(f"Missing key in answer3: {e}")
            return None
        except ValueError as e:
            print(f"Invalid value in answer3: {e}")
            return None
    else:
        print("answer3 not found in response JSON.")
        return None


def get_all_json_files(base_dir):
    json_files = []
    for root, dirs, files in os.walk(base_dir):
        dirs.sort()
        files.sort()
        for filename in files:
            if filename.endswith('.json'):
                json_files.append(os.path.join(root, filename))
    return json_files


if __name__ == "__main__":
    Scenario_database = {}
    json_files = get_all_json_files('./data/save/time_record/')
    for json_file in json_files:
        print(f"Processing {json_file}...")
        try:
            # gpt
            scenario_description = str(get_frame_data(json_file)).replace("\n", "").replace(' ', '')
            question = prompt + "\n scenario snapshot:\n" + str(
                scenario_description + "\n___\n Scenario-dict\n" + str(Scenario_database))
            response = call_gpt(question, model_version=None, max_tokens=10000)  # 使用配置文件中的默认模型
            print("Response:", response)
            response_json = extract_json(response)
            Scenario_database = add_answer1_to_database(response_json, Scenario_database, 30)
            answer3_vehicle_info = get_answer3_vehicle_info(response_json)
            print("Answer3 Vehicle Info:", answer3_vehicle_info)
            mutate_info = answer3_vehicle_info
            data = modify_json_file(json_file, mutate_info)
            # Save modified JSON file
            new_json_file = json_file.replace('.json', '_modified.json').replace("./data_ot/save/time_record/",
                                                                                 "./data/new/time_record/")
            with open(new_json_file, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Modified JSON file saved as: {new_json_file}")
        # reload scenario state
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
