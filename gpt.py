import ast
import glob
import os
import random
import re
import sys

import requests
import json
import time
from collections import OrderedDict
from pathlib import Path

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
API_QUOTA_LIMIT = 100000  # Example daily quota limit in tokens

PROXY = {
    "http": "http://127.0.0.1:8080",
    "https": "http://127.0.0.1:8080"
}

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


def call_gpt4_vision(question: str, image_url: str, max_tokens: int = 100, retries: int = None) -> str:
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

    url = "https://api.openai.com/v1/chat/completions"
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
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
            return answer
        except requests.exceptions.RequestException as e:
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


def call_gpt(question: str, model_version: str = "gpt-4-turbo", max_tokens: int = 100, retries: int = None) -> str:
    """
    Call GPT model (3.5, 4, 4-turbo) with retry and quota check, supporting proxy settings.
    Uses infinite retries by default to ensure the experiment completes.

    Args:
        question (str): The question to ask the model.
        model_version (str): The model version to use ("gpt-3.5-turbo", "gpt-4", "gpt-4-turbo").
        max_tokens (int): Maximum token length for the response.
        retries (int): Number of retry attempts in case of request failure. None means infinite retries.

    Returns:
        str: The response from the specified GPT model or an error message.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_version,
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    attempt = 0
    max_wait_time = 60  # Maximum wait time between retries (seconds)
    
    while retries is None or attempt < retries:
        attempt += 1
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), proxies=PROXY)
            response.raise_for_status()
            response_data = response.json()

            answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No response")

            usage_info = response_data.get("usage", {})
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            total_tokens = usage_info.get("total_tokens", 0)
            print(f"Tokens used: Prompt = {prompt_tokens}, Completion = {completion_tokens}, Total = {total_tokens}")
            
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

            return answer
        except requests.exceptions.RequestException as e:
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


def call_gpt_with_rag(question: str, rag_engine, model_version: str = "gpt-4-turbo", 
                      max_tokens: int = 1500, retries: int = None) -> str:
    """
    Call GPT with RAG enhancement
    
    Args:
        question: Base question/prompt
        rag_engine: RAGEngine instance
        model_version: GPT model version to use
        max_tokens: Maximum tokens for response
        retries: Number of retry attempts
        
    Returns:
        GPT response string
    """
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
            response = call_gpt(question, model_version="gpt-4-turbo", max_tokens=1500)
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
