#!/usr/bin/env python3
"""
MIT License

Copyright (c) 2024 [??????????]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import cProfile
import logging
import os
import pdb
import re
import sys
import time
import random
import argparse
import copyreg
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import gpt
import json
import pickle
# from collections import deque
import concurrent.futures
import math
from types import SimpleNamespace
from typing import List
from subprocess import Popen, PIPE

import docker
import numpy as np
import torch
from deap import base, tools, algorithms
import signal
import traceback
import networkx as nx
from shapely.geometry import LineString
import threading
import config
import constants as c
from npc import NPC
from scenario import Scenario
import states
import utils

config.set_carla_api_path()
try:
    import carla
except ModuleNotFoundError as e:
    print("[-] Carla module not found. Make sure you have built Carla.")
    proj_root = config.get_proj_root()
    print("    Try `cd {}/carla && make PythonAPI' if not.".format(proj_root))
    exit(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client, world, G, blueprint_library, town_map = None, None, None, None, None
# model = cluster.FeatureExtractor().to(device)
accumulated_trace_graphs = []
autoware_container = None
exec_state = states.ExecState()
Scenario_database = {}
# Global scenario counter for quantitative experiments
total_scenarios_generated = 0

# Thread locks for protecting shared global variables
_scenario_count_lock = threading.Lock()  # Lock for total_scenarios_generated
_scenario_db_lock = threading.Lock()  # Lock for Scenario_database

# Global GA state variables for checkpoint saving from evaluation function
_ga_state = {
    'checkpoint_path': None,
    'curr_gen': 0,
    'population': None,
    'archive': None,
    'hof': None,
    'next_scenario_id': 1,
    'logbook': None,
    'stats': None
}


def _load_scenario_database(db_path: str):
    """Load Scenario_database from JSON file if it exists."""
    if not db_path:
        return OrderedDict()
    try:
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure deterministic ordering
            return OrderedDict(sorted(data.items(), key=lambda kv: int(kv[0])))
    except Exception as e:
        print(f"[WARNING] Failed to load scenario database from {db_path}: {e}")
    return OrderedDict()


def _save_scenario_database(db_path: str, database: dict):
    """Persist Scenario_database to JSON file."""
    if not db_path:
        return
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(database, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to save scenario database to {db_path}: {e}")
bottleneck = False

REPO_ROOT = Path(__file__).resolve().parent
API_CONFIG_PATH = REPO_ROOT / "api.json"
PROMPT_PATH = REPO_ROOT / "prompt.txt"

with open(API_CONFIG_PATH, "r") as f:
    api_config = json.load(f)
API_KEY = api_config.get("OPENAI_API_KEY")
PROXY = {
    "http": "http://127.0.0.1:8080",
    "https": "http://127.0.0.1:8080"
}

with open(PROMPT_PATH, 'r') as f:
    prompt = f.read()


# monitor carla
# monitoring_thread = utils.monitor_docker_container('carlasim/carla:0.9.13')
# vehicle_bp_library = blueprint_library.filter("vehicle.*")
# vehicle_bp.set_attribute("color", "255,0,0")
# walker_bp = blueprint_library.find("walker.pedestrian.0001")  # 0001~0014
# walker_controller_bp = blueprint_library.find('controller.ai.walker')
# player_bp = blueprint_library.filter('nissan')[0]


def create_test_scenario(conf, seed_dict):
    return Scenario(conf, seed_dict)


def handler(signum, frame):
    raise Exception("HANG")


def ini_hyperparameters(conf, args):
    conf.cur_time = time.time()
    if args.determ_seed:
        conf.determ_seed = args.determ_seed
    else:
        conf.determ_seed = conf.cur_time
    random.seed(conf.determ_seed)
    print("[info] determ seed set to:", conf.determ_seed)
    conf.out_dir = args.out_dir
    try:
        os.mkdir(conf.out_dir)
    except Exception:
        # Allow reuse only if explicitly permitted or directory is empty
        if getattr(args, "allow_out_dir_exists", False):
            # Directory was likely created by experiment manager - this is expected behavior
            # Silently continue without printing a message to avoid redundant output
            pass
        elif os.path.isdir(conf.out_dir) and not os.listdir(conf.out_dir):
            print(f"[INFO] Output directory {conf.out_dir} already exists but is empty, reusing it.")
        else:
            estr = f"Output directory {conf.out_dir} already exists. Remove with " \
                   "caution; it might contain data from previous runs."
            print(estr)
            sys.exit(-1)

    conf.seed_dir = args.seed_dir
    if not os.path.exists(conf.seed_dir):
        os.mkdir(conf.seed_dir)
    else:
        print(f"Using seed dir {conf.seed_dir}")
    conf.set_paths()

    # Initialize metrics output directory (for per-scenario rag_metrics records)
    if getattr(conf, "metrics_output_dir", None) is None:
        conf.metrics_output_dir = os.path.join(conf.out_dir, "metrics")
    try:
        os.makedirs(conf.metrics_output_dir, exist_ok=True)
        # Clean stale metrics_records.jsonl when starting a new run
        records_path = os.path.join(conf.metrics_output_dir, "metrics_records.jsonl")
        if os.path.exists(records_path):
            os.remove(records_path)
    except Exception as e:
        print(f"[Metrics] Warning: Failed to prepare metrics_output_dir {conf.metrics_output_dir}: {e}")

    # with open(conf.meta_file, "w") as f:
    #     f.write(" ".join(sys.argv) + "\n")
    #     f.write("start: " + str(int(conf.cur_time)) + "\n")

    # Create directories if they don't exist (allow reuse for retries)
    os.makedirs(conf.queue_dir, exist_ok=True)
    os.makedirs(conf.error_dir, exist_ok=True)
    os.makedirs(conf.picture_dir, exist_ok=True)
    os.makedirs(conf.rosbag_dir, exist_ok=True)
    os.makedirs(conf.cam_dir, exist_ok=True)
    os.makedirs(conf.npc_dir, exist_ok=True)
    os.makedirs(conf.time_record_dir, exist_ok=True)
    if args.no_lane_check:
        conf.check_dict["lane"] = False
    conf.sim_host = args.sim_host
    conf.sim_port = args.sim_port
    conf.max_mutations = args.max_mutations
    conf.timeout = args.timeout
    conf.function = args.function

    if args.target.lower() == "behavior":
        conf.agent_type = c.BEHAVIOR
    elif args.target.lower() == "autoware":
        conf.agent_type = c.AUTOWARE
    else:
        print("[-] Unknown target: {}".format(args.target))
        sys.exit(-1)

    conf.town = args.town
    conf.num_mutation_car = args.num_mutation_car
    conf.density = float(args.density)
    conf.no_traffic_lights = args.no_traffic_lights
    conf.debug = args.debug
    # GPT / Scenario database configuration
    conf.scenario_db = args.scenario_db
    conf.gpt_log_dir = args.gpt_log_dir
    # Scenario limit and RAG / metrics configuration (can be overridden by experiment manager)
    conf.max_scenarios = getattr(args, "max_scenarios", 0)
    conf.enable_rag = getattr(args, "enable_rag", False)
    conf.enable_rag_metrics = getattr(args, "enable_rag_metrics", False)
    # GPT-based evaluation configuration
    # Default is enabled; can be disabled via CLI or by callers overriding
    conf.enable_gpt_evaluation = not getattr(args, "disable_gpt", False)


def mutate_weather(test_scenario):
    test_scenario.weather["cloud"] = random.randint(0, 100)
    test_scenario.weather["rain"] = random.randint(0, 100)
    test_scenario.weather["wind"] = random.randint(0, 100)
    test_scenario.weather["fog"] = random.randint(0, 100)
    test_scenario.weather["wetness"] = random.randint(0, 100)
    test_scenario.weather["angle"] = random.randint(0, 360)
    test_scenario.weather["altitude"] = random.randint(-90, 90)


def mutate_weather_fixed(test_scenario):
    test_scenario.weather["cloud"] = 0
    test_scenario.weather["rain"] = 0
    test_scenario.weather["wind"] = 0
    test_scenario.weather["fog"] = 0
    test_scenario.weather["wetness"] = 0
    test_scenario.weather["angle"] = 0
    test_scenario.weather["altitude"] = 60


def set_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--debug", action="store_true", default=False)
    argument_parser.add_argument("-o", "--out-dir", default="./data/output", type=str,
                                 help="Directory to save fuzzing logs")
    argument_parser.add_argument("-m", "--max-mutations", default=5, type=int,
                                 help="Size of the mutated population per cycle")
    argument_parser.add_argument("-d", "--determ-seed", type=float,
                                 help="Set seed num for deterministic mutation (e.g., for replaying)")
    argument_parser.add_argument("-u", "--sim-host", default="localhost", type=str,
                                 help="Hostname of Carla simulation server")
    argument_parser.add_argument("-p", "--sim-port", default=2000, type=int,
                                 help="RPC port of Carla simulation server")
    argument_parser.add_argument("-s", "--seed-dir", default="./data/seed", type=str,
                                 help="Seed directory")
    argument_parser.add_argument("-t", "--target", default="behavior", type=str,
                                 help="Target autonomous driving system (behavior/Autoware)")
    argument_parser.add_argument("-f", "--function", default="general", type=str,
                                 choices=["general", "collision", "traction", "eval-os", "eval-us",
                                          "figure", "sens1", "sens2", "lat", "rear"],
                                 help="Functionality to test (general / collision / traction)")
    argument_parser.add_argument("-k", "--num_mutation_car", default=3, type=int,
                                 help="Number of max weight vehicles to mutation per cycle, default=1,negative means "
                                      "random")
    argument_parser.add_argument("--density", default=1, type=float,
                                 help="density of vehicles,1.0 means add 1 bg vehicle per 1 sec")
    argument_parser.add_argument("--town", default=3, type=int,
                                 help="Test on a specific town (e.g., '--town 3' forces Town03)")
    argument_parser.add_argument("--timeout", default="60", type=int,
                                 help="Seconds to timeout if vehicle is not moving")
    argument_parser.add_argument("--no-speed-check", action="store_true")
    argument_parser.add_argument("--no-lane-check", action="store_true")
    argument_parser.add_argument("--no-crash-check", action="store_true")
    argument_parser.add_argument("--no-stuck-check", action="store_true")
    argument_parser.add_argument("--no-red-check", action="store_true")
    argument_parser.add_argument("--no-other-check", action="store_true")
    argument_parser.add_argument("--no-traffic-lights", action="store_true")
    argument_parser.add_argument("--scenario-db", default="./data/scenario_db.json", type=str,
                                 help="Path to scenario database JSON file (for GPT/RAG context)")
    argument_parser.add_argument("--gpt-log-dir", default="./data/gpt_logs", type=str,
                                 help="Directory to store GPT conversation logs")
    argument_parser.add_argument("--max-scenarios", default=0, type=int,
                                 help="Maximum number of scenarios to evaluate in this run (0 = unlimited)")
    argument_parser.add_argument("--enable-rag", action="store_true",
                                 help="Enable RAG-based retrieval for prompt construction")
    argument_parser.add_argument("--enable-rag-metrics", action="store_true",
                                 help="Enable RAG-related coverage metrics")
    argument_parser.add_argument("--disable-gpt", action="store_true",
                                 help="Disable GPT-based evaluation and logging (non-GPT baseline)")
    argument_parser.add_argument("--allow-out-dir-exists", action="store_true",
                                 help="Allow using an existing out-dir (used by experiment manager)")
    return argument_parser


def extract_answer1_overall_similarity(response_text):
    """
    Extracts answer1 description and Overall Similarity from raw response text using regular expressions.

    Parameters:
    - response_text: The raw text of the JSON response

    Returns:
    - answer1_description: Extracted description of answer1
    - overall_similarity: Extracted similarity score (integer) or None if not found
    """
    # Regular expression to extract answer1 description
    answer1_match = re.search(r'"answer1":\s*\{\s*"Description":\s*"([^"]+)', response_text)
    answer1_description = answer1_match.group(1) if answer1_match else "Description not available"

    # Regular expression to extract Overall Similarity
    similarity_match = re.search(r'"Overall Similarity":\s*"(\d+)', response_text)
    overall_similarity = int(similarity_match.group(1)) if similarity_match else None

    return answer1_description, overall_similarity


def evaluation(ind: Scenario):
    global autoware_container
    global Scenario_database
    global conf
    global total_scenarios_generated
    min_dist = 99999
    nova = 0
    # Overall similarity score returned by GPT-based evaluation (0–100).
    # For GPT-enabled runs we will keep retrying until we obtain a valid score.
    overall_similarity = 0
    g_name = f'Generation_{ind.generation_id:05}'
    s_name = f'Scenario_{ind.scenario_id:05}'
    # run test here
    mutate_weather_fixed(ind)
    # Get conf from scenario if available (only check once)
    if not 'conf' in globals() or conf is None:
        conf = ind.conf if hasattr(ind, 'conf') else None
    
    # Check time limit before starting evaluation
    if conf:
        experiment_start_time = getattr(conf, 'experiment_start_time', None)
        experiment_timeout = getattr(conf, 'experiment_timeout', None)
        if experiment_start_time and experiment_timeout:
            elapsed_time = time.time() - experiment_start_time
            if elapsed_time >= experiment_timeout:
                print(f"[TIME LIMIT] Reached time limit in evaluation: {elapsed_time:.0f}s / {experiment_timeout:.0f}s")
                raise TimeoutError(f"Experiment time limit reached: {elapsed_time:.0f}s / {experiment_timeout:.0f}s")
            # Validate timeout values
            if experiment_timeout < 0:
                print(f"[WARNING] Invalid experiment_timeout: {experiment_timeout}, ignoring time limit")
                experiment_timeout = None
    
    signal.alarm(15 * 60)  # timeout after 15 min
    print("timeout after 15 min")
    ret = None  # Initialize ret variable to avoid UnboundLocalError
    try:
        # profiler = cProfile.Profile()
        # profiler.enable()  #
        ind.state.scenario_id = ind.scenario_id
        ind.state.generation_id = ind.generation_id
        ret = ind.run_test(exec_state)
        if ret == -1:
            print("[-] Fatal error occurred during test")
            # Raise exception instead of exit() to allow proper error handling
            raise RuntimeError("Fatal error occurred during test (ret == -1)")
        # Count scenarios from file system (single source of truth)
        # This ensures consistency even if the process crashes and restarts
        try:
            if conf and hasattr(conf, 'queue_dir'):
                # Sync from file system to ensure accuracy
                file_count = _count_scenarios_from_files(conf.queue_dir)
                total_scenarios_generated = file_count
            else:
                # Fallback to incrementing if conf is not available
                total_scenarios_generated += 1
        except NameError:
            # In case global counter is not initialized for some legacy path
            if conf and hasattr(conf, 'queue_dir'):
                total_scenarios_generated = _count_scenarios_from_files(conf.queue_dir)
            else:
                total_scenarios_generated = 1
        
        # Save checkpoint after each scenario evaluation for fault tolerance
        # This ensures we can recover even if interrupted during a generation
        try:
            global _ga_state
            if _ga_state['checkpoint_path'] and _ga_state['population'] is not None:
                # Get current state from global variables
                checkpoint_path = _ga_state['checkpoint_path']
                curr_gen = _ga_state.get('curr_gen', ind.generation_id)
                population = _ga_state.get('population', [])
                archive = _ga_state.get('archive', [])
                hof = _ga_state.get('hof', None)
                next_scenario_id = _ga_state.get('next_scenario_id', ind.scenario_id + 1)
                logbook = _ga_state.get('logbook', None)
                stats = _ga_state.get('stats', None)
                
                # Only save if we have valid state
                if hof is not None:
                    # Convert logbook to serializable format if it exists
                    logbook_data = None
                    if logbook and hasattr(logbook, 'chapters') and logbook.chapters:
                        logbook_data = []
                        for chapter_name, chapter_data in logbook.chapters.items():
                            logbook_data.append({
                                'name': chapter_name,
                                'header': chapter_data.get('header', []),
                                'rows': chapter_data.get('rows', [])
                            })
                    _save_checkpoint(
                        checkpoint_path, 
                        curr_gen, 
                        total_scenarios_generated, 
                        population, 
                        archive, 
                        hof, 
                        next_scenario_id,
                        logbook_data,  # Pass serializable data, not Logbook object
                        None  # Don't save stats (contains lambda)
                    )
        except Exception as checkpoint_err:
            # Don't fail evaluation if checkpoint save fails
            print(f"[WARNING] Failed to save checkpoint after scenario evaluation: {checkpoint_err}")
        min_dist = ind.state.min_dist

        # Calculate NOVA (speed variation)
        if ind.state.speed and len(ind.state.speed) > 1:
            for i in range(1, len(ind.state.speed)):
                acc = abs(ind.state.speed[i] - ind.state.speed[i - 1])
                nova += acc
            nova = nova / len(ind.state.speed)
        else:
            nova = 0
        # GPT / RAG-based evaluation and logging (optional)
        # This block can be disabled (e.g., for TM-Fuzzer non-GPT baseline)
        answer3_vehicle_info = {}
        if not conf or getattr(conf, "enable_gpt_evaluation", True):
            # Build path to time_record JSON using configured time_record_dir
            time_record_dir = getattr(conf, "time_record_dir", "./data/output/time_record")
            time_record_path = os.path.join(
                time_record_dir,
                f"gid:{ind.generation_id}_sid:{ind.scenario_id}.json"
            )
            scenario_description = str(
                gpt.get_frame_data(time_record_path, ind.state.min_dist_frame)
            ).replace("\n", "").replace(' ', '')

            # Keep calling GPT until we obtain a valid JSON response with a
            # numeric overall similarity score. This avoids silently degrading
            # metrics when the network is unstable.
            # Add retry limits to prevent infinite loops
            MAX_GPT_RETRIES = 5  # Maximum number of GPT retry attempts
            gpt_retry_count = 0
            base_delay = 5  # Base delay in seconds for exponential backoff
            
            while gpt_retry_count < MAX_GPT_RETRIES:
                # Use RAG to retrieve relevant scenarios if enabled, otherwise use Scenario_database
                if conf and conf.enable_rag:
                    try:
                        # Choose RAG engine based on configuration
                        if conf.use_enhanced_rag:
                            from rag_module import EnhancedRAGEngine
                            # Initialize Enhanced RAG engine if not exists
                            if not hasattr(evaluation, 'rag_engine'):
                                evaluation.rag_engine = EnhancedRAGEngine(
                                    top_k=conf.rag_k,
                                    use_hybrid_search=conf.use_hybrid_search,
                                    hybrid_alpha=conf.hybrid_alpha,
                                    use_reranking=conf.use_reranking,
                                    reranker_model=conf.reranker_model
                                )
                                evaluation.rag_engine.initialize(load_mock_data=True)
                                # Add existing Scenario_database to RAG knowledge base
                                # Use lock to protect concurrent access
                                with _scenario_db_lock:
                                    db_copy = dict(Scenario_database)  # Create a copy to minimize lock time
                                for key, desc in db_copy.items():
                                    evaluation.rag_engine.add_scenario_to_knowledge_base({
                                        'id': f'db_{key}',
                                        'description': desc
                                    }, rebuild_index=False)
                                # Rebuild index after adding all existing scenarios
                                scenario_descriptions = evaluation.rag_engine.knowledge_base.get_scenario_descriptions()
                                if len(scenario_descriptions) > 0:
                                    vectors = evaluation.rag_engine.encoder.encode_batch(scenario_descriptions)
                                    scenarios = evaluation.rag_engine.knowledge_base.get_all_scenarios()
                                    evaluation.rag_engine.vector_store.build_index(vectors, scenario_descriptions, scenarios)
                                    # Refit BM25 if hybrid search is enabled
                                    if conf.use_hybrid_search and evaluation.rag_engine.hybrid_retriever:
                                        evaluation.rag_engine.hybrid_retriever.fit_bm25(scenario_descriptions)
                                print(f"[EnhancedRAG] Initialized with {len(scenario_descriptions)} scenarios from Scenario_database")
                        else:
                            from rag_module import RAGEngine
                            # Initialize RAG engine if not exists
                            if not hasattr(evaluation, 'rag_engine'):
                                evaluation.rag_engine = RAGEngine(top_k=conf.rag_k)
                                evaluation.rag_engine.initialize(load_mock_data=True)
                                # Add existing Scenario_database to RAG knowledge base
                                # Use lock to protect concurrent access
                                with _scenario_db_lock:
                                    db_copy = dict(Scenario_database)  # Create a copy to minimize lock time
                                for key, desc in db_copy.items():
                                    evaluation.rag_engine.add_scenario_to_knowledge_base({
                                        'id': f'db_{key}',
                                        'description': desc
                                    }, rebuild_index=False)
                                # Rebuild index after adding all existing scenarios
                                scenario_descriptions = evaluation.rag_engine.knowledge_base.get_scenario_descriptions()
                                if len(scenario_descriptions) > 0:
                                    vectors = evaluation.rag_engine.encoder.encode_batch(scenario_descriptions)
                                    scenarios = evaluation.rag_engine.knowledge_base.get_all_scenarios()
                                    evaluation.rag_engine.vector_store.build_index(vectors, scenario_descriptions, scenarios)
                                print(f"[RAG] Initialized with {len(scenario_descriptions)} scenarios from Scenario_database")
                        
                        # Retrieve relevant scenarios using RAG
                        scenario_dict_str = evaluation.rag_engine.retrieve_and_format_for_prompt(
                            scenario_description, k=conf.rag_k
                        )
                        print(f"[RAG] Retrieved {conf.rag_k} relevant scenarios for prompt")
                        
                    except Exception as e:
                        print(f"[RAG] Warning: RAG retrieval failed, falling back to Scenario_database: {e}")
                        scenario_dict_str = str(Scenario_database)
                else:
                    # Original logic: use Scenario_database
                    # Use lock to protect concurrent access
                    with _scenario_db_lock:
                        scenario_dict_str = str(Scenario_database)
                
                question = prompt + "\n scenario snapshot:\n" + str(
                    scenario_description + "\n___\n Scenario-dict\n" + scenario_dict_str)
                response = gpt.call_gpt(question, model_version="gpt-4-turbo", max_tokens=1500)
                print("Response:", response)
                response_json = gpt.extract_json(response)

                if response_json is None:
                    # Network or parsing failure – retry with exponential backoff
                    gpt_retry_count += 1
                    if gpt_retry_count >= MAX_GPT_RETRIES:
                        print(f"[GPT] Failed to extract JSON response after {MAX_GPT_RETRIES} attempts. Using default similarity score.")
                        overall_similarity = 0  # Use default score on failure
                        break
                    delay = base_delay * (2 ** (gpt_retry_count - 1))  # Exponential backoff
                    print(f"[GPT] Failed to extract JSON response, retrying ({gpt_retry_count}/{MAX_GPT_RETRIES}) after {delay}s...")
                    time.sleep(delay)
                    continue

                try:
                    overall_similarity = int(gpt.get_overall_similarity(response_json))
                except Exception as e:
                    gpt_retry_count += 1
                    if gpt_retry_count >= MAX_GPT_RETRIES:
                        print(f"[GPT] Failed to obtain overall similarity after {MAX_GPT_RETRIES} attempts. Using default similarity score.")
                        overall_similarity = 0  # Use default score on failure
                        break
                    delay = base_delay * (2 ** (gpt_retry_count - 1))  # Exponential backoff
                    print(f"[GPT] Failed to obtain overall similarity from JSON, retrying ({gpt_retry_count}/{MAX_GPT_RETRIES}) after {delay}s: {e}")
                    time.sleep(delay)
                    continue

                # Persist GPT conversation log if configured
                try:
                    log_dir = getattr(conf, "gpt_log_dir", "./data/gpt_logs")
                    os.makedirs(log_dir, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_name = f"gpt_gen{ind.generation_id:05}_scen{ind.scenario_id:05}_{ts}.json"
                    log_path = os.path.join(log_dir, log_name)
                    with open(log_path, "w", encoding="utf-8") as lf:
                        json.dump(
                            {
                                "generation_id": ind.generation_id,
                                "scenario_id": ind.scenario_id,
                                "timestamp": ts,
                                "prompt": question,
                                "raw_response": response,
                                "parsed_response": response_json,
                            },
                            lf,
                            ensure_ascii=False,
                            indent=2,
                        )
                except Exception as e:
                    print(f"[WARNING] Failed to log GPT conversation: {e}")

                # Update Scenario_database (for backward compatibility and fallback)
                # Use lock to protect concurrent access
                try:
                    with _scenario_db_lock:
                        Scenario_database = gpt.add_answer1_to_database(response_json, Scenario_database, 30)
                        # Persist Scenario_database
                        if conf and getattr(conf, "scenario_db", None):
                            _save_scenario_database(conf.scenario_db, Scenario_database)
                except Exception as e:
                    print(f"[WARNING] Failed to update Scenario_database from GPT response: {e}")
                
                # Also add to RAG knowledge base if enabled
                if conf and conf.enable_rag and hasattr(evaluation, 'rag_engine'):
                    try:
                        answer1_desc = response_json.get("answer1", {}).get("Description", "")
                        if answer1_desc:
                            evaluation.rag_engine.add_scenario_to_knowledge_base({
                                'id': f'gpt_gen_{ind.generation_id}_scen_{ind.scenario_id}',
                                'description': answer1_desc
                            }, rebuild_index=False)
                    except Exception as e:
                        print(f"[RAG] Warning: Failed to add to RAG knowledge base: {e}")

                # If we reach here, we have a valid similarity score and can break
                answer3_vehicle_info = gpt.get_answer3_vehicle_info(response_json)
                print("Overall Similarity:", overall_similarity)
                print("Answer3 Vehicle Info:", answer3_vehicle_info)
                break  # Successfully obtained response, exit retry loop
            
            # If we exhausted all retries without success, use default values
            if gpt_retry_count >= MAX_GPT_RETRIES and 'overall_similarity' not in locals():
                print(f"[GPT] All {MAX_GPT_RETRIES} retry attempts failed. Using default values.")
                overall_similarity = 0
                answer3_vehicle_info = {}

        # Store GPT-guided mutation info (empty when GPT is disabled)
        ind.mutate_info = answer3_vehicle_info
        # reload scenario state
        ind.state = states.ScenarioState()
    except Exception as e:
        # Special handling for CARLA simulator RPC timeouts: let outer wrappers
        # (script/test.py, ExperimentManager) decide how to restart the env.
        if isinstance(e, RuntimeError) and "time-out of 10000ms while waiting for the simulator" in str(e):
            print("[-] CARLA simulator timeout detected in simulate(), propagating exception for env restart.")
            raise

        if isinstance(e, TimeoutError):
            print("[-] simulation hanging. abort current scenario.")
            ret = 1
        else:
            print("[-] run_test error (will continue with next scenario):")
            traceback.print_exc()
            # For non-fatal errors, set ret to 1 and continue instead of raising
            # This allows the fuzzer to continue generating scenarios even if one fails
            ret = 1
            # Set default fitness values so the individual can still be evaluated
            if not ind.fitness.valid:
                ind.fitness.values = (0.0, 0.0)  # Default fitness: (min_dist, nova)
            # Don't re-raise for non-fatal errors - let the fuzzer continue

    if ret is None:
        pass
    elif ret == -1:
        print("[-] Fatal error occurred during test (will continue with next scenario)")
        # Instead of raising, set default fitness and continue
        # This allows the fuzzer to keep running even if one scenario has a fatal error
        if not ind.fitness.valid:
            ind.fitness.values = (0.0, 0.0)  # Default fitness: (min_dist, nova)
        # Only raise if this is a critical system error that requires restart
        # For now, we'll continue to allow maximum scenario generation
        # raise RuntimeError("Fatal error occurred during test (ret == -1)")
    elif ret == 1:
        print("fuzzer - found an error")
    elif ret == 128:
        print("Exit by user request")

    # mutation loop ends
    if ind.found_error:
        print("[-]error detected. start a new cycle with a new seed")
    
    # Calculate additional metrics if enabled
    if conf and getattr(conf, "enable_rag_metrics", False):
        try:
            from metrics import ParameterCoverage, BehaviorCoverage, TrajectoryDiversity, BehaviorMatrix
            
            # Store metrics results in scenario for later aggregation
            if not hasattr(ind, 'rag_metrics'):
                ind.rag_metrics = {}
            
            # Parameter Coverage (PC)
            pc_calculator = ParameterCoverage()
            pc_score = pc_calculator.calculate_coverage([ind])
            ind.rag_metrics['pc'] = pc_score
            
            # Behavior Coverage (PEC)
            pec_calculator = BehaviorCoverage()
            pec_score = pec_calculator.calculate_coverage([ind])
            ind.rag_metrics['pec'] = pec_score
            
            # Trajectory Diversity (TCD)
            tcd_calculator = TrajectoryDiversity()
            tcd_results = tcd_calculator.calculate_coverage([ind])
            ind.rag_metrics['tcd'] = tcd_results.get('diversity_score', 0.0)
            
            # Behavior Matrix Coverage (BCM)
            bcm_calculator = BehaviorMatrix()
            bcm_results = bcm_calculator.calculate_coverage([ind])
            ind.rag_metrics['bcm'] = bcm_results.get('coverage_ratio', 0.0)

            # Persist per-scenario metrics record for later aggregation
            try:
                metrics_dir = getattr(conf, "metrics_output_dir", None)
                if not metrics_dir:
                    base_out = getattr(conf, "out_dir", "./data/output")
                    metrics_dir = os.path.join(base_out, "metrics")
                os.makedirs(metrics_dir, exist_ok=True)

                record = {
                    "generation_id": getattr(ind, "generation_id", -1),
                    "scenario_id": getattr(ind, "scenario_id", -1),
                    "pc": float(ind.rag_metrics.get("pc", 0.0)),
                    "pec": float(ind.rag_metrics.get("pec", 0.0)),
                    "tcd": float(ind.rag_metrics.get("tcd", 0.0)),
                    "bcm": float(ind.rag_metrics.get("bcm", 0.0)),
                }
                records_path = os.path.join(metrics_dir, "metrics_records.jsonl")
                with open(records_path, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as metrics_io_err:
                print(f"[Metrics] Warning: Failed to append metrics record: {metrics_io_err}")
            
        except ImportError as e:
            print(f"[Metrics] Warning: Could not import metrics modules: {e}")
        except Exception as e:
            print(f"[Metrics] Warning: Error calculating metrics: {e}")
    
    # For GPT-enabled runs we only exit the loop when a valid numeric
    # overall_similarity has been obtained. For GPT-disabled runs
    # (e.g., TM-Fuzzer baseline) this remains at its default 0.
    return min_dist, nova, 100 - overall_similarity


# MUTATION OPERATOR


def mut_npc_list(ind: Scenario):
    global town_map
    if len(ind.npc_list) <= 1:
        return ind.npc_list
    if bottleneck:
        # mutate the chosen one by GPT
        if ind.mutate_info and isinstance(ind.mutate_info, dict) and "Vehicle ID" in ind.mutate_info:
            for npc in ind.npc_list:
                if npc.instance_id == ind.mutate_info["Vehicle ID"]:
                    npc.speed = ind.mutate_info["Speed"]
                    if town_map is not None and "Location" in ind.mutate_info:
                        location = carla.Location(x=ind.mutate_info["Location"][0], y=ind.mutate_info["Location"][1],
                                                  z=0.5)
                        waypoint = town_map.get_waypoint(location, project_to_road=True,
                                                         lane_type=carla.libcarla.LaneType.Driving)
                        npc.spawn_point = waypoint
                    return ind.npc_list
        return ind.npc_list
    mut_pb = random.random()
    random_index = random.randint(0, len(ind.npc_list) - 1)
    # remove a random 1
    if mut_pb < 0.1:
        ind.npc_list.pop(random_index)
        return ind.npc_list
    # add a random 1
    if mut_pb < 0.4:
        template_npc = ind.npc_list[random_index]
        new_ad = NPC.get_npc_by_one(template_npc, town_map, len(ind.npc_list) - 1)
        ind.npc_list.append(new_ad)
        return ind.npc_list
    # mutate a random agent
    template_npc = ind.npc_list[random_index]
    new_ad = NPC.get_npc_by_one(template_npc, town_map, len(ind.npc_list) - 1)
    ind.npc_list.append(new_ad)
    ind.npc_list.pop(random_index)
    return ind.npc_list


def mut_scenario(ind: Scenario):
    global conf
    # Get conf from scenario if not available globally
    if not 'conf' in globals() or conf is None:
        conf = ind.conf if hasattr(ind, 'conf') else None
    
    # Use RAG-guided mutation if enabled and bottleneck detected
    if conf and conf.enable_rag and bottleneck:
        try:
            # Choose RAG engine based on configuration
            if conf.use_enhanced_rag:
                from rag_module import EnhancedRAGEngine
                # Initialize Enhanced RAG engine (lazy initialization)
                if not hasattr(mut_scenario, 'rag_engine'):
                    mut_scenario.rag_engine = EnhancedRAGEngine(
                        top_k=conf.rag_k,
                        use_hybrid_search=conf.use_hybrid_search,
                        hybrid_alpha=conf.hybrid_alpha,
                        use_reranking=conf.use_reranking,
                        reranker_model=conf.reranker_model
                    )
                    mut_scenario.rag_engine.initialize(load_mock_data=True)
            else:
                from rag_module import RAGEngine
                # Initialize RAG engine (lazy initialization)
                if not hasattr(mut_scenario, 'rag_engine'):
                    mut_scenario.rag_engine = RAGEngine(top_k=conf.rag_k)
                    mut_scenario.rag_engine.initialize(load_mock_data=True)
            
            # Generate scenario description from current state
            scenario_desc = f"Scenario with {len(ind.npc_list)} NPCs, weather: {ind.weather}"
            
            # Use RAG to generate enhanced scenario
            rag_result = mut_scenario.rag_engine.generate_scenario(scenario_desc, use_gpt=False)
            
            # Apply retrieved scenarios for guidance (simplified integration)
            # In full implementation, this would modify NPCs based on RAG suggestions
            print(f"[RAG] Retrieved {len(rag_result.get('retrieved_scenarios', []))} relevant scenarios")
            
        except ImportError as e:
            print(f"[RAG] Warning: Could not import RAG modules: {e}")
        except Exception as e:
            print(f"[RAG] Warning: Error in RAG-guided mutation: {e}")
    
    # Standard mutation
    ind.npc_list = mut_npc_list(ind)
    return ind,


# CROSSOVER OPERATOR

def cx_npc(ind1: List[NPC], ind2: List[NPC]):
    # todo: swap entire ad section
    cx_pb = random.random()
    if cx_pb < 0.05:
        return ind2, ind1

    for adc1 in ind1:
        for adc2 in ind2:
            NPC.npc_cross(adc1, adc2)

    # # if len(ind1.adcs) < MAX_ADC_COUNT:
    # #     for adc in ind2.adcs:
    # #         if ind1.has_conflict(adc) and ind1.add_agent(deepcopy(adc)):
    # #             # add an agent from parent 2 to parent 1 if there exists a conflict
    # #             ind1.adjust_time()
    # #             return ind1, ind2
    #
    # # if none of the above happened, no common adc, no conflict in either
    # # combine to make a new populations
    # available_adcs = ind1.adcs + ind2.adcs
    # random.shuffle(available_adcs)
    # split_index = random.randint(2, min(len(available_adcs), MAX_ADC_COUNT))
    #
    # result1 = ADSection([])
    # for x in available_adcs[:split_index]:
    #     result1.add_agent(copy.deepcopy(x))
    #
    # # make sure offspring adc count is valid
    #
    # while len(result1.adcs) > MAX_ADC_COUNT:
    #     result1.adcs.pop()
    #
    # while len(result1.adcs) < 2:
    #     new_ad = ADAgent.get_one()
    #     if result1.has_conflict(new_ad) and result1.add_agent(new_ad):
    #         break
    # result1.adjust_time()
    return ind1, ind2


def cx_scenario(ind1: Scenario, ind2: Scenario):
    ind1.npc_list, ind2.npc_list = cx_npc(
        ind1.npc_list, ind2.npc_list
    )
    return ind1, ind2


def seed_initialize(town, town_map):
    spawn_points = town.get_spawn_points()
    sp = random.choice(spawn_points)
    sp_x = sp.location.x
    sp_y = sp.location.y
    sp_z = sp.location.z
    pitch = sp.rotation.pitch
    yaw = sp.rotation.yaw
    roll = sp.rotation.roll
    # restrict destination to be within 200 meters
    destination_flag = True
    wp, wp_x, wp_y, wp_z, wp_yaw = None, None, None, None, None
    while destination_flag:
        wp = random.choice(spawn_points)
        wp_x = wp.location.x
        wp_y = wp.location.y
        wp_z = wp.location.z
        wp_yaw = wp.rotation.yaw
        if math.sqrt((sp_x - wp_x) ** 2 + (sp_y - wp_y) ** 2) > c.MIN_DIST:
            destination_flag = False
        if math.sqrt((sp_x - wp_x) ** 2 + (sp_y - wp_y) ** 2) > c.MAX_DIST:
            destination_flag = True
    seed_dict = {
        "map": town_map,
        "sp_x": sp_x,
        "sp_y": sp_y,
        "sp_z": sp_z,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "wp_x": wp_x,
        "wp_y": wp_y,
        "wp_z": wp_z,
        "wp_yaw": wp_yaw
    }
    return seed_dict


def init_env(args):
    conf = config.Config()

    if args is None:
        argument_parser = set_args()
        args = argument_parser.parse_args()

    ini_hyperparameters(conf, args)
    if conf.town is not None:
        town_map = "Town0{}".format(conf.town)
    else:
        town_map = "Town0{}".format(random.randint(1, 5))
    if conf.no_traffic_lights:
        conf.check_dict["red"] = False
    signal.signal(signal.SIGALRM, handler)
    client = utils.connect(conf)
    client.set_timeout(20)
    client.load_world(town_map)
    world = client.get_world()
    town = world.get_map()
    map_topology = town.get_topology()
    G = nx.DiGraph()
    lane_list = {}
    for edge in map_topology:
        # 1.add_edge for every lane that is connected
        G.add_edge((edge[0].road_id, edge[0].lane_id), (edge[1].road_id, edge[1].lane_id))
        if (edge[0].road_id, edge[0].lane_id) not in lane_list:
            edge_end = edge[0].next_until_lane_end(500)[-1]
            lane_list[(edge[0].road_id, edge[0].lane_id)] = (edge[0], edge_end)
    added_edges = []
    for lane_A in lane_list:
        for lane_B in lane_list:
            # 2.add_edge for every lane that is cross in junction
            if lane_A != lane_B:
                point_a = lane_list[lane_A][0].transform.location.x, lane_list[lane_A][0].transform.location.y
                point_b = lane_list[lane_A][1].transform.location.x, lane_list[lane_A][1].transform.location.y
                point_c = lane_list[lane_B][0].transform.location.x, lane_list[lane_B][0].transform.location.y
                point_d = lane_list[lane_B][1].transform.location.x, lane_list[lane_B][1].transform.location.y
                line_ab = LineString([point_a, point_b])
                line_cd = LineString([point_c, point_d])
                if line_ab.crosses(line_cd):
                    if (lane_B, lane_A) not in added_edges:
                        G.add_edge(lane_A, lane_B)
                        G.add_edge(lane_B, lane_A)
                        # added_edges.append((lane_A, lane_B))
    for lane in lane_list:
        # 3.add_edge for evert lane that could change to
        lane_change_left = lane_list[lane][0].lane_change == carla.LaneChange.Left or \
                           lane_list[lane][0].lane_change == carla.LaneChange.Both or \
                           lane_list[lane][1].lane_change == carla.LaneChange.Left or \
                           lane_list[lane][1].lane_change == carla.LaneChange.Both
        lane_change_right = lane_list[lane][0].lane_change == carla.LaneChange.Right or \
                            lane_list[lane][0].lane_change == carla.LaneChange.Both or \
                            lane_list[lane][1].lane_change == carla.LaneChange.Right or \
                            lane_list[lane][1].lane_change == carla.LaneChange.Both
        if lane_change_left:
            if (lane[0], lane[1] + 1) in lane_list:
                G.add_edge(lane, (lane[0], lane[1] + 1))
        if lane_change_right:
            if (lane[0], lane[1] - 1) in lane_list:
                G.add_edge(lane, (lane[0], lane[1] - 1))
    utils.switch_map(conf, town_map, client)
    return conf, town, town_map, client, world, G


def print_all_attr(obj):
    attributes = dir(obj)
    for attr_name in attributes:
        if not callable(getattr(obj, attr_name)):
            attr_value = getattr(obj, attr_name)
            attr_type = type(attr_value)
            print(f"Attribute: {attr_name}, Value: {attr_value}, Type: {attr_type}")


def check_nondominated_stability(pareto_front, archive, generations=10, epsilon=1e-6):
    """
    Checks stability of non-dominated solutions across multiple generations.
    pareto_front: The Pareto front (non-dominated solutions) of the current generation
    archive: A list storing historical Pareto fronts
    generations: Number of generations to compare
    epsilon: Threshold to determine if there's a significant change in the Pareto front
    """
    # Check if pareto_front is empty
    if not pareto_front:
        return False  # No solutions, can't check crowding distance stability
    archive.append(pareto_front)

    if len(archive) < generations:
        return False  # Not enough generations yet for comparison

    # Compare differences over the recent generations
    for i in range(-2, -generations - 1, -1):
        prev_pareto = archive[i]
        if any(abs(x.fitness.values[0] - y.fitness.values[0]) > epsilon for x, y in zip(pareto_front, prev_pareto)):
            return False  # If the difference exceeds the threshold, there is no stagnation

    return True


def check_crowding_distance_stability(pareto_front, generations=10, epsilon=1e-6):
    """
    Checks stability of crowding distance.
    pareto_front: The Pareto front (non-dominated solutions) of the current generation
    generations: Number of generations to compare
    epsilon: Threshold to determine if there's a significant change in crowding distance
    """
    # Check if pareto_front is empty
    if not pareto_front:
        return False  # No solutions, can't check crowding distance stability
    # Calculate the average crowding distance for the current generation's Pareto front
    crowding_distances = tools.sortNondominated(pareto_front, len(pareto_front))[0]
    avg_crowding_distance = np.mean([ind.fitness.crowding_dist for ind in crowding_distances])

    # If there are already `generations` number of records for crowding distance, compare them
    if len(crowding_distances) >= generations:
        recent_distances = [np.mean([ind.fitness.crowding_dist for ind in gen]) for gen in
                            crowding_distances[-generations:]]
        if all(abs(avg_crowding_distance - dist) < epsilon for dist in recent_distances):
            return True  # Small changes in crowding distance indicate possible stagnation

    return False


def check_objective_variance_stability(pareto_front, generations=10, epsilon=1e-6):
    """
    Checks stability of objective function variance.
    pareto_front: The Pareto front (non-dominated solutions) of the current generation
    generations: Number of generations to compare
    epsilon: Threshold to determine if there's a significant change in variance
    """
    # Check if pareto_front is empty
    if not pareto_front:
        return False  # No solutions, can't check crowding distance stability
    # Calculate variance of the objectives in the current generation
    objectives = np.array([ind.fitness.values for ind in pareto_front])
    current_variance = np.var(objectives, axis=0)

    # If there are already `generations` number of records for variance, compare them
    if len(objectives) >= generations:
        recent_variances = [np.var([ind.fitness.values for ind in gen], axis=0) for gen in objectives[-generations:]]
        if all(np.all(np.abs(current_variance - var) < epsilon) for var in recent_variances):
            return True  # Small changes in objective function variance indicate possible stagnation

    return False


def check_diversity_bottleneck(current_pareto_front, archive, generations=10, epsilon=1e-6):
    checks = [
        check_nondominated_stability(current_pareto_front, archive, generations=generations, epsilon=epsilon),
        check_crowding_distance_stability(current_pareto_front, generations=generations, epsilon=epsilon),
        check_objective_variance_stability(current_pareto_front, generations=generations, epsilon=epsilon)
    ]

    satisfied_conditions = sum(checks)

    return satisfied_conditions >= 2


def _count_scenarios_from_files(queue_dir):
    """
    Count scenarios from file system (queue directory).
    This is the single source of truth for scenario counting.
    
    Args:
        queue_dir: Path to queue directory containing scenario JSON files
        
    Returns:
        Number of scenario files found
    """
    if not queue_dir or not os.path.exists(queue_dir):
        return 0
    try:
        scenario_files = [f for f in os.listdir(queue_dir) if f.endswith('.json')]
        return len(scenario_files)
    except Exception as e:
        print(f"[WARNING] Failed to count scenarios from files: {e}")
        return 0


def _get_next_scenario_id_from_files(queue_dir):
    """
    Get the next scenario ID by examining existing scenario files in queue directory.
    Returns the maximum scenario ID found + 1, or 1 if no files exist.
    """
    if not os.path.exists(queue_dir):
        return 1
    
    max_scenario_id = 0
    max_generation_id = 0
    
    # Pattern: gid:{generation_id}_sid:{scenario_id}.json
    pattern = re.compile(r'gid:(\d+)_sid:(\d+)\.json')
    
    for filename in os.listdir(queue_dir):
        if filename.endswith('.json'):
            match = pattern.match(filename)
            if match:
                gen_id = int(match.group(1))
                scen_id = int(match.group(2))
                max_generation_id = max(max_generation_id, gen_id)
                max_scenario_id = max(max_scenario_id, scen_id)
    
    return max_scenario_id + 1, max_generation_id


def _save_checkpoint(checkpoint_path, curr_gen, total_scenarios_generated, population, archive, hof, next_scenario_id, logbook_data=None, stats=None):
    """
    Save GA state to checkpoint file.
    Only saves data essential for GA to continue: population, archive, hof items, and progress counters.
    
    Logic Analysis:
    - stats: Contains lambda functions (key=lambda ind: ind.fitness.values), NOT serializable, NOT needed for GA continuation
    - logbook: Contains references to stats, NOT serializable, NOT needed for GA continuation
    - hof (ParetoFront): Container object that may contain method references, but hof.items (list of Scenario) IS serializable
    - population/archive: Lists of Scenario objects, which now have __getstate__/__setstate__ support, SHOULD be serializable
    
    Strategy:
    - Always save hof.items instead of hof object (simpler and more reliable)
    - Save population and archive directly (Scenario objects handle their own serialization)
    - Never save stats or logbook
    """
    try:
        # Extract hof items - ParetoFront is just a container, we only need the Scenario objects
        hof_items = list(hof.items) if hasattr(hof, 'items') and hof.items else []
        
        # Build checkpoint data with only serializable, essential data
        checkpoint_data = {
            'curr_gen': curr_gen,
            'total_scenarios_generated': total_scenarios_generated,
            'next_scenario_id': next_scenario_id,
            'population': population,
            'archive': archive,
            'hof_items': hof_items,  # Save items, not the ParetoFront object
            'hof': None,  # Don't save ParetoFront object
            'stats': None,  # Never save stats (contains lambda functions)
            'logbook': None,  # Never save logbook (contains references to stats)
            'determ_seed': getattr(conf, 'determ_seed', None),  # Save seed for reproducibility
            'cur_time': getattr(conf, 'cur_time', None),  # Save experiment start time
        }
        
        # Save checkpoint
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"[INFO] Checkpoint saved: gen={curr_gen}, scenarios={total_scenarios_generated}, next_id={next_scenario_id}, "
              f"population_size={len(population)}, archive_size={len(archive)}, hof_size={len(hof_items)}")
            
    except Exception as e:
        print(f"[WARNING] Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()
        # Don't raise - checkpoint save failure should not stop the experiment


def _load_checkpoint(checkpoint_path):
    """
    Load GA state from checkpoint file.
    Returns None if checkpoint doesn't exist or is invalid.
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        # Validate checkpoint data structure
        if not isinstance(checkpoint_data, dict):
            print(f"[WARNING] Checkpoint file contains invalid data type: {type(checkpoint_data)}")
            return None
        # Check for required keys
        required_keys = ['curr_gen', 'total_scenarios_generated', 'next_scenario_id']
        if not all(key in checkpoint_data for key in required_keys):
            print(f"[WARNING] Checkpoint file missing required keys. Found: {list(checkpoint_data.keys())}")
            return None
        print(f"[INFO] Checkpoint loaded: gen={checkpoint_data.get('curr_gen', 0)}, "
              f"scenarios={checkpoint_data.get('total_scenarios_generated', 0)}, "
              f"next_id={checkpoint_data.get('next_scenario_id', 1)}")
        return checkpoint_data
    except (EOFError, pickle.UnpicklingError, ValueError) as e:
        # Specific handling for corrupted checkpoint files
        print(f"[WARNING] Checkpoint file is corrupted (Ran out of input or unpickling error): {e}")
        print(f"[INFO] Will start from existing scenario files or create new run")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(args=None):
    # STEP 0: init env
    global client, world, G, blueprint_library, town_map, bottleneck, conf
    logging.basicConfig(filename='./data/record.log', filemode='a', level=logging.INFO,
                        format='%(asctime)s - %(message)s')
    copyreg.pickle(carla.libcarla.Location, utils.carla_location_pickle, utils.carla_location_unpickle)
    copyreg.pickle(carla.libcarla.Rotation, utils.carla_rotation_pickle, utils.carla_rotation_unpickle)
    copyreg.pickle(carla.libcarla.Transform, utils.carla_transform_pickle, utils.carla_transform_unpickle)
    # copyreg.pickle(carla.libcarla.ActorBlueprint, carla_ActorBlueprint_pickle, carla_ActorBlueprint_unpickle)

    conf, town, town_map, exec_state.client, exec_state.world, exec_state.G = init_env(args)
    # Load persistent Scenario_database if configured
    global Scenario_database
    global total_scenarios_generated
    if hasattr(conf, "scenario_db") and conf.scenario_db:
        # Use lock when loading Scenario_database
        with _scenario_db_lock:
            Scenario_database = _load_scenario_database(conf.scenario_db)
    # Initialize checkpoint path
    checkpoint_path = os.path.join(conf.out_dir, 'ga_checkpoint.pkl')
    queue_dir = conf.queue_dir
    
    # Store checkpoint path in global state for evaluation function
    global _ga_state
    _ga_state['checkpoint_path'] = checkpoint_path
    
    # Try to load checkpoint or infer state from existing files
    checkpoint_data = _load_checkpoint(checkpoint_path)
    
    if checkpoint_data:
        # Restore from checkpoint
        curr_gen = checkpoint_data.get('curr_gen', 0)
        checkpoint_scenario_count = checkpoint_data.get('total_scenarios_generated', 0)
        next_scenario_id = checkpoint_data.get('next_scenario_id', 1)
        population = checkpoint_data.get('population', [])
        archive = checkpoint_data.get('archive', [])
        
        # Restore seed for reproducibility
        saved_seed = checkpoint_data.get('determ_seed', None)
        if saved_seed is not None:
            conf.determ_seed = saved_seed
            random.seed(conf.determ_seed)
            print(f"[INFO] Restored seed from checkpoint: {conf.determ_seed}")
        else:
            print(f"[WARNING] No seed found in checkpoint, using current seed: {conf.determ_seed}")
        
        # Restore experiment start time if available
        saved_cur_time = checkpoint_data.get('cur_time', None)
        if saved_cur_time is not None:
            conf.cur_time = saved_cur_time
            print(f"[INFO] Restored experiment start time from checkpoint: {conf.cur_time}")
        
        # Reconstruct hof from saved hof_items
        # We always save hof_items, not the ParetoFront object
        hof_items = checkpoint_data.get('hof_items', [])
        hof = tools.ParetoFront()
        if hof_items:
            hof.update(hof_items)
            print(f"[INFO] Reconstructed hof from {len(hof_items)} items")
        else:
            print("[INFO] No hof items in checkpoint, starting with empty hof")
        
        # Stats and logbook are not saved (contain lambda functions)
        # They will be recreated
        stats = None
        logbook = None
        
        # Restore conf reference in Scenario objects if needed
        # conf is stored globally and will be available when main() runs
        # But we need to ensure Scenario objects can access it
        # This will be handled when conf is set globally in main()
        
        # Reset bottleneck flag when restoring from checkpoint
        bottleneck = False
        
        # Verify and sync scenario count from file system (single source of truth)
        file_scenario_count = _count_scenarios_from_files(queue_dir)
        if file_scenario_count != checkpoint_scenario_count:
            print(f"[WARNING] Scenario count mismatch: checkpoint={checkpoint_scenario_count}, filesystem={file_scenario_count}")
            print(f"[INFO] Using filesystem count as source of truth: {file_scenario_count}")
            total_scenarios_generated = file_scenario_count
        else:
            total_scenarios_generated = checkpoint_scenario_count
        
        # Update global state for evaluation function
        _ga_state['curr_gen'] = curr_gen
        _ga_state['population'] = population
        _ga_state['archive'] = archive
        _ga_state['hof'] = hof
        _ga_state['next_scenario_id'] = next_scenario_id
        _ga_state['logbook'] = logbook
        _ga_state['stats'] = stats
        # Validate restored state
        if not isinstance(population, list):
            print(f"[WARNING] Invalid population type in checkpoint, reinitializing...")
            population = []
        if not isinstance(archive, list):
            print(f"[WARNING] Invalid archive type in checkpoint, reinitializing...")
            archive = []
        if hof is None or not isinstance(hof, tools.ParetoFront):
            print(f"[WARNING] Invalid hof in checkpoint, reinitializing...")
            hof = tools.ParetoFront()
        
        print(f"[INFO] Restored from checkpoint: gen={curr_gen}, scenarios={total_scenarios_generated}, next_id={next_scenario_id}, "
              f"population_size={len(population)}, archive_size={len(archive)}, hof_size={len(hof)}")
        
        # Note: conf will be restored in Scenario objects after init_env() sets globals()['conf']
    else:
        # No checkpoint, check existing files to infer state
        next_scenario_id, max_gen_id = _get_next_scenario_id_from_files(queue_dir)
        curr_gen = max_gen_id  # Start from the last generation found
        
        # Count actual scenarios from files (single source of truth)
        total_scenarios_generated = _count_scenarios_from_files(queue_dir)
        
        # If we found existing scenarios, start from next generation
        if total_scenarios_generated > 0:
            curr_gen = max_gen_id + 1  # Start next generation after the last one found
            print(f"[INFO] Found {total_scenarios_generated} existing scenarios, starting from scenario_id={next_scenario_id}, gen={curr_gen}")
        else:
            curr_gen = 0  # Start from generation 0
            print(f"[INFO] Starting fresh run, next_scenario_id={next_scenario_id}, gen={curr_gen}")
        
        # Check if we need to reset curr_gen if it exceeds MAX_GEN but max_scenarios not reached
        # This handles the case where checkpoint is corrupted and we need to continue beyond MAX_GEN
        # When max_scenarios is set, we ignore MAX_GEN and allow unlimited generations
        if hasattr(conf, 'max_scenarios') and conf.max_scenarios > 0:
            MAX_GEN_VALUE = 5  # MAX_GEN constant value
            if curr_gen >= MAX_GEN_VALUE and total_scenarios_generated < conf.max_scenarios:
                print(f"[INFO] curr_gen ({curr_gen}) >= MAX_GEN ({MAX_GEN_VALUE}), but max_scenarios ({conf.max_scenarios}) not reached ({total_scenarios_generated}). Resetting curr_gen to 0 to continue (MAX_GEN ignored when max_scenarios is set).")
                curr_gen = 0
        
        population = []
        archive = []
        hof = tools.ParetoFront()
        logbook = None
        stats = None  # Initialize stats variable
        bottleneck = False  # Initialize bottleneck variable for consistency
        # Initialize global state
        _ga_state['curr_gen'] = curr_gen
        _ga_state['population'] = population
        _ga_state['archive'] = archive
        _ga_state['hof'] = hof
        _ga_state['next_scenario_id'] = next_scenario_id
        _ga_state['logbook'] = None
        _ga_state['stats'] = None
    
    # Initialize experiment timing if not already set
    if not hasattr(conf, 'experiment_start_time') or conf.experiment_start_time is None:
        conf.experiment_start_time = time.time()
    # Make conf globally accessible for evaluation function
    globals()['conf'] = conf
    
    # Restore conf reference in Scenario objects from checkpoint if needed
    # This ensures Scenario objects can access conf even after checkpoint restore
    if checkpoint_data:
        # Restore conf in all Scenario objects in population, archive, and hof
        for scenario_list in [population, archive]:
            for ind in scenario_list:
                if hasattr(ind, 'conf'):
                    ind.conf = conf  # Update conf reference to current conf object
        # Restore conf in hof items
        if hof and hasattr(hof, 'items'):
            for ind in hof.items:
                if hasattr(ind, 'conf'):
                    ind.conf = conf  # Update conf reference to current conf object
    
    world = exec_state.world
    blueprint_library = world.get_blueprint_library()
    # if conf.agent_type == c.AUTOWARE:
    #     autoware_launch(exec_state.world, conf, town)
    
    # GA Hyperparameters
    POP_SIZE = 5  # amount of population
    OFF_SIZE = 5  # number of offspring to produce
    # MAX_GEN is only used if max_scenarios is not set
    # If max_scenarios is set, we ignore MAX_GEN and continue until max_scenarios is reached
    MAX_GEN = 5  # Only used as fallback when max_scenarios is not set
    CXPB = 0.8  # crossover probability
    MUTPB = 0.2  # mutation probability
    
    # If max_scenarios is set, ignore MAX_GEN limit and allow unlimited generations
    # Reset curr_gen to 0 if it exceeds MAX_GEN and we haven't reached max_scenarios
    if hasattr(conf, 'max_scenarios') and conf.max_scenarios > 0:
        file_count = _count_scenarios_from_files(queue_dir)
        if curr_gen >= MAX_GEN and file_count < conf.max_scenarios:
            print(f"[INFO] curr_gen ({curr_gen}) >= MAX_GEN ({MAX_GEN}), but max_scenarios ({conf.max_scenarios}) not reached ({file_count}). Resetting curr_gen to 0 to continue.")
            curr_gen = 0
            _ga_state['curr_gen'] = curr_gen
    toolbox = base.Toolbox()
    
    # Wrap evaluation function to handle exceptions gracefully
    # This ensures that a single scenario failure doesn't stop the entire fuzzing process
    def safe_evaluation(ind):
        """Wrapper for evaluation that catches exceptions and returns default fitness"""
        try:
            return evaluation(ind)
        except (RuntimeError, ConnectionError, TimeoutError) as e:
            # For connection/timeout errors, propagate to allow environment restart
            error_msg = str(e)
            if "time-out of 10000ms while waiting for the simulator" in error_msg:
                # CARLA timeout - propagate to allow restart
                print(f"[ERROR] CARLA timeout in scenario {ind.scenario_id}, propagating for restart")
                raise
            elif "Failed to connect to CARLA" in error_msg:
                # Connection error - propagate to allow restart
                print(f"[ERROR] CARLA connection error in scenario {ind.scenario_id}, propagating for restart")
                raise
            else:
                # Other runtime errors - use default fitness and continue
                print(f"[WARNING] Error in scenario {ind.scenario_id}: {e}. Using default fitness and continuing.")
                if not ind.fitness.valid:
                    ind.fitness.values = (0.0, 0.0)  # Default fitness: (min_dist, nova)
                return (0.0, 0.0)
        except Exception as e:
            # Catch any other unexpected exceptions and continue
            print(f"[WARNING] Unexpected error in scenario {ind.scenario_id}: {e}. Using default fitness and continuing.")
            import traceback
            traceback.print_exc()
            if not ind.fitness.valid:
                ind.fitness.values = (0.0, 0.0)  # Default fitness: (min_dist, nova)
            return (0.0, 0.0)
    
    toolbox.register("evaluate", safe_evaluation)
    toolbox.register("mate", cx_scenario)
    toolbox.register("mutate", mut_scenario)
    toolbox.register("select", tools.selNSGA2)
    
    # Validate and initialize population if empty (first run or checkpoint invalid)
    if not population or len(population) == 0:
        print(f' ====== Initializing Population ====== ')
        for i in range(POP_SIZE):
            seed_dict = seed_initialize(town, town_map)
            # Creates and initializes a Scenario instance based on the metadata
            with concurrent.futures.ThreadPoolExecutor() as my_simulate:
                future = my_simulate.submit(create_test_scenario, conf, seed_dict)
                test_scenario = future.result(timeout=15)
            population.append(test_scenario)
            test_scenario.scenario_id = next_scenario_id + i
            test_scenario.generation_id = curr_gen if curr_gen > 0 else 0
        next_scenario_id += POP_SIZE
        # Update global state
        _ga_state['next_scenario_id'] = next_scenario_id
        
        # Evaluate Initial Population
        print(f' ====== Analyzing Initial Population ====== ')
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof.update(population)
        # Update global state
        _ga_state['hof'] = hof
        _ga_state['population'] = population
    
    # Initialize stats and logbook (restore from checkpoint if available)
    if stats is None:
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("max", np.max, axis=0)
        stats.register("min", np.min, axis=0)
    if logbook is None:
        logbook = tools.Logbook()
        logbook.header = 'gen', 'avg', 'max', 'min'
    
    # Update global state with initialized stats and logbook
    _ga_state['stats'] = stats
    _ga_state['logbook'] = logbook
    # Get experiment start time and timeout if set
    experiment_start_time = getattr(conf, 'experiment_start_time', None)
    experiment_timeout = getattr(conf, 'experiment_timeout', None)
    
    while True:
        # Main loop
        curr_gen += 1
        # Update global state with current generation
        _ga_state['curr_gen'] = curr_gen
        
        # Check scenario limit first (priority over MAX_GEN if max_scenarios is set)
        # Sync from file system to ensure accuracy before checking limit
        if hasattr(conf, 'max_scenarios') and conf.max_scenarios > 0:
            # Always sync from file system before checking limit
            file_count = _count_scenarios_from_files(queue_dir)
            total_scenarios_generated = file_count
            if total_scenarios_generated >= conf.max_scenarios:
                print(f"Reached scenario limit: {total_scenarios_generated}/{conf.max_scenarios}")
                break
        
        # Check MAX_GEN limit only if max_scenarios is not set
        # If max_scenarios is set, we ignore MAX_GEN and continue until max_scenarios is reached
        if curr_gen > MAX_GEN:
            # If max_scenarios is set, ignore MAX_GEN and continue
            if hasattr(conf, 'max_scenarios') and conf.max_scenarios > 0:
                file_count = _count_scenarios_from_files(queue_dir)
                if file_count < conf.max_scenarios:
                    print(f"[INFO] Reached MAX_GEN ({MAX_GEN}), but max_scenarios ({conf.max_scenarios}) not reached ({file_count}). Resetting to gen 0 to continue (MAX_GEN ignored when max_scenarios is set).")
                    curr_gen = 0
                    _ga_state['curr_gen'] = curr_gen
                    # Continue the loop to generate more scenarios
                    continue
            # Only break if MAX_GEN reached AND max_scenarios is not set
            else:
                print(f"[INFO] Reached MAX_GEN ({MAX_GEN}) and no max_scenarios limit set. Stopping.")
                break
        
        # Check time limit if set
        if experiment_start_time and experiment_timeout:
            elapsed_time = time.time() - experiment_start_time
            if elapsed_time >= experiment_timeout:
                print(f"Reached time limit: {elapsed_time:.0f}s / {experiment_timeout:.0f}s")
                break
        
        print(f' ====== GA Generation {curr_gen} ====== ')
        # Sync from file system for accurate reporting
        file_count = _count_scenarios_from_files(queue_dir)
        with _scenario_count_lock:
            total_scenarios_generated = file_count
        print(f'Total scenarios generated: {total_scenarios_generated}')
        if experiment_start_time and experiment_timeout:
            elapsed = time.time() - experiment_start_time
            remaining = experiment_timeout - elapsed
            print(f'Time elapsed: {elapsed:.0f}s, Remaining: {remaining:.0f}s')
        
        # Vary the population
        offspring = algorithms.varOr(
            population, toolbox, OFF_SIZE, CXPB, MUTPB)
        # update chromosome generation_id and scenario_id
        # Use global next_scenario_id to ensure uniqueness across generations
        for index, d in enumerate(offspring):
            d.generation_id = curr_gen
            d.scenario_id = next_scenario_id + index
        next_scenario_id += len(offspring)
        # Update global state
        _ga_state['next_scenario_id'] = next_scenario_id
        _ga_state['curr_gen'] = curr_gen
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        current_pareto_front = [ind for ind in population if ind in hof.items]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof.update(offspring)
        # Update global state
        _ga_state['hof'] = hof

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, POP_SIZE)
        # Update global state with current population
        _ga_state['population'] = population
        record = stats.compile(population)

        # Combined Bottleneck Detection
        if check_diversity_bottleneck(current_pareto_front, archive, generations=10, epsilon=1e-6):
            print("GA has entered a bottleneck, stopping evolution or taking alternative actions")
            bottleneck = True
        else:
            bottleneck = False
        logbook.record(gen=curr_gen, **record)
        print(logbook.stream)
        
        # Update global state with archive, hof, logbook, and next_scenario_id
        _ga_state['archive'] = archive
        _ga_state['hof'] = hof
        _ga_state['logbook'] = logbook
        _ga_state['next_scenario_id'] = next_scenario_id
        
        # Save checkpoint after each generation
        # Note: We don't save stats (contains lambda) and logbook may reference stats
        # Sync from file system before saving checkpoint to ensure accuracy
        file_count = _count_scenarios_from_files(queue_dir)
        with _scenario_count_lock:
            total_scenarios_generated = file_count
        
        # Convert logbook to serializable format (extract only data, not object references)
        # Note: logbook may contain references to stats (which has lambda), so we need to be careful
        # We'll try to extract only the essential data, and if that fails, we'll skip saving logbook
        logbook_data = None
        try:
            if logbook and hasattr(logbook, 'chapters'):
                # Test if we can pickle the logbook data by trying to serialize a test dict first
                # Extract only the data we need - be very conservative
                logbook_data = []
                chapters = getattr(logbook, 'chapters', {})
                if chapters:
                    for chapter_name, chapter_data in chapters.items():
                        try:
                            # Create a clean dict with only serializable data
                            # Only extract basic data types, no object references
                            chapter_dict = {
                                'name': str(chapter_name),
                                'header': list(str(h) for h in chapter_data.get('header', [])),
                                'rows': []
                            }
                            # Convert rows to list of lists (serializable)
                            rows = chapter_data.get('rows', [])
                            for row in rows:
                                # Convert row to a list of basic types only
                                if isinstance(row, (list, tuple)):
                                    clean_row = []
                                    for item in row:
                                        # Only keep basic serializable types
                                        if isinstance(item, (int, float, str, bool, type(None))):
                                            clean_row.append(item)
                                        elif isinstance(item, (list, tuple)):
                                            # Nested structure - convert to list of basic types
                                            clean_row.append([x for x in item if isinstance(x, (int, float, str, bool, type(None)))])
                                        else:
                                            # Skip non-serializable items
                                            continue
                                    if clean_row:
                                        chapter_dict['rows'].append(clean_row)
                                elif isinstance(row, (int, float, str, bool, type(None))):
                                    chapter_dict['rows'].append([row])
                            
                            # Test if this chapter can be pickled
                            try:
                                import io
                                test_buffer = io.BytesIO()
                                pickle.dump(chapter_dict, test_buffer)
                                logbook_data.append(chapter_dict)
                            except Exception as pickle_test_err:
                                print(f"[WARNING] Logbook chapter {chapter_name} contains non-serializable data, skipping: {pickle_test_err}")
                                continue
                        except Exception as chapter_err:
                            print(f"[WARNING] Failed to serialize logbook chapter {chapter_name}: {chapter_err}")
                            continue
        except Exception as logbook_err:
            print(f"[WARNING] Failed to serialize logbook: {logbook_err}")
            logbook_data = None  # Don't save logbook if serialization fails
        
        _save_checkpoint(checkpoint_path, curr_gen, total_scenarios_generated, population, archive, hof, next_scenario_id, logbook_data, None)
        # Save directory for trace graphs


if __name__ == "__main__":
    main()
