import glob
import os
import pdb
import sys

import constants as c


def get_proj_root():
    config_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(config_path)
    proj_root = os.path.dirname(src_dir)
    return proj_root


def set_carla_api_path():
    """
    Ensure CARLA PythonAPI egg is on sys.path.

    This resolves the egg path relative to the project root so that
    it works no matter what the current working directory is.
    """
    proj_root = get_proj_root()

    # We **only** accept CARLA 0.9.13 to match the running simulator.
    target_version = "0.9.13"
    py_ver = f"py{sys.version_info.major}.{sys.version_info.minor}"
    platform_tag = "win-amd64" if os.name == "nt" else "linux-x86_64"

    # Prefer the dist/ folder with versioned eggs if available
    dist_path = os.path.join(proj_root, "carla", "PythonAPI", "carla", "dist")

    # 1) Exact pattern under dist/, e.g. carla-0.9.13-py3.6-linux-x86_64.egg
    exact_pattern = os.path.join(
        dist_path, f"carla-{target_version}-{py_ver}-{platform_tag}.egg"
    )

    candidate_paths = []
    if os.path.exists(exact_pattern):
        candidate_paths.append(exact_pattern)

    # 2) If exact name不存在, 在 dist 目录中查找所有包含 0.9.13 的 egg
    if not candidate_paths and os.path.isdir(dist_path):
        for path in glob.glob(os.path.join(dist_path, "carla-*.egg")):
            if target_version in os.path.basename(path):
                candidate_paths.append(path)

    # 3) 旧版 fallback：项目根目录下的固定文件名（兼容之前的 py3.6 环境）
    if not candidate_paths:
        fallback36 = os.path.join(
            proj_root,
            "carla",
            "PythonAPI",
            "carla-0.9.13-py3.6-linux-x86_64.egg",
        )
        if os.path.exists(fallback36):
            candidate_paths.append(fallback36)
    
    # 3.5) 检查项目目录下的 CARLA（如果 proj_root 不是项目目录本身）
    if not candidate_paths:
        # 获取 config.py 所在目录（项目目录）
        config_path = os.path.abspath(__file__)
        project_dir = os.path.dirname(config_path)  # ScenarioFuzz-LLM 目录
        project_carla = os.path.join(
            project_dir,
            "carla",
            "PythonAPI",
            f"carla-{target_version}-{py_ver}-{platform_tag}.egg",
        )
        if os.path.exists(project_carla):
            candidate_paths.append(project_carla)

    # 4) 兼容你之前的备份路径：~/backup/carla-autoware/carla-api/carla-0.9.13-py3.7-linux-x86_64.egg
    #    这里根据当前 Python 版本自动拼接 py{major}.{minor}
    if not candidate_paths:
        backup_api = os.path.join(
            proj_root,
            "backup",
            "carla-autoware",
            "carla-api",
            f"carla-{target_version}-{py_ver}-{platform_tag}.egg",
        )
        if os.path.exists(backup_api):
            candidate_paths.append(backup_api)

    if not candidate_paths:
        print("Couldn't find Carla 0.9.13 PythonAPI egg.")
        print("Expected one of:")
        print("  -", exact_pattern)
        print(
            "  - Any egg in",
            dist_path,
            "whose filename contains '0.9.13'",
        )
        print("  -", fallback36)
        print("  -", backup_api)
        print(
            "Please install/build CARLA 0.9.13 PythonAPI and ensure the egg file exists at one of the above locations."
        )
        sys.exit(-1)

    # 为了确定性，按字典序选第一个
    api_path = sorted(candidate_paths)[0]

    if api_path not in sys.path:
        sys.path.append(api_path)
        print(f"API: {api_path}")


class Config:
    """
    A class defining fuzzing configuration and helper methods.
    An instance of this class should be created by the main module (fuzzer.py)
    and then be shared across other modules as a context handler.
    """

    def __init__(self):
        self.score_dir = None
        self.rosbag_dir = None
        self.cam_dir = None
        # self.meta_file = None
        self.npc_dir = None
        self.time_record_dir = None
        self.error_dir = None
        self.picture_dir = None
        self.queue_dir = None
        self.recorder_dir = None
        self.debug = True

        # simulator config
        self.sim_host = "localhost"
        self.sim_port = 0
        self.sim_tm_port = 0

        # Fuzzer config
        self.topo_k = 2
        self.immobile_percentage = 0  # the percentage of the npcs is immobile forever
        self.max_cycles = 0
        self.max_mutation = 0
        self.num_dry_runs = 1
        self.density = 1
        self.num_mutation_car = 1
        self.density = 1
        self.no_traffic_lights = True

        # Fuzzing metadata
        self.town = None
        self.cur_time = None
        self.determ_seed = None
        self.out_dir = None
        self.seed_dir = None

        # Target config
        self.agent_type = c.AUTOWARE  # c.AUTOWARE

        # Enable/disable Various Checks
        self.check_dict = {
            "speed": True,
            "lane": True,
            "crash": True,
            "stuck": True,
            "red": False,
            "other": True,
        }

        # Functional testing
        self.function = "general"

        # Sim-debug settings
        self.view = c.BIRDSEYE
        
        # RAG configuration
        self.enable_rag = False
        self.rag_k = 5  # Number of top-k scenarios to retrieve
        self.rag_config_path = "./config/rag_config.json"
        # Similarity / feature configuration (used by SimilarityComparison)
        self.similarity_scoring_method = "answer2"
        self.hybrid_embedding_weight = 0.6
        self.feature_position_weight = 0.3
        self.feature_speed_weight = 0.3
        self.feature_angular_accel_weight = 0.2
        self.feature_relative_position_weight = 0.2
        
        # Enhanced RAG configuration
        self.use_enhanced_rag = False  # Use EnhancedRAGEngine instead of RAGEngine
        self.use_hybrid_search = True  # Enable hybrid search (vector + BM25)
        self.hybrid_alpha = 0.7  # Weight for vector retrieval in hybrid search (0-1)
        self.use_reranking = True  # Enable reranking after retrieval
        self.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model for reranking
        
        # Note: Metrics calculation has been moved to experiments/analysis/calculate_metrics.py
        # This decouples data collection from metrics calculation

        # GPT / Scenario database configuration
        # By default we enable GPT-based evaluation; this can be disabled
        # for non-GPT baselines such as TM-Fuzzer experiments.
        self.enable_gpt_evaluation = True
        self.scenario_db = "./data/scenario_db.json"
        self.gpt_log_dir = "./data/gpt_logs"

    def set_paths(self):
        self.queue_dir = os.path.join(self.out_dir, "queue")
        self.picture_dir = os.path.join(self.out_dir, "picture")
        self.error_dir = os.path.join(self.out_dir, "errors")
        # self.meta_file = os.path.join(self.out_dir, "meta")
        self.npc_dir = os.path.join(self.out_dir, "npc")
        self.time_record_dir = os.path.join(self.out_dir, "time_record")
        self.cam_dir = os.path.join(self.out_dir, "camera")
        self.rosbag_dir = os.path.join(self.out_dir, "rosbags")
        self.recorder_dir = os.path.join(self.out_dir, "recorder")

    # def enqueue_seed_scenarios(self):
    #     try:
    #         seed_scenarios = os.listdir(self.seed_dir)
    #     except:
    #         print("[-] Error - cannot find seed directory ({})".format(self.seed_dir))
    #         sys.exit(-1)
    #
    #     queue = [seed for seed in seed_scenarios if not seed.startswith(".")
    #              and seed.endswith(".json")]
    #
    #     return queue
