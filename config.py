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
    # proj_root = get_proj_root()
    #
    # dist_path = os.path.join(proj_root, "carla/PythonAPI/carla/dist")
    # glob_path = os.path.join(dist_path, "carla-*%d.%d-%s.egg" % (
    #     sys.version_info.major,
    #     sys.version_info.minor,
    #     "win-amd64" if os.name == "nt" else "linux-x86_64"
    # ))

    try:
        # api_path = glob.glob(glob_path)[0]carla-0.9.13-py3.6-linux-x86_64.egg
        api_path = "./carla/PythonAPI/carla-0.9.13-py3.6-linux-x86_64.egg"
    except IndexError:
        print("Couldn't set Carla API path.")
        exit(-1)

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

    def set_paths(self):
        self.queue_dir = os.path.join(self.out_dir, "queue")
        self.picture_dir = os.path.join(self.out_dir, "picture")
        self.error_dir = os.path.join(self.out_dir, "errors")
        # self.meta_file = os.path.join(self.out_dir, "meta")
        self.npc_dir = os.path.join(self.out_dir, "npc")
        self.time_record_dir = os.path.join(self.out_dir, "time_record")
        self.cam_dir = os.path.join(self.out_dir, "camera")
        self.rosbag_dir = os.path.join(self.out_dir, "rosbags")

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
