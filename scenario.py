import cProfile
import json
import os
import pdb
import shutil
from typing import List

import carla
import cv2
import numpy as np
import deap.base

from npc import NPC
# from cluster import draw_picture, shift_scale_points_group
from simulate import simulate
import constants as c
from states import ScenarioState

colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 0, 0),
    (128, 128, 0), (0, 128, 0)
]


def get_seed_sp_transform(seed):
    sp = carla.Transform(
        carla.Location(seed["sp_x"], seed["sp_y"], seed["sp_z"]),
        carla.Rotation(seed["roll"], seed["yaw"], seed["pitch"])
    )

    return sp


def get_seed_wp_transform(seed):
    wp = carla.Transform(
        carla.Location(seed["wp_x"], seed["wp_y"], seed["wp_z"]),
        carla.Rotation(0.0, seed["wp_yaw"], 0.0)
    )

    return wp


class ScenarioFitness(deap.base.Fitness):
    """
    Class to represent weight of each fitness function
    """
    # minimize the closest distance between a pair of ADC
    # for test
    weights = (-1.0, -1.0, 5.0)
    """
    Todo: note: 
    """


class Scenario:
    generation_id: int = -1
    scenario_id: int = -1
    fitness: deap.base.Fitness = ScenarioFitness()
    seed_data = {}
    town = None
    weather = {}
    npc_now = []
    npc_list: List[NPC]
    driving_quality_score = None
    found_error = False
    username = os.getenv("USER")

    def __init__(self, conf, seed_data):
        """
        When initializing, perform dry run and get the oracle state
        """
        self.log_filename = None
        self.conf = conf
        self.seed_data = seed_data
        self.state = ScenarioState()

        self.weather["cloud"] = 0
        self.weather["rain"] = 0
        self.weather["wind"] = 0
        self.weather["fog"] = 0
        self.weather["wetness"] = 0
        self.weather["angle"] = 0
        self.weather["altitude"] = 90


        self.npc_now = []
        self.npc_list = []
        self.driving_quality_score = 0
        self.found_error = False
        self.mutate_info = None

        self.sp = {
            "Location": (self.seed_data["sp_x"], self.seed_data["sp_y"], self.seed_data["sp_z"]),
            "Rotation": (self.seed_data["roll"], self.seed_data["yaw"], self.seed_data["pitch"])
        }
        self.town = self.seed_data["map"]
        # utils.switch_map(conf, self.town, client)

    def __getstate__(self):
        """
        Custom pickle support for Scenario objects.
        Excludes non-serializable objects (conf, state.client/world/G, town) that will be restored on load.
        """
        state = self.__dict__.copy()
        # Don't save conf - it will be restored from globals on load
        state['conf'] = None
        # Don't save town (CARLA Map object) - it's not serializable
        if 'town' in state:
            state['town'] = None
        # Check seed_data for CARLA Map objects
        if 'seed_data' in state and isinstance(state['seed_data'], dict):
            seed_data_copy = state['seed_data'].copy()
            # If seed_data contains a CARLA Map object in 'map' field, replace it with string
            if 'map' in seed_data_copy:
                map_value = seed_data_copy['map']
                # Check if it's a CARLA Map object (has get_waypoint method)
                if hasattr(map_value, 'get_waypoint'):
                    # It's a CARLA Map object, replace with None or string representation
                    # The map will be restored from exec_state.world.get_map() on load
                    seed_data_copy['map'] = None
                elif isinstance(map_value, str):
                    # Already a string, keep it
                    pass
                else:
                    # Unknown type, set to None for safety
                    seed_data_copy['map'] = None
            state['seed_data'] = seed_data_copy
        # Don't save CARLA objects in state - they will be reinitialized on load
        if hasattr(self, 'state') and self.state:
            state_copy = self.state.__dict__.copy()
            # Clear CARLA object references
            state_copy['client'] = None
            state_copy['world'] = None
            state_copy['G'] = None
            state_copy['spawn_failed_object'] = None
            # Clean laneinvasion_event list - remove CARLA LaneInvasionEvent objects
            # Only keep serializable information (frame, timestamp)
            if 'laneinvasion_event' in state_copy and state_copy['laneinvasion_event']:
                state_copy['laneinvasion_event'] = [
                    {'frame': getattr(e, 'frame', None), 'timestamp': getattr(e, 'timestamp', None)}
                    if hasattr(e, 'frame') else None
                    for e in state_copy['laneinvasion_event']
                ]
            
            # Clean closest_cars_list - remove CARLA Vehicle objects
            # Only keep serializable data (vehicle id, location, etc.)
            if 'closest_cars_list' in state_copy and state_copy['closest_cars_list']:
                cleaned_list = []
                for car in state_copy['closest_cars_list']:
                    if isinstance(car, dict):
                        # Already in dictionary format, keep as is
                        cleaned_list.append(car)
                    elif hasattr(car, 'id'):
                        # CARLA Vehicle object, convert to dictionary
                        try:
                            transform = car.get_transform()
                            cleaned_list.append({
                                'id': car.id,
                                'type_id': getattr(car, 'type_id', 'unknown'),
                                'location': (transform.location.x, transform.location.y, transform.location.z),
                                'rotation': (transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll)
                            })
                        except Exception:
                            # If unable to get transform, just keep id
                            cleaned_list.append({'id': car.id})
                    else:
                        # Other type, convert to string
                        try:
                            cleaned_list.append(str(car))
                        except Exception:
                            # If even string conversion fails, skip
                            pass
                state_copy['closest_cars_list'] = cleaned_list
            
            # Ensure collision_to is a basic type (int or None)
            if 'collision_to' in state_copy:
                if hasattr(state_copy['collision_to'], 'id'):
                    # If it's an Actor object, extract id
                    state_copy['collision_to'] = state_copy['collision_to'].id
                elif not isinstance(state_copy['collision_to'], (int, type(None))):
                    # If not int or None, set to None
                    state_copy['collision_to'] = None
            
            # Validate drawn_points is serializable
            if 'drawn_points' in state_copy and state_copy['drawn_points']:
                try:
                    # Test if set is serializable
                    import pickle
                    pickle.dumps(state_copy['drawn_points'])
                except Exception:
                    # If not serializable, convert to list
                    try:
                        state_copy['drawn_points'] = list(state_copy['drawn_points'])
                    except Exception:
                        # If even list conversion fails, create empty list
                        state_copy['drawn_points'] = []
            
            state['state'] = state_copy
        return state

    def __setstate__(self, state):
        """
        Custom unpickle support for Scenario objects.
        Restores conf from globals and reinitializes state.
        Note: conf will be restored in main() after init_env() sets globals()['conf']
        """
        self.__dict__.update(state)
        
        # Try to restore conf from globals immediately
        # This is more robust than waiting for main() to set it
        if 'conf' not in self.__dict__ or self.conf is None:
            if 'conf' in globals() and globals()['conf'] is not None:
                self.conf = globals()['conf']
                # Verify conf has required attributes
                if not hasattr(self.conf, 'queue_dir'):
                    print(f"[WARNING] Restored conf from globals but missing required attributes")
            else:
                # Try to get conf from fuzzer module (it may be set there)
                try:
                    import fuzzer
                    if hasattr(fuzzer, 'conf') and fuzzer.conf is not None:
                        self.conf = fuzzer.conf
                    else:
                        self.conf = None
                except (ImportError, AttributeError):
                    self.conf = None
                # Don't log warning here - conf will be restored later in main() or checkpoint loading
                # Only log if we're sure it won't be restored (which we can't know at this point)
        
        # Validate conf if it exists
        if self.conf is not None:
            required_attrs = ['queue_dir', 'out_dir', 'timeout']
            missing_attrs = [attr for attr in required_attrs if not hasattr(self.conf, attr)]
            if missing_attrs:
                print(f"[WARNING] Scenario {getattr(self, 'scenario_id', 'unknown')}: "
                      f"conf is missing required attributes: {missing_attrs}")
        
        # Reinitialize state if needed
        if 'state' in self.__dict__ and self.state:
            from states import ScenarioState
            if isinstance(self.state, dict):
                # Reconstruct ScenarioState from dict
                new_state = ScenarioState()
                new_state.__dict__.update(self.state)
                self.state = new_state
            elif not isinstance(self.state, ScenarioState):
                # If state is not a ScenarioState object, create a new one
                self.state = ScenarioState()
        
        # Restore CARLA object references from exec_state if available
        # This ensures state.client, state.world, state.G are set correctly
        if hasattr(self, 'state') and self.state:
            try:
                import fuzzer
                if hasattr(fuzzer, 'exec_state'):
                    exec_state = fuzzer.exec_state
                    if exec_state.client is not None:
                        self.state.client = exec_state.client
                    if exec_state.world is not None:
                        self.state.world = exec_state.world
                    if exec_state.G is not None:
                        self.state.G = exec_state.G
            except (AttributeError, ImportError) as e:
                # exec_state not available yet, will be set later
                pass
        
        # Restore town (CARLA Map) from exec_state if available
        # town was set to None during serialization
        if hasattr(self, 'town') and self.town is None:
            try:
                import fuzzer
                if hasattr(fuzzer, 'exec_state') and hasattr(fuzzer.exec_state, 'world'):
                    if fuzzer.exec_state.world is not None:
                        self.town = fuzzer.exec_state.world.get_map()
                        # Also update seed_data['map'] if it exists and was set to None
                        if hasattr(self, 'seed_data') and isinstance(self.seed_data, dict):
                            if self.seed_data.get('map') is None:
                                self.seed_data['map'] = self.town
            except (AttributeError, ImportError) as e:
                # exec_state not available yet, will be set later
                pass
        
        # Restore CARLA object references from exec_state if available
        # This ensures state.client, state.world, state.G are set correctly
        if hasattr(self, 'state') and self.state:
            try:
                import fuzzer
                if hasattr(fuzzer, 'exec_state'):
                    exec_state = fuzzer.exec_state
                    if exec_state.client is not None:
                        self.state.client = exec_state.client
                    if exec_state.world is not None:
                        self.state.world = exec_state.world
                    if exec_state.G is not None:
                        self.state.G = exec_state.G
            except (AttributeError, ImportError) as e:
                # exec_state not available yet, will be set later
                pass

    def get_distance_from_player(self, location):
        sp = get_seed_sp_transform(self.seed_data)
        return location.distance(sp.location)

    def dump_states(self, state, log_type):
        if self.conf.debug:
            print("[debug] dumping {} data".format(log_type))
        event_dict = {
            "crash": state.crashed,
            "stuck": state.stuck,
            "lane_invasion": state.laneinvaded,
            "red": state.red_violation,
            "speeding": state.speeding,
            "other": state.other_error,
            "other_error_val": state.other_error_val
        }
        config_dict = {
            "fps": c.FRAME_RATE,
            "max_dist_from_player": c.MAX_DIST_FROM_PLAYER,
            "min_dist_from_player": c.MIN_DIST_FROM_PLAYER,
            "abort_seconds": self.conf.timeout,
            "wait_autoware_num_topics": c.WAIT_AUTOWARE_NUM_NODES
        }

        # state_dict = {"fuzzing_start_time": self.conf.cur_time, "determ_seed": self.conf.determ_seed,
        #               "seed": self.seed_data, "weather": self.weather, "autoware_cmd": state.autoware_cmd,
        #               "autoware_goal": state.autoware_goal, "first_frame_id": state.first_frame_id,
        #               "first_sim_elapsed_time": state.first_sim_elapsed_time, "sim_start_time": state.sim_start_time,
        #               "num_frames": state.num_frames, "elapsed_time": state.elapsed_time, "events": event_dict,
        #               "config": config_dict}

        state_dict = {"events": event_dict, "config": config_dict}
        filename = "gid:{}_sid:{}.json".format(self.generation_id, self.scenario_id)
        if log_type == "queue":
            out_dir = self.conf.queue_dir
        with open(os.path.join(out_dir, filename), "w") as fp:
            json.dump(state_dict, fp)
        if self.conf.debug:
            print("[debug] dumped")
        return filename

    def run_test(self, exec_state):
        # Ensure conf is not None - restore from globals if needed
        if self.conf is None:
            # Try to restore from globals
            if 'conf' in globals() and globals()['conf'] is not None:
                self.conf = globals()['conf']
                print(f"[INFO] Restored conf from globals for scenario {getattr(self, 'scenario_id', 'unknown')}")
            else:
                # Last resort: try to get from fuzzer module
                try:
                    import fuzzer
                    if hasattr(fuzzer, 'conf') and fuzzer.conf is not None:
                        self.conf = fuzzer.conf
                        print(f"[INFO] Restored conf from fuzzer module for scenario {getattr(self, 'scenario_id', 'unknown')}")
                    else:
                        raise RuntimeError(
                            f"conf is None for scenario {getattr(self, 'scenario_id', 'unknown')}. "
                            f"Cannot run test. This usually indicates a serialization/deserialization issue. "
                            f"Ensure init_env() has been called and conf is set globally."
                        )
                except (ImportError, AttributeError):
                    raise RuntimeError(
                        f"conf is None for scenario {getattr(self, 'scenario_id', 'unknown')}. "
                        f"Cannot run test. This usually indicates a serialization/deserialization issue. "
                        f"Ensure init_env() has been called and conf is set globally."
                    )
        
        # Validate conf has required attributes
        if self.conf is not None:
            required_attrs = ['queue_dir', 'out_dir', 'timeout']
            missing_attrs = [attr for attr in required_attrs if not hasattr(self.conf, attr)]
            if missing_attrs:
                raise RuntimeError(
                    f"conf is missing required attributes: {missing_attrs}. "
                    f"This indicates conf was not properly restored from serialization."
                )
        
        if self.conf.debug:
            print("[debug] use scenario:id=", self.scenario_id)
        self.reload_state()
        sp = get_seed_sp_transform(self.seed_data)
        wp = get_seed_wp_transform(self.seed_data)
        ret, self.npc_list, self.state = simulate(
            conf=self.conf,
            state=self.state,
            exec_state=exec_state,
            sp=sp,
            wp=wp,
            weather_dict=self.weather,
            npc_list=self.npc_list
        )
        if ret == -1:
            return -1
        if not self.conf.function.startswith("eval"):
            if ret == 128:
                return 128
        log_filename = self.dump_states(self.state, log_type="queue")
        self.log_filename = log_filename
        error = self.check_error(self.state)
        # # reload scenario state
        # self.state = ScenarioState()
        self.save_video(error, log_filename)
        # if self.state.trace_graph_important != []:
        #     self.save_trace(self.state.trace_graph_important, log_filename)
        if error:
            self.found_error = True
            return 1

        # if state.num_frames <= c.FRAME_RATE:
        #     # Trap for an unlikely situation where test target didn't load
        #     # but we somehow got here.
        #     print("[-] Not enough data for scoring ({} frames)".format(
        #         state.num_frames))
        #     return 1

        # with open(os.path.join(self.conf.score_dir, log_filename), "w") as fp:
        #     json.dump(state.deductions, fp)

    def reload_state(self):
        self.state.crashed = False
        self.state.collision_to = None
        self.state.stuck = False
        self.state.stuck_duration = 0
        self.state.laneinvaded = False
        self.state.laneinvasion_event = []
        self.state.speeding = False
        self.state.speed = []
        self.state.speed_lim = []
        self.state.on_red = False
        self.state.on_red_speed = []
        self.state.red_violation = False
        self.state.other_error = False
        self.state.other_error_val = 0

    def save_video(self, error, log_filename):
        try:
            # if self.conf.agent_type == c.AUTOWARE:
            #     if error:
            #         # print("copying bag & video files")
            #         shutil.copyfile(
            #             os.path.join(self.conf.queue_dir, log_filename),
            #             os.path.join(self.conf.error_dir, log_filename)
            #         )
            #         # shutil.copyfile(
            #         #     f"/tmp/fuzzerdata/{c.USERNAME}/bagfile.lz4.bag",
            #         #     os.path.join(self.conf.rosbag_dir, log_filename.replace(".json", ".bag"))
            #         # )
            #
            #     shutil.copyfile(
            #         f"/tmp/fuzzerdata/{c.USERNAME}/front.mp4",
            #         os.path.join(self.conf.cam_dir, log_filename.replace(".json", "-front.mp4"))
            #     )
            #     shutil.copyfile(
            #         f"/tmp/fuzzerdata/{c.USERNAME}/top.mp4",
            #         os.path.join(self.conf.cam_dir, log_filename.replace(".json", "-top.mp4"))
            #     )
            # elif self.conf.agent_type == c.BEHAVIOR:
            if error:
                shutil.copyfile(
                    os.path.join(self.conf.queue_dir, log_filename),
                    os.path.join(self.conf.error_dir, log_filename)
                )
            shutil.copyfile(
                f"/tmp/fuzzerdata/{self.username}/front.mp4",
                os.path.join(
                    self.conf.cam_dir,
                    log_filename.replace(".json", "-front.mp4")
                )
            )

            shutil.copyfile(
                f"/tmp/fuzzerdata/{self.username}/top.mp4",
                os.path.join(
                    self.conf.cam_dir,
                    log_filename.replace(".json", "-top.mp4")
                )
            )
            print("save video done")
        except FileNotFoundError:
            print("FileNotFoundError")
            # os._exit(0)

    def save_trace_point(self, trace_graph_points, param, log_filename):
        output_filename = log_filename.replace(".json", ".txt")
        output_path = os.path.join(self.conf.trace_dir, output_filename)

        # Open the file for writing
        with open(output_path, 'w') as file:
            for i, trace in enumerate(trace_graph_points):
                file.write(f'Trace {i + 1}:\n')
                # Calculate the step for sampling points from each trace
                step = max(1, len(trace) // param)
                # Sample points from the trace
                sampled_points = trace[::step]
                # If the number of points is less than param, use all points
                if len(trace) < param:
                    sampled_points = trace

                # Write the coordinates of each point to the file
                for point in sampled_points:
                    file.write(f'({point[0]},{point[1]},{point[2]})\n')

    # def save_trace(self, trace_graph, log_filename):
    #     new_trace_graph = np.array([np.array([point[:2] for point in trace]) for trace in trace_graph])
    #     trace_graph_points = shift_scale_points_group(np.array(new_trace_graph), (1024, 1024))
    #     img = np.full((1024, 1024, 3), 255, dtype=np.uint8)
    #     for j, trace in enumerate(trace_graph_points):
    #         color = colors[j % len(colors)]
    #         img = draw_picture(trace, color=color, base_image=img)
    #         cv2.imwrite(os.path.join(
    #             self.conf.trace_dir,
    #             log_filename.replace(".json", ".png")
    #         ), img)
    #     self.save_trace_point(trace_graph, 15, log_filename)
    #     print("save trace done")

    def check_error(self, state):
        if self.conf.debug:
            print("----- Check for errors -----")
        error = False
        if self.conf.check_dict["crash"] and state.crashed:
            if self.conf.debug:
                print("[debug] Crashed with:", state.collision_to)
                oa = state.collision_to
                print(f"  - against {oa}")
            error = True
        if self.conf.check_dict["stuck"] and state.stuck:
            if self.conf.debug:
                print("[debug] Vehicle stuck:", state.stuck_duration)
            error = True
        if self.conf.check_dict["lane"] and state.laneinvaded:
            if self.conf.debug:
                le_list = state.laneinvasion_event
                le = le_list[0]  # only consider the very first invasion
                print("[debug] Lane invasion:", le)
                lm_list = le.crossed_lane_markings
                for lm in lm_list:
                    print("  - crossed {} lane (allows {} change)".format(
                        lm.color, lm.lane_change))
            error = True
        if self.conf.check_dict["red"] and state.red_violation:
            error = True
        if self.conf.check_dict["speed"] and state.speeding:
            if self.conf.debug:
                print("[debug] Speeding: {} km/h".format(state.speed[-1]))
            error = True
        if self.conf.check_dict["other"] and state.other_error:
            if state.other_error == "timeout":
                if self.conf.debug:
                    print("[debug] Simulation took too long")
            elif state.other_error == "goal":
                if self.conf.debug:
                    print("[debug] Goal is too far:", state.other_error_val, "m")
            error = True
        return error
