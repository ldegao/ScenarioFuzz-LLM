import copy
import pdb
import random

import config
import constants as c
import math
from shapely.geometry import Polygon

config.set_carla_api_path()
import carla

import utils


class NPC:
    npc_id: int
    npc_type: int
    npc_bp_id = str
    spawn_point = carla.Waypoint
    speed: int
    spawn_stuck_frame: int
    instance: carla.Actor
    ego_loc: carla.Location
    fresh: bool
    death_time: int

    sensor_collision: carla.Actor
    sensor_lane_invasion: carla.Actor

    def __init__(self, npc_type, spawn_point, npc_id=0, speed=0, ego_loc=None,
                 spawn_stuck_frame=0, npc_bp_id=None):
        self.npc_type = npc_type
        self.spawn_point = spawn_point
        self.npc_id = npc_id
        self.speed = speed
        self.ego_loc = ego_loc
        self.spawn_stuck_frame = spawn_stuck_frame
        self.npc_bp_id = npc_bp_id
        self.fresh = True
        self.instance = None
        self.instance_id = -1
        self.sensor_collision = None
        self.sensor_lane_invasion = None
        self.stuck_duration = 0
        self.death_time = -1

    def __deepcopy__(self, memo):
        npc_copy = NPC(
            copy.deepcopy(self.npc_type, memo),
            copy.deepcopy(self.spawn_point, memo),
            copy.deepcopy(self.npc_id, memo),
            copy.deepcopy(self.speed, memo),
            copy.deepcopy(self.ego_loc, memo),
            copy.deepcopy(self.spawn_stuck_frame, memo),
            copy.deepcopy(self.npc_bp_id, memo),
        )
        return npc_copy

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.ego_loc:
            state['ego_loc'] = utils.carla_location_pickle(self.ego_loc)
        if self.spawn_point:
            # Check spawn_point type - handle both Waypoint and Transform
            try:
                transform_to_save = None
                spawn_point_type = 'unknown'
                
                # Strategy 1: Try as Transform (most common case)
                try:
                    if hasattr(self.spawn_point, 'location') and hasattr(self.spawn_point, 'rotation'):
                        loc = getattr(self.spawn_point, 'location')
                        rot = getattr(self.spawn_point, 'rotation')
                        # Check if these are actual Location/Rotation objects
                        if hasattr(loc, 'x') and hasattr(rot, 'pitch'):
                            # It's a Transform object
                            transform_to_save = self.spawn_point
                            spawn_point_type = 'transform'
                        else:
                            raise AttributeError("Not a Transform")
                    else:
                        raise AttributeError("Not a Transform")
                except (AttributeError, TypeError):
                    # Strategy 2: Try as Waypoint - extract transform
                    try:
                        if hasattr(self.spawn_point, 'transform'):
                            transform_attr = getattr(self.spawn_point, 'transform')
                            
                            # Check if transform is callable (method) or a property
                            if callable(transform_attr):
                                # transform is a method, call it
                                transform_obj = transform_attr()
                                # Verify it's a Transform object
                                if hasattr(transform_obj, 'location') and hasattr(transform_obj, 'rotation'):
                                    transform_to_save = transform_obj
                                    spawn_point_type = 'waypoint'
                                else:
                                    raise ValueError("transform() did not return a Transform object")
                            elif hasattr(transform_attr, 'location') and hasattr(transform_attr, 'rotation'):
                                # transform is a property returning Transform
                                transform_to_save = transform_attr
                                spawn_point_type = 'waypoint'
                            else:
                                raise AttributeError("transform is not accessible")
                        else:
                            raise AttributeError("No transform attribute")
                    except (AttributeError, TypeError, ValueError) as waypoint_err:
                        # Strategy 3: Try to extract location/rotation from waypoint directly
                        try:
                            if hasattr(self.spawn_point, 'location') and hasattr(self.spawn_point, 'rotation'):
                                waypoint_loc = getattr(self.spawn_point, 'location')
                                waypoint_rot = getattr(self.spawn_point, 'rotation')
                                if hasattr(waypoint_loc, 'x') and hasattr(waypoint_rot, 'pitch'):
                                    # Create Transform from waypoint's location/rotation
                                    transform_to_save = carla.Transform(waypoint_loc, waypoint_rot)
                                    spawn_point_type = 'waypoint'
                                else:
                                    raise ValueError("Waypoint location/rotation are not valid")
                            else:
                                raise AttributeError("No location/rotation attributes")
                        except (AttributeError, TypeError, ValueError) as direct_err:
                            # All strategies failed - log detailed error but don't set to None yet
                            print(f"[WARNING] Cannot extract transform from spawn_point. "
                                  f"Type: {type(self.spawn_point)}, "
                                  f"Attributes: {dir(self.spawn_point)[:10]}, "
                                  f"Errors: Transform={waypoint_err}, Direct={direct_err}")
                            # Try one last time: if spawn_point has any location-like attributes
                            try:
                                # Try to get any location information
                                if hasattr(self.spawn_point, 'location'):
                                    loc = getattr(self.spawn_point, 'location')
                                    if hasattr(loc, 'x'):
                                        # At least we have location, create a default Transform
                                        default_rot = carla.Rotation(pitch=0, yaw=0, roll=0)
                                        transform_to_save = carla.Transform(loc, default_rot)
                                        spawn_point_type = 'waypoint_fallback'
                                        print(f"[WARNING] Using fallback: extracted location only, using default rotation")
                            except Exception as fallback_err:
                                print(f"[ERROR] All spawn_point serialization strategies failed. "
                                      f"Last error: {fallback_err}. Setting spawn_point to None.")
                                transform_to_save = None
                                spawn_point_type = 'unknown'
                
                # Save the transform if we successfully extracted one
                if transform_to_save is not None:
                    try:
                        state['spawn_point'] = utils.carla_transform_pickle(transform_to_save)
                        state['spawn_point_type'] = spawn_point_type
                    except Exception as pickle_err:
                        print(f"[ERROR] Failed to pickle transform: {pickle_err}")
                        state['spawn_point'] = None
                        state['spawn_point_type'] = 'unknown'
                else:
                    state['spawn_point'] = None
                    state['spawn_point_type'] = 'unknown'
                    
            except Exception as e:
                # If any unexpected error occurs during serialization
                print(f"[ERROR] Unexpected error serializing spawn_point: {e}")
                import traceback
                traceback.print_exc()
                state['spawn_point'] = None
                state['spawn_point_type'] = 'unknown'
        state['instance'] = None
        state['sensor_collision'] = None
        state['sensor_lane_invasion'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if state.get('ego_loc'):
            self.ego_loc = utils.carla_location_unpickle(state['ego_loc'])
        if state.get('spawn_point'):
            # Restore spawn_point based on saved type
            # Note: We always save as Transform, so restore as Transform
            # The type annotation says Waypoint, but we handle both
            try:
                self.spawn_point = utils.carla_transform_unpickle(state['spawn_point'])
            except Exception as e:
                # If unpickling fails, raise the exception instead of silently setting to None
                # This ensures we know about serialization problems immediately
                npc_id = state.get('npc_id', 'unknown')
                raise RuntimeError(f"Failed to unpickle spawn_point for NPC {npc_id}: {e}. "
                                 f"This indicates a serialization/deserialization error that needs to be fixed.")
            # Remove spawn_point_type from state if present (it's metadata only)
            if 'spawn_point_type' in self.__dict__:
                del self.__dict__['spawn_point_type']
        else:
            # spawn_point was None in saved state
            # This is OK if instance is set later (e.g., during scenario execution)
            # Only log if we're sure it's a problem (which we can't know at deserialization time)
            # Don't print warning here - spawn_point may be set later or instance may be used instead
            self.spawn_point = None

    def safe_check(self, another_npc, width=1.5, adjust=2):
        """
        :param another_npc: another npc
        :param width: the width of the vehicle
        :param adjust: to adjust of the HARD_ACC_THRES
        :return: True if safe, False if not safe

        check if two vehicles are safe to each other, if not, return False
        """
        # calculate points of two vehicles safe rectangle
        points_list1 = calculate_safe_rectangle(self.get_position_now(), self.get_speed_now(),
                                                c.HARD_ACC_THRES / 3.6 / adjust,
                                                width)
        points_list2 = calculate_safe_rectangle(another_npc.get_position_now(), another_npc.get_speed_now(),
                                                c.HARD_ACC_THRES / 3.6 / adjust, width)
        self_rect = Polygon(points_list1)
        another_rect = Polygon(points_list2)
        if self_rect.intersects(another_rect):
            # print("not safe")
            return False
        else:
            return True

    def get_position_now(self):
        if self.instance is None:
            if self.spawn_point is None:
                raise ValueError(f"NPC {self.npc_id}: spawn_point is None and instance is None, cannot get position")
            position = self.spawn_point.location
        else:
            position = self.instance.get_transform().location
        return position

    def get_speed_now(self):
        if self.instance is None:
            if self.spawn_point is None:
                raise ValueError(f"NPC {self.npc_id}: spawn_point is None and instance is None, cannot get speed")
            roll_degrees = self.spawn_point.rotation.roll
            roll_rad = math.radians(roll_degrees)
            speed_x = self.speed * math.cos(roll_rad)
            speed_y = self.speed * math.sin(roll_rad)
            speed = carla.Vector3D(speed_x, speed_y, 0)
        else:
            speed = self.instance.get_velocity()
        return speed

    def set_instance(self, npc_vehicle):
        self.instance = npc_vehicle
        self.instance_id = npc_vehicle.id

    def get_waypoint(self, town_map):
        if self.instance is None:
            if self.spawn_point is None:
                raise ValueError(f"NPC {self.npc_id}: spawn_point is None and instance is None, cannot get waypoint")
            location = self.spawn_point.location
        else:
            location = self.instance.get_transform().location
        waypoint = town_map.get_waypoint(location, project_to_road=True,
                                         lane_type=carla.libcarla.LaneType.Driving)
        return waypoint

    def get_lane_width(self, town_map):
        return self.get_waypoint(town_map).lane_width

    def attach_collision(self, world, sensors, state):
        # Attach collision detector
        blueprint_library = world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')
        sensor_collision = world.spawn_actor(collision_bp, carla.Transform(),
                                             attach_to=self.instance)
        sensor_collision.listen(lambda event: utils._on_collision(event, state))
        sensors.append(sensor_collision)
        self.sensor_collision = sensor_collision

    def attach_lane_invasion(self, world, sensors, state):
        # Attach lane invasion detector
        blueprint_library = world.get_blueprint_library()
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        sensor_lane_invasion = world.spawn_actor(lane_invasion_bp, carla.Transform(),
                                                 attach_to=self.instance)
        sensor_lane_invasion.listen(lambda event: utils._on_invasion(event, state))
        sensors.append(sensor_lane_invasion)
        self.sensor_lane_invasion = sensor_lane_invasion

    @classmethod
    def get_npc_by_one(cls, npc, town_map, npc_id):
        # split a vehicle into two similar vehicles
        # return the new vehicle
        while True:
            # Handle case where spawn_point might be None
            if npc.spawn_point is None:
                # Try to get position from instance or raise error
                npc_loc = npc.get_position()
            else:
                npc_loc = npc.spawn_point.location
            x = 0
            y = 0
            while -2 <= x <= 2:
                x = random.uniform(-5, 5)
            while -2 <= y <= 2:
                y = random.uniform(-5, 5)
            new_speed = npc.speed + random.uniform(-5, 5)
            location = carla.Location(x=npc_loc.x + x, y=npc_loc.y + y, z=npc_loc.z)
            waypoint = town_map.get_waypoint(location, project_to_road=True,
                                             lane_type=carla.libcarla.LaneType.Driving)
            new_vehicle = NPC(npc.npc_type, waypoint.transform, npc_id,
                              new_speed,
                              npc.ego_loc,
                              spawn_stuck_frame=npc.spawn_stuck_frame,npc_bp_id=npc.npc_bp_id)
            new_vehicle.fresh = True
            if new_vehicle.safe_check(npc):
                print("split:", npc.npc_id, "to", npc.npc_id, npc_id)
                return new_vehicle

    def npc_cross(self, adc2):
        pass


class Pedestrian(NPC):
    def __init__(self, npc_id, spawn_point, speed, ego_loc, spawn_stuck_frame):
        super().__init__(npc_type=c.PEDESTRIAN, spawn_point=spawn_point,
                         npc_id=npc_id, speed=speed, ego_loc=ego_loc,
                         spawn_stuck_frame=spawn_stuck_frame)


class Vehicle(NPC):
    def __init__(self, npc_id, spawn_point, speed, ego_loc, spawn_stuck_frame):
        super().__init__(npc_type=c.VEHICLE, spawn_point=spawn_point,
                         npc_id=npc_id, speed=speed, ego_loc=ego_loc,
                         spawn_stuck_frame=spawn_stuck_frame)


def calculate_safe_rectangle(position, speed, acceleration, lane_width):
    """
    :param position: the position of the vehicle,
    :param speed: the speed of the vehicle,
    :param acceleration: the acceleration of the vehicle,
    :param lane_width: the width of the lane,
    :return: the four points of the rectangle

    calculate the safe rectangle points of vehicle in the next time step
    """
    t = math.sqrt(speed.x ** 2 + speed.y ** 2) / acceleration
    rect_length = acceleration * (t ** 2) / 2
    # add car length
    rect_length = rect_length + 10
    rect_width = 2 * lane_width
    rect_direction = math.atan2(speed.y, speed.x)
    rect_half_length = rect_length / 2
    rect_center = (position.x + speed.x * t / 2, position.y + speed.y * t / 2)
    rect_points = calculate_rectangle_points(rect_center, rect_half_length, rect_width, rect_direction)
    return rect_points


def calculate_rectangle_points(center, half_length, width, direction):
    """
    :param center: the center of the rectangle
    :param half_length: half of the length of the rectangle
    :param width: the width of the rectangle
    :param direction: the direction of the rectangle
    :return: the four points of the rectangle
    """
    dx = math.cos(direction) * half_length
    dy = math.sin(direction) * half_length
    point1 = (center[0] + dx - width / 2 * math.sin(direction),
              center[1] + dy + width / 2 * math.cos(direction))
    point2 = (center[0] + dx + width / 2 * math.sin(direction),
              center[1] + dy - width / 2 * math.cos(direction))
    point3 = (center[0] - dx + width / 2 * math.sin(direction),
              center[1] - dy - width / 2 * math.cos(direction))
    point4 = (center[0] - dx - width / 2 * math.sin(direction),
              center[1] - dy + width / 2 * math.cos(direction))
    return [point1, point2, point3, point4]
