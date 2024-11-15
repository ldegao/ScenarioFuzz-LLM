"""Define constants used across DriveFuzz modules."""
import os

USERNAME = os.environ['USER']
# ERROR MESSAGES
MSG_BAD_WEATHER_ARG = "Weather argument should be in range [0.0:100.0]."
MSG_BAD_SUN_ARG = "Sun args require --angle [0:360], --altitude [-90:90]"
MSG_EMPTY_FRICTION_ARG = "--friction should be followed by level, size " \
                         "and location coordinate."
MSG_BAD_FRICTION_ARG = "Number of friction args has to be 7."
MSG_BAD_FRICTION_LEVEL = "Friction level should be in range [0.0-1.0]."
MSG_EMPTY_ACTOR_ARG = "--npc should be followed by npc_type, and " \
                      "corresponding values."
MSG_BAD_ACTOR_TYPE = "--npc expects 0, 1, or 2 as npc_type."
MSG_BAD_ACTOR_ATTR = "Vehicle or walker requires 12 args: " \
                     "npc_type, nav_type, sp_x, sp_y, sp_z, " \
                     "sp_pitch, sp_yaw, sp_roll, dp_x, dp_y, dp_z, " \
                     "speed."
MSG_BAD_SPAWN_ARG = "--spawn expects six args: x, y, z, pitch, yaw, roll"
MSG_BAD_DEST_ARG = "--dest expects three args: x, y, z"
# scope

MIN_DIST = 50
MAX_DIST = 150

# Agent Types
BEHAVIOR = 1
AUTOWARE = 2
AUTOWARE_UNIVERSE = 3
REPLAY = 4
OTHER = 9

TARGET_SPEED = 60
MAX_THROTTLE = 1  # 0.75
MAX_BRAKE = 1  # 0.3
MAX_STEERING = 0.8

# Static configurations
MAX_DIST_FROM_PLAYER = 40
MIN_DIST_FROM_PLAYER = 5
FRAME_RATE = 25
INIT_SKIP_SECONDS = 2
WAIT_AUTOWARE_NUM_NODES = 64

# Actors
NULL = -1  # special type for traction testing
VEHICLE = 0
PEDESTRIAN = 1
ACTOR_LIST = [VEHICLE, PEDESTRIAN]
ACTOR_NAMES = ["vehicle", "walker"]

# NPC Navigation Type
BEHAVIOR_AGENT = 1
IMMOBILE = 2
STOP = 3
EGO = 4
NAVTYPE_LIST = [BEHAVIOR_AGENT, IMMOBILE, STOP]
NAVTYPE_NAMES = ["behavior_agent", "immobile", "stop"]

# NPC Attributes
VEHICLE_MAX_SPEED = 30  # multiplied with forward vector
WALKER_MAX_SPEED = 10  # m/s
VIDEO_TIME = 15

# # Puddle Attributes
# PROB_PUDDLE = 25 # probability of adding a new puddle
# PUDDLE_MAX_SIZE = 500 # centimeters

# Maneuver Attributes
FRAMES_PER_TIMESTEP = 100  # Five seconds (tentative)

# Number of waypoints per town
NUM_WAYPOINTS = {
    "Town01": 255,
    "Town02": 101,
    "Town03": 265,
    "Town04": 372,
    "Town05": 302,
    "Town06": 436,
}

# Camera View Setting
ONROOF = 0
BIRDSEYE = 1

# Driving Quality
HARD_ACC_THRES = 21.2  # km/h per second
HARD_BRAKE_THRES = -21.2  # km/h per second

# Filter config
CUTOFF_FREQ_LIGHT = 3.5
CUTOFF_FREQ_HEAVY = 0.5

# Mutation targets
WEATHER = 0
ACTOR = 1
# PUDDLE = 2
MUTATION_TARGET = [WEATHER, ACTOR]

# relative position
FRONT = 0
BACK = 1
# relative lane
MIDDLE = 0
LEFT = 1
RIGHT = 2

# Input mutation strategies
ALL = 0
CONGESTION = 1
ENTROPY = 2
INSTABILITY = 3
TRAJECTORY = 4

# Misc
DEVNULL = "2> /dev/null"
