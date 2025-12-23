import argparse
import os
import select
import shutil
import socket
import subprocess
import time
import traceback
from datetime import datetime
from types import SimpleNamespace
import sys

# Ensure project root is on sys.path so we can import project modules when running from script/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Common project directories (absolute paths, independent of current working directory)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")
SEED_ARTIFACT_DIR = os.path.join(DATA_DIR, "seed-artifact")
SAVE_BASE_DIR = os.path.join(DATA_DIR, "save")

import torch
import config
import states

# Configure CARLA PythonAPI path
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
DEFAULT_SIM_PORT = 4000

import fuzzer


def run_command(command, wait=True):
    """Runs a shell command and captures its output in real-time without blocking."""
    print(f"Running command: {command}")  # Debugging output
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, bufsize=1)

    # Real-time output capturing
    if wait:
        stdout_lines = []
        stderr_lines = []
        try:
            while True:
                # Use select to avoid blocking on readline
                ready_to_read, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

                if process.stdout in ready_to_read:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        print(f"Standard Output: {stdout_line.strip()}")
                        stdout_lines.append(stdout_line)

                if process.stderr in ready_to_read:
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        print(f"Standard Error: {stderr_line.strip()}")
                        stderr_lines.append(stderr_line)

                # Check if process has finished
                if process.poll() is not None:
                    # Process has finished, make sure to flush remaining output
                    stdout_remaining = process.stdout.read()
                    stderr_remaining = process.stderr.read()
                    if stdout_remaining:
                        stdout_lines.append(stdout_remaining)
                    if stderr_remaining:
                        stderr_lines.append(stderr_remaining)
                    break

        except Exception as e:
            print(f"Error occurred: {e}")

        return ''.join(stdout_lines), ''.join(stderr_lines)

    return process


def init_environment(sim_port=DEFAULT_SIM_PORT):
    """Equivalent to init.sh functionality."""
    fuzzerdata_dir = f"/tmp/fuzzerdata/{os.getlogin()}"
    docker_name = f"carla-{os.getlogin()}"

    # Create fuzzerdata_dir if it doesn't exist
    if not os.path.exists(fuzzerdata_dir):
        os.makedirs(fuzzerdata_dir)
        print(f"Created directory {fuzzerdata_dir}")

    stop_autoware()
    # Check if Docker container is running
    docker_status, _ = run_command(f"docker inspect -f '{{{{.State.Status}}}}' {docker_name} 2>/dev/null")
    if "running" not in docker_status:
        print(f"Docker container {docker_name} is not in running state. Running stop_carla()...")
        stop_carla()

    # Check if Docker container exists
    docker_exists, _ = run_command(f"docker ps -a --filter name={docker_name} --format '{{{{.Names}}}}'")
    if not docker_exists:
        print(f"Docker container {docker_name} doesn't exist. Running run_carla()...")
        run_carla(port=sim_port)
    else:
        # Verify RPC port readiness; restart if necessary
        if not wait_for_carla_server(sim_port, timeout=30, interval=2):
            print(f"[WARNING] CARLA container {docker_name} is unresponsive on port {sim_port}, restarting...")
            stop_carla()
            run_carla(port=sim_port)

    # Remove files in fuzzerdata_dir
    for filename in os.listdir(fuzzerdata_dir):
        file_path = os.path.join(fuzzerdata_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Call save_files functionality
    save_files()

    # Remove directories under the project data directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Removed {OUTPUT_DIR} directory")

    if os.path.exists(SEED_ARTIFACT_DIR):
        shutil.rmtree(SEED_ARTIFACT_DIR)
        print(f"Removed {SEED_ARTIFACT_DIR} directory")

# Remove specific Docker containers
    containers, _ = run_command("docker ps -a --filter ancestor=carla-autoware:improved-record --format='{{.ID}}'")
    if containers:
        run_command(f"docker rm -f {containers}")
        print(f"Removed Docker containers: {containers}")


def wait_for_carla_server(port, timeout=90, interval=2):
    """Poll the CARLA RPC port until it becomes available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=interval):
                print(f"[INFO] CARLA server on port {port} is ready.")
                return True
        except (OSError, ConnectionError):
            time.sleep(interval)
    print(f"[WARNING] CARLA server on port {port} did not become ready within {timeout} seconds.")
    return False


def run_carla(port=DEFAULT_SIM_PORT):
    """Equivalent to run_carla.sh functionality."""
    # idle_gpu = 0
    carla_cmd = f"./CarlaUE4.sh -RenderOffScreen -carla-rpc-port={port} -quality-level=Epic && /bin/bash"
    docker_name = f"carla-{os.getlogin()}"

    # Prepare recorder directory volume mapping
    recorder_dir = os.path.join(OUTPUT_DIR, "recorder")
    os.makedirs(recorder_dir, exist_ok=True)
    # Set permissions to allow CARLA container to write recorder files
    # CARLA runs as user 998 (carla), so we need to ensure the directory is writable
    os.chmod(recorder_dir, 0o777)
    recorder_volume = f"-v {recorder_dir}:/home/carla/recordings:rw"

    # Run CARLA Docker
    command = f"docker run --name='carla-{os.getlogin()}' -d --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw {recorder_volume} carlasim/carla:0.9.13 {carla_cmd}"
    # command = f"docker run --name='carla-{os.getlogin()}' -d --gpus --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.13 {carla_cmd}"
    # run_command(command)
    subprocess.Popen(command, shell=True)
    if wait_for_carla_server(port, timeout=120):
        print(f"Started CARLA Docker container {docker_name}")
    else:
        print(f"[WARNING] CARLA Docker container {docker_name} may not be ready yet.")


def save_files():
    """Store all files and directories from data/output/ into a timestamped directory under data/save/."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = os.path.join(SAVE_BASE_DIR, timestamp)

    output_dir = OUTPUT_DIR

    # Check if output_dir exists
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Skipping save operation.")
        return

    # Create the save directory
    os.makedirs(save_dir, exist_ok=True)

    # Iterate over all items (files and directories) in output_dir
    # This includes recorder directory which contains CARLA recorder log files
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        target_path = os.path.join(save_dir, item)

        if os.path.isdir(item_path):
            # Use copytree for directories to handle nested structures
            if not os.listdir(item_path):  # Check if directory is empty
                print(f"Directory {item_path} is empty. Skipping...")
                continue

            shutil.copytree(item_path, target_path)
            print(f"Copied directory {item_path} to {target_path}")

        elif os.path.isfile(item_path):
            # Handle files directly in the output_dir root
            shutil.copy(item_path, save_dir)
            print(f"Copied file {item_path} to {save_dir}")

        else:
            print(f"{item_path} is neither a file nor a directory. Skipping...")

    # If save_dir is empty after copying, remove it
    if not os.listdir(save_dir):
        shutil.rmtree(save_dir)
        print(f"No files were saved. Removed empty save directory {save_dir}")


def stop_carla():
    """Equivalent to stop_carla.sh functionality."""
    docker_name = f"carla-{os.getlogin()}"
    run_command(f"docker rm -f {docker_name}")
    print(f"Stopped and removed Docker container {docker_name}")


def stop_autoware():
    """Equivalent to stop_autoware.sh functionality."""
    docker_name = f"autoware-{os.getlogin()}"
    run_command(f"docker rm -f {docker_name}")
    print(f"Stopped and removed Docker container {docker_name}")


def close_processes():
    """Equivalent to close.sh functionality."""
    process_grep = "/usr/bin/python2 /opt/ros/melodic/bin/rostopic echo /decision_maker/state"
    pids_output, _ = run_command(f"ps -u {os.getlogin()} -o pid,command | grep '{process_grep}' | awk '{{print $1}}'")
    pids = pids_output.split()

    for pid in pids:
        run_command(f"kill {pid}")
        print(f"Killed process {pid}")


def run_test(sim_port, target, density, town, duration, max_failures=3, out_dir=None, max_scenarios=None):
    """Directly call the main function from fuzzer.py to run the simulation test."""

    # Get default argument values from argparse
    argument_parser = fuzzer.set_args()
    default_args = vars(argument_parser.parse_args([]))  # Get all default values as a dictionary

    # Update the default arguments with the specific values we want to pass
    custom_args = {
        "sim_port": sim_port,
        "target": target,
        "density": density,
        "town": town,
        "timeout": duration
    }
    
    # Set output directory if provided (for experiment manager integration)
    if out_dir is not None:
        custom_args["out_dir"] = out_dir
        custom_args["allow_out_dir_exists"] = True  # Allow using existing experiment directory
    
    # Set max_scenarios if provided (for precise scenario count control)
    if max_scenarios is not None and max_scenarios > 0:
        custom_args["max_scenarios"] = max_scenarios

    # Merge default arguments with custom arguments
    default_args.update(custom_args)

    # Configure this run as a non-GPT baseline with metrics enabled:
    # - Disable GPT-based evaluation/logging
    # - Ensure multi-dimensional metrics (PC/PEC/TCD/BCM) are computed
    default_args["disable_gpt"] = True
    default_args["enable_rag_metrics"] = True

    # Convert to SimpleNamespace for compatibility with the fuzzer's main function
    args = SimpleNamespace(**default_args)

    start_time = time.time()

    failure_count = 0

    while True:
        # Initialize environment
        init_environment(sim_port=sim_port)

        current_time = time.time()
        total_duration = current_time - start_time
        if total_duration >= duration:
            print(f"Total duration exceeded {duration} seconds. Exiting...")
            break

        # Save the current directory and switch to the project root directory
        current_dir = os.getcwd()
        os.chdir(PROJECT_ROOT)  # Switch to project root so fuzzer runs from a stable base path

        try:
            # Directly call the main function from fuzzer.py
            fuzzer.main(args)
            failure_count = 0
        except KeyboardInterrupt:
            print(f"\nKeyboardInterrupt: Experiment interrupted by user")
            return
        except SystemExit as e:
            print(f"SystemExit: {e}")
            raise  # Re-raise SystemExit
        except TimeoutError as e:
            print(f"TimeoutError: {e}")
            raise  # Re-raise TimeoutError
        except Exception as e:
            print(f"Unexpected exception: {e}")
            traceback.print_exc()
            failure_count += 1
            if failure_count >= max_failures:
                print(f"[ERROR] Reached {failure_count} consecutive failures, aborting run.")
                raise  # Re-raise to ensure error is visible
            else:
                print(f"[WARNING] Failure #{failure_count}, retrying after short delay...")
                time.sleep(2)

        os.chdir(current_dir)
        # Check if the Docker container is still running
        docker_name = f"carla-{os.getlogin()}"
        docker_status, _ = run_command(f"docker inspect -f '{{{{.State.Status}}}}' {docker_name}")
        if "running" not in docker_status:
            print(f"{docker_name} is not in 'running' state. Restarting...")

        time.sleep(1)


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="ScenarioFuzz test runner (behavior/autoware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", choices=["behavior", "autoware"],
                        help="Target ADS to test")
    parser.add_argument("density", type=str,
                        help="Traffic density (string to match original scripts, e.g., '0.4')")
    parser.add_argument("town", type=int,
                        help="CARLA town id (e.g., 3 for Town03)")
    parser.add_argument("duration", type=int,
                        help="Total test duration in seconds")
    parser.add_argument("--sim-port", type=int, default=DEFAULT_SIM_PORT,
                        help="CARLA RPC port")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="Abort after this many consecutive failures")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for scenarios (default: ./data/output)")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Maximum number of scenarios to generate (0 = unlimited, default: None)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run_test(
        sim_port=args.sim_port,
        target=args.target,
        density=args.density,
        town=str(args.town),
        duration=args.duration,
        max_failures=args.max_failures,
        out_dir=getattr(args, 'out_dir', None),
        max_scenarios=getattr(args, 'max_scenarios', None),
    )
