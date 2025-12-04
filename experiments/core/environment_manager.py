"""
Environment Manager Module
Integrates script/init.sh functionality for environment management
"""

import os
import subprocess
import shutil
import socket
import time
from pathlib import Path
from datetime import datetime


def run_init_script(script_dir: Path, project_root: Path):
    """
    Run script/init.sh to manage environment, OR use Python equivalent
    
    Args:
        script_dir: Path to script directory
        project_root: Path to project root
    """
    # Use Python implementation instead of calling shell script
    # This gives us better control and error handling
    
    # 1. Create/check fuzzerdata directory
    # Security: Validate username to prevent path traversal
    username = os.getlogin()
    # Sanitize username to prevent path traversal attacks
    if not username or '/' in username or '..' in username:
        raise ValueError(f"Invalid username for fuzzerdata directory: {username}")
    
    fuzzerdata_dir = Path(f"/tmp/fuzzerdata/{username}")
    # Ensure path is within /tmp/fuzzerdata (security check)
    fuzzerdata_base = Path("/tmp/fuzzerdata")
    try:
        fuzzerdata_dir.resolve().relative_to(fuzzerdata_base.resolve())
    except ValueError:
        raise ValueError(f"Invalid fuzzerdata path: {fuzzerdata_dir}")
    
    if not fuzzerdata_dir.exists():
        fuzzerdata_dir.mkdir(parents=True, mode=0o755)
        print(f"[INFO] Created directory {fuzzerdata_dir}")
    
    # 2. Stop autoware
    stop_autoware()
    
    # 3. Check and manage CARLA container
    # Security: Use validated username from above
    docker_name = f"carla-{username}"
    docker_status_result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", docker_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    if docker_status_result.returncode != 0 or docker_status_result.stdout.strip() != "running":
        # Container not running, stop it first
        subprocess.run(
            ["docker", "rm", "-f", docker_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    # Check if container exists
    docker_exists_result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={docker_name}", "--format", "{{.Names}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    if not docker_exists_result.stdout.strip():
        # Container doesn't exist, start it
        run_carla_script = script_dir / "run_carla.sh"
        if run_carla_script.exists():
            result = subprocess.run(
                ["bash", str(run_carla_script)],
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise RuntimeError(f"Failed to start CARLA container: {error_msg}")
            print(f"[INFO] Started CARLA container {docker_name}")
        else:
            raise FileNotFoundError(f"run_carla.sh not found: {run_carla_script}. Cannot start CARLA container.")
    
    # 4. Clean fuzzerdata directory
    if fuzzerdata_dir.exists():
        for file in fuzzerdata_dir.iterdir():
            if file.is_file():
                file.unlink()
        print(f"[INFO] Cleaned {fuzzerdata_dir}")
    
    # 5. Save files (equivalent to savefile.sh)
    save_files(project_root)
    
    # 6. Remove data/output and data/seed-artifact
    # NOTE: This only removes the default data/output directory, NOT experiment-specific output directories
    # Experiment outputs are stored in experiment_results/ and should NOT be cleaned here
    output_dir = project_root / "data" / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[INFO] Removed default data/output directory: {output_dir}")
        print(f"[INFO] Note: Experiment outputs in experiment_results/ are preserved")
    
    seed_artifact_dir = project_root / "data" / "seed-artifact"
    if seed_artifact_dir.exists():
        shutil.rmtree(seed_artifact_dir)
        print(f"[INFO] Removed {seed_artifact_dir}")
    
    # 7. Remove autoware containers
    cleanup_environment(project_root)
    
    # 8. Close ROS processes
    close_ros_processes()
    
    return True


def manage_carla_container(script_dir: Path, action: str = "check"):
    """
    Manage CARLA docker container using script/run_carla.sh or stop_carla.sh
    
    Args:
        script_dir: Path to script directory
        action: "start", "stop", or "check"
    """
    docker_name = f"carla-{os.getlogin()}"
    
    if action == "check":
        # Check if container is running
        # Python 3.6 compatibility: use stdout/stderr instead of capture_output
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", docker_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if result.returncode == 0:
            status = result.stdout.strip()
            return status == "running"
        return False
    
    elif action == "start":
        run_carla_script = script_dir / "run_carla.sh"
        if not run_carla_script.exists():
            raise FileNotFoundError(f"run_carla.sh not found: {run_carla_script}")
        
        result = subprocess.run(
            ["bash", str(run_carla_script)],
            cwd=str(script_dir)
        )
        return result.returncode == 0
    
    elif action == "stop":
        stop_carla_script = script_dir / "stop_carla.sh"
        if not stop_carla_script.exists():
            raise FileNotFoundError(f"stop_carla.sh not found: {stop_carla_script}")
        
        result = subprocess.run(
            ["bash", str(stop_carla_script)],
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.returncode == 0
    
    else:
        raise ValueError(f"Unknown action: {action}")


def cleanup_environment(project_root: Path):
    """
    Clean up environment (equivalent to init.sh functionality)
    
    Args:
        project_root: Path to project root
    """
    # Clean fuzzerdata directory
    fuzzerdata_dir = Path(f"/tmp/fuzzerdata/{os.getlogin()}")
    if fuzzerdata_dir.exists():
        for file in fuzzerdata_dir.iterdir():
            if file.is_file():
                file.unlink()
    
    # Remove data/output and data/seed-artifact
    output_dir = project_root / "data" / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    seed_artifact_dir = project_root / "data" / "seed-artifact"
    if seed_artifact_dir.exists():
        shutil.rmtree(seed_artifact_dir)
    
    # Remove autoware containers
    # Python 3.6 compatibility: use stdout/stderr instead of capture_output
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "ancestor=carla-autoware:improved-record", "--format", "{{.ID}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if result.returncode == 0 and result.stdout.strip():
        container_ids = result.stdout.strip().split('\n')
        for container_id in container_ids:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )


def ensure_carla_running(script_dir: Path, project_root: Path):
    """
    Ensure CARLA container is running (call init.sh if needed)
    
    Args:
        script_dir: Path to script directory
        project_root: Path to project root
    
    Raises:
        RuntimeError: If CARLA container cannot be started
        FileNotFoundError: If run_carla.sh script is not found
    """
    docker_name = f"carla-{os.getlogin()}"
    
    # Check if container exists and is running
    # Python 3.6 compatibility: use stdout/stderr instead of capture_output
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", docker_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    if result.returncode != 0:
        # Container doesn't exist, run init.sh
        print(f"[INFO] CARLA container {docker_name} doesn't exist, running init...")
        run_init_script(script_dir, project_root)
        # Verify that container is now running
        verify_result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", docker_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if verify_result.returncode != 0 or verify_result.stdout.strip() != "running":
            raise RuntimeError(f"Failed to start CARLA container {docker_name} after init")
    elif result.stdout.strip() != "running":
        # Container exists but not running, run init.sh
        print(f"[INFO] CARLA container {docker_name} is not running, running init...")
        run_init_script(script_dir, project_root)
        # Verify that container is now running
        verify_result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", docker_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if verify_result.returncode != 0 or verify_result.stdout.strip() != "running":
            raise RuntimeError(f"Failed to start CARLA container {docker_name} after init")
    else:
        print(f"[INFO] CARLA container {docker_name} is running")


def save_files(project_root: Path):
    """
    Save files from data/output/ to data/save/{timestamp}/
    Equivalent to script/savefile.sh functionality
    
    Args:
        project_root: Path to project root
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = project_root / "data" / "save" / timestamp
    output_dir = project_root / "data" / "output"
    
    # Check if output_dir exists
    if not output_dir.exists():
        print(f"[INFO] Output directory {output_dir} does not exist. Skipping save operation.")
        return
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Directories to save (matching savefile.sh)
    dirs_to_save = {
        "camera": output_dir / "camera",
        "errors": output_dir / "errors",
        "time_record": output_dir / "time_record"
    }
    
    saved_anything = False
    
    for dir_name, source_dir in dirs_to_save.items():
        if source_dir.exists() and source_dir.is_dir():
            # Check if directory is not empty
            if any(source_dir.iterdir()):
                target_dir = save_dir / dir_name
                shutil.copytree(source_dir, target_dir)
                print(f"[INFO] Copied {source_dir} to {target_dir}")
                saved_anything = True
            else:
                print(f"[INFO] Directory {source_dir} is empty. Skipping...")
        else:
            print(f"[INFO] Directory {source_dir} does not exist. Skipping...")
    
    # Also copy any files directly in output_dir
    for item in output_dir.iterdir():
        if item.is_file():
            shutil.copy(item, save_dir)
            print(f"[INFO] Copied file {item} to {save_dir}")
            saved_anything = True
    
    # If nothing was saved, remove empty save directory
    if not saved_anything:
        shutil.rmtree(save_dir)
        print(f"[INFO] No files were saved. Removed empty save directory {save_dir}")
    else:
        print(f"[INFO] Saving done: {save_dir}")


def close_ros_processes():
    """
    Close ROS-related processes (equivalent to script/close.sh)
    Kills processes matching: /usr/bin/python2 /opt/ros/melodic/bin/rostopic echo /decision_maker/state
    """
    try:
        username = os.getlogin()
        # Find processes matching the pattern
        result = subprocess.run(
            ["ps", "-u", username, "-o", "pid,command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode == 0:
            pattern = "/usr/bin/python2 /opt/ros/melodic/bin/rostopic echo /decision_maker/state"
            for line in result.stdout.split('\n'):
                if pattern in line:
                    # Extract PID (first field)
                    parts = line.split()
                    if parts:
                        pid = parts[0]
                        try:
                            subprocess.run(["kill", pid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            print(f"[INFO] Killed ROS process {pid}")
                        except Exception as e:
                            print(f"[WARNING] Failed to kill process {pid}: {e}")
    except Exception as e:
        print(f"[WARNING] Failed to close ROS processes: {e}")


def stop_autoware():
    """
    Stop autoware container (equivalent to script/test.py stop_autoware)
    """
    docker_name = f"autoware-{os.getlogin()}"
    result = subprocess.run(
        ["docker", "rm", "-f", docker_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    if result.returncode == 0:
        print(f"[INFO] Stopped and removed Docker container {docker_name}")
    else:
        # Container might not exist, which is OK
        pass


def wait_for_port(host: str, port: int, timeout: int = 300, check_interval: float = 2.0):
    """
    Wait for a port to become available
    
    Args:
        host: Host address
        port: Port number
        timeout: Maximum time to wait in seconds (default 300 = 5 minutes)
        check_interval: Time between checks in seconds (default 2.0)
    
    Returns:
        True if port becomes available, False if timeout
    """
    start_time = time.time()
    print(f"[INFO] Waiting for port {host}:{port} to become available...")
    
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"[INFO] Port {host}:{port} is now available")
                return True
        except Exception as e:
            pass
        
        time.sleep(check_interval)
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0:  # Print every 10 seconds
            print(f"[INFO] Still waiting for port {host}:{port}... ({elapsed}s elapsed)")
    
    print(f"[WARNING] Timeout waiting for port {host}:{port} after {timeout} seconds")
    return False


def restart_carla_container(script_dir: Path, project_root: Path, port: int = 4000):
    """
    Restart CARLA docker container and wait for port to be available
    
    Args:
        script_dir: Path to script directory
        project_root: Path to project root
        port: Port number to wait for (default 4000)
    
    Returns:
        True if successfully restarted and port is available
    """
    docker_name = f"carla-{os.getlogin()}"
    
    print(f"[INFO] Restarting CARLA container {docker_name}...")
    
    # Stop and remove existing container
    subprocess.run(
        ["docker", "rm", "-f", docker_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for cleanup
    time.sleep(2)
    
    # Start new container
    run_carla_script = script_dir / "run_carla.sh"
    if not run_carla_script.exists():
        print(f"[ERROR] run_carla.sh not found: {run_carla_script}")
        return False
    
    result = subprocess.run(
        ["bash", str(run_carla_script)],
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if result.returncode != 0:
        print(f"[ERROR] Failed to start CARLA container")
        return False
    
    print(f"[INFO] CARLA container {docker_name} started, waiting for port {port}...")
    
    # Wait for port to become available
    # Use localhost since docker uses --net=host
    port_available = wait_for_port("localhost", port, timeout=300)
    
    if port_available:
        print(f"[INFO] CARLA container {docker_name} is ready on port {port}")
        return True
    else:
        print(f"[WARNING] CARLA container started but port {port} not available yet")
        return False

