"""
真实集成测试：Recorder功能真实Docker测试

这些测试会真实地：
1. 启动Docker容器
2. 创建recorder文件
3. 验证文件保存和迁移

注意：这些测试需要Docker环境，如果Docker不可用，测试会被跳过
"""

import os
import shutil
import subprocess
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# 检查Docker是否可用
def check_docker_available():
    """检查Docker是否可用"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_carla_image():
    """检查CARLA镜像是否存在"""
    try:
        result = subprocess.run(
            ["docker", "images", "carlasim/carla:0.9.13", "--format", "{{.Repository}}:{{.Tag}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return "carlasim/carla:0.9.13" in result.stdout.decode()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def wait_for_port(host, port, timeout=30):
    """等待端口可用"""
    import socket
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (OSError, ConnectionError):
            time.sleep(1)
    return False

def cleanup_container(container_name):
    """清理Docker容器"""
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
    except Exception:
        pass

@pytest.fixture(scope="module")
def docker_available():
    """检查Docker是否可用"""
    if not check_docker_available():
        pytest.skip("Docker不可用，跳过真实Docker测试")
    if not check_carla_image():
        pytest.skip("CARLA镜像不存在，跳过真实Docker测试")
    return True

@pytest.fixture
def test_container_name():
    """生成测试容器名称"""
    import getpass
    username = getpass.getuser()
    return f"carla-test-recorder-{int(time.time())}"

@pytest.fixture
def test_port_base():
    """生成测试端口号（避免冲突）"""
    import random
    # 使用5000-6000范围的端口，避免与常用端口冲突
    return random.randint(5000, 6000)

@pytest.fixture
def test_recorder_dir(tmp_path):
    """创建测试recorder目录"""
    recorder_dir = tmp_path / "recorder"
    recorder_dir.mkdir(parents=True, exist_ok=True)
    return recorder_dir

class TestRealDockerRecorder:
    """真实Docker环境下的recorder测试"""
    
    def test_run_carla_with_recorder_volume(self, docker_available, test_container_name, test_recorder_dir, test_port_base):
        """测试真实启动CARLA容器并配置recorder volume"""
        # 清理可能存在的旧容器
        cleanup_container(test_container_name)
        
        try:
            # 参考真实CARLA启动脚本的方式
            # 使用端口映射而不是--net=host，更接近实际使用场景
            # 添加recorder volume映射
            import os
            import getpass
            
            user = getpass.getuser()
            xsock = "/tmp/.X11-unix"
            xauth = os.path.expanduser(f"~{user}/.Xauthority")
            
            # 构建volume映射（参考提供的脚本格式）
            volumes = [
                f"--volume={xsock}:{xsock}:rw",
                f"--volume={test_recorder_dir}:/home/carla/recordings:rw"
            ]
            
            # 如果XAUTH文件存在，添加它
            if os.path.exists(xauth):
                volumes.append(f"--volume={xauth}:{xauth}:rw")
            
            # 使用sleep命令保持容器运行，测试volume映射
            # 实际CARLA启动需要GPU，这里只测试volume配置
            # 使用动态端口避免冲突
            port = test_port_base
            docker_cmd = [
                "docker", "run",
                "--name", test_container_name,
                "-d", "--rm",  # 自动清理
                "-p", f"{port}:4000",  # 只映射主端口
            ] + volumes + [
                "--privileged",
                "carlasim/carla:0.9.13",
                "sleep", "30"  # 保持容器运行30秒用于测试
            ]
            
            # 启动容器
            result = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            assert result.returncode == 0, f"Docker容器启动失败: {result.stderr.decode()}"
            
            # 等待容器启动
            time.sleep(2)
            
            # 检查容器是否运行
            check_result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", test_container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            assert check_result.returncode == 0, "无法检查容器状态"
            container_status = check_result.stdout.decode().strip()
            assert container_status == "running", f"容器未运行，状态: {container_status}"
            
            # 验证volume映射：在主机上创建测试文件
            test_file_content = "test recorder content"
            test_file_path = test_recorder_dir / "test_recorder.log"
            test_file_path.write_text(test_file_content)
            
            # 验证文件在容器内可见（通过exec检查）
            exec_result = subprocess.run(
                ["docker", "exec", test_container_name, "test", "-f", "/home/carla/recordings/test_recorder.log"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            assert exec_result.returncode == 0, "文件应该在容器内可见"
            
            # 验证文件在主机上存在
            assert test_file_path.exists(), "测试文件应该在主机上存在"
            assert test_file_path.read_text() == test_file_content, "文件内容应该正确"
            
            # 在容器内创建文件，验证在主机上可见
            container_file_content = "content from container"
            exec_create = subprocess.run(
                ["docker", "exec", test_container_name, "bash", "-c", 
                 f"echo '{container_file_content}' > /home/carla/recordings/from_container.log"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            
            # 如果创建失败，先设置权限再重试
            if exec_create.returncode != 0:
                # 使用root用户设置权限
                check_dir_cmd = [
                    "docker", "exec", "-u", "root", test_container_name,
                    "bash", "-c",
                    "mkdir -p /home/carla/recordings && chown -R carla:carla /home/carla/recordings && chmod 777 /home/carla/recordings"
                ]
                subprocess.run(check_dir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                
                # 重试创建文件
                exec_create = subprocess.run(
                    ["docker", "exec", test_container_name, "bash", "-c", 
                     f"echo '{container_file_content}' > /home/carla/recordings/from_container.log"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
            
            assert exec_create.returncode == 0, f"应该在容器内创建文件: {exec_create.stderr.decode()}"
            
            # 验证文件在主机上存在
            host_file = test_recorder_dir / "from_container.log"
            assert host_file.exists(), "容器内创建的文件应该在主机上可见"
            assert container_file_content in host_file.read_text(), "文件内容应该正确"
            
        finally:
            # 清理容器
            cleanup_container(test_container_name)
    
    def test_recorder_file_creation_in_container(self, docker_available, test_container_name, test_recorder_dir, test_port_base):
        """测试在容器内创建recorder文件"""
        cleanup_container(test_container_name)
        
        try:
            # 参考真实CARLA启动方式
            import os
            import getpass
            
            user = getpass.getuser()
            xsock = "/tmp/.X11-unix"
            xauth = os.path.expanduser(f"~{user}/.Xauthority")
            
            volumes = [
                f"--volume={xsock}:{xsock}:rw",
                f"--volume={test_recorder_dir}:/home/carla/recordings:rw"
            ]
            
            if os.path.exists(xauth):
                volumes.append(f"--volume={xauth}:{xauth}:rw")
            
            port = test_port_base + 1
            docker_cmd = [
                "docker", "run",
                "--name", test_container_name,
                "-d", "--rm",
                "-p", f"{port}:4000",
            ] + volumes + [
                "--privileged",
                "carlasim/carla:0.9.13",
                "sleep", "30"
            ]
            
            result = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            assert result.returncode == 0, f"容器启动失败: {result.stderr.decode()}"
            time.sleep(5)
            
            # 在容器内创建recorder文件
            recorder_filename = "gid:1_sid:1.log"
            recorder_content = "test recorder log content\nframe 1\naction 1\n"
            
            # 确保目录存在且有写权限（使用root用户）
            check_dir_cmd = [
                "docker", "exec", "-u", "root", test_container_name,
                "bash", "-c",
                "mkdir -p /home/carla/recordings && chown -R carla:carla /home/carla/recordings && chmod 777 /home/carla/recordings"
            ]
            subprocess.run(check_dir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            
            # 使用docker exec在容器内创建文件（使用carla用户）
            create_cmd = [
                "docker", "exec", test_container_name,
                "bash", "-c",
                f"echo '{recorder_content}' > /home/carla/recordings/{recorder_filename}"
            ]
            
            create_result = subprocess.run(
                create_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            # 如果创建失败，输出错误信息
            if create_result.returncode != 0:
                error_msg = create_result.stderr.decode() if create_result.stderr else "Unknown error"
                # 检查目录内容
                ls_cmd = ["docker", "exec", test_container_name, "ls", "-la", "/home/carla/recordings"]
                ls_result = subprocess.run(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                pytest.fail(f"无法在容器内创建文件: {error_msg}\n目录内容: {ls_result.stdout.decode()}")
            
            # 等待文件同步
            time.sleep(1)
            
            # 验证文件在主机上存在
            host_file = test_recorder_dir / recorder_filename
            assert host_file.exists(), f"Recorder文件应该在主机上存在: {host_file}"
            # 读取文件时可能有一些格式差异，只检查内容包含
            file_content = host_file.read_text()
            assert "test recorder log content" in file_content, f"文件内容应该正确，实际内容: {file_content[:100]}"
            
            # 验证文件格式
            assert host_file.name.startswith("gid:"), "文件名应该以gid:开头"
            assert host_file.name.endswith(".log"), "文件名应该以.log结尾"
            assert "sid:" in host_file.name, "文件名应该包含sid:"
            
        finally:
            cleanup_container(test_container_name)
    
    def test_multiple_recorder_files(self, docker_available, test_container_name, test_recorder_dir, test_port_base):
        """测试创建多个recorder文件"""
        cleanup_container(test_container_name)
        
        try:
            # 参考真实CARLA启动方式
            import os
            import getpass
            
            user = getpass.getuser()
            xsock = "/tmp/.X11-unix"
            xauth = os.path.expanduser(f"~{user}/.Xauthority")
            
            volumes = [
                f"--volume={xsock}:{xsock}:rw",
                f"--volume={test_recorder_dir}:/home/carla/recordings:rw"
            ]
            
            if os.path.exists(xauth):
                volumes.append(f"--volume={xauth}:{xauth}:rw")
            
            port = test_port_base + 2
            docker_cmd = [
                "docker", "run",
                "--name", test_container_name,
                "-d", "--rm",
                "-p", f"{port}:4000",
            ] + volumes + [
                "--privileged",
                "carlasim/carla:0.9.13",
                "sleep", "30"
            ]
            
            result = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            assert result.returncode == 0, f"容器启动失败: {result.stderr.decode()}"
            time.sleep(5)
            
            # 确保目录权限
            check_dir_cmd = [
                "docker", "exec", "-u", "root", test_container_name,
                "bash", "-c",
                "mkdir -p /home/carla/recordings && chown -R carla:carla /home/carla/recordings && chmod 777 /home/carla/recordings"
            ]
            subprocess.run(check_dir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            
            # 创建多个recorder文件
            num_files = 5
            for i in range(num_files):
                recorder_filename = f"gid:1_sid:{i}.log"
                recorder_content = f"recorder content for scenario {i}\n"
                
                create_cmd = [
                    "docker", "exec", test_container_name,
                    "bash", "-c",
                    f"echo '{recorder_content}' > /home/carla/recordings/{recorder_filename}"
                ]
                
                result = subprocess.run(
                    create_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5
                )
                if result.returncode != 0:
                    pytest.fail(f"无法创建文件 {recorder_filename}: {result.stderr.decode()}")
            
            # 等待文件同步
            time.sleep(1)
            
            # 验证所有文件都在主机上存在
            log_files = list(test_recorder_dir.glob("*.log"))
            assert len(log_files) == num_files, f"应该有{num_files}个recorder文件，实际有{len(log_files)}个"
            
            # 验证文件内容
            for i in range(num_files):
                expected_file = test_recorder_dir / f"gid:1_sid:{i}.log"
                assert expected_file.exists(), f"文件应该存在: {expected_file}"
                content = expected_file.read_text()
                assert f"scenario {i}" in content, f"文件内容应该正确: {expected_file}"
            
        finally:
            cleanup_container(test_container_name)


class TestRealRecorderSaveAndArchive:
    """真实环境下的recorder保存和归档测试"""
    
    def test_save_files_with_real_recorder_dir(self, docker_available, test_recorder_dir, tmp_path):
        """测试save_files函数处理真实的recorder目录"""
        import sys
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from experiments.core import environment_manager
        
        # 创建项目结构
        project_test = tmp_path / "project"
        output_dir = project_test / "data" / "output"
        recorder_dir = output_dir / "recorder"
        recorder_dir.mkdir(parents=True)
        
        # 复制测试recorder文件到output/recorder
        test_file = recorder_dir / "gid:1_sid:1.log"
        test_file.write_text("real recorder content\nframe data\n")
        
        # 调用save_files
        environment_manager.save_files(project_test)
        
        # 验证文件被保存
        save_base = project_test / "data" / "save"
        assert save_base.exists(), "save目录应该存在"
        
        save_dirs = sorted(save_base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if save_dirs:
            latest_save = save_dirs[0]
            saved_recorder = latest_save / "recorder"
            
            assert saved_recorder.exists(), "recorder目录应该被保存"
            saved_file = saved_recorder / "gid:1_sid:1.log"
            assert saved_file.exists(), "recorder文件应该被保存"
            assert saved_file.read_text() == "real recorder content\nframe data\n", "文件内容应该正确"
    
    def test_archive_with_real_recorder_files(self, docker_available, tmp_path):
        """测试归档功能处理真实的recorder文件"""
        import sys
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from experiments.core import experiment_manager
        from unittest.mock import MagicMock, patch
        
        project_test = tmp_path / "project"
        method_name = "TestMethod"
        experiment_id = "test_exp_real"
        
        # 创建实验目录结构
        method_dir = project_test / "experiment_results" / method_name / experiment_id
        recorder_dir = method_dir / "recorder"
        recorder_dir.mkdir(parents=True)
        
        # 创建真实的recorder文件
        for i in range(3):
            recorder_file = recorder_dir / f"gid:1_sid:{i}.log"
            recorder_file.write_text(f"real recorder content {i}\nframe data {i}\n")
        
        # Mock依赖
        with patch('experiments.core.experiment_manager.PROJECT_ROOT', project_test):
            with patch('experiments.core.experiment_manager.ProgressTracker') as mock_progress:
                with patch('experiments.core.experiment_manager.TimeEstimator') as mock_time:
                    mock_progress_instance = MagicMock()
                    mock_progress_instance.checkpoint_file = str(project_test / "checkpoint.json")
                    mock_progress.return_value = mock_progress_instance
                    
                    mock_time_instance = MagicMock()
                    mock_time_instance.history_file = str(project_test / "time_history.json")
                    mock_time.return_value = mock_time_instance
                    
                    manager = experiment_manager.ExperimentManager(
                        output_base_dir=str(project_test / "experiment_results")
                    )
                    manager.progress_tracker = mock_progress_instance
                    manager.time_estimator = mock_time_instance
                    
                    # 调用归档
                    manager._archive_experiment_run(method_name, experiment_id, method_dir)
        
        # 验证归档
        archive_dir = project_test / "data" / "experiment_snapshots" / method_name / experiment_id
        assert archive_dir.exists(), "归档目录应该存在"
        
        archived_recorder = archive_dir / "results" / "recorder"
        assert archived_recorder.exists(), "recorder目录应该被归档"
        
        archived_files = list(archived_recorder.glob("*.log"))
        assert len(archived_files) == 3, f"应该有3个recorder文件，实际有{len(archived_files)}个"
        
        # 验证文件内容
        for i in range(3):
            archived_file = archived_recorder / f"gid:1_sid:{i}.log"
            assert archived_file.exists(), f"文件应该被归档: {archived_file}"
            content = archived_file.read_text()
            assert f"content {i}" in content, f"文件内容应该正确: {archived_file}"


class TestRealRecorderWorkflow:
    """完整的真实recorder工作流程测试"""
    
    def test_complete_real_workflow(self, docker_available, test_container_name, test_recorder_dir, tmp_path, test_port_base):
        """测试完整的真实工作流程：Docker -> 创建文件 -> 保存 -> 归档"""
        import sys
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from experiments.core import environment_manager, experiment_manager
        from unittest.mock import MagicMock, patch
        
        cleanup_container(test_container_name)
        
        try:
            # 1. 启动Docker容器（参考真实CARLA启动方式）
            import os
            import getpass
            
            user = getpass.getuser()
            xsock = "/tmp/.X11-unix"
            xauth = os.path.expanduser(f"~{user}/.Xauthority")
            
            volumes = [
                f"--volume={xsock}:{xsock}:rw",
                f"--volume={test_recorder_dir}:/home/carla/recordings:rw"
            ]
            
            if os.path.exists(xauth):
                volumes.append(f"--volume={xauth}:{xauth}:rw")
            
            port = test_port_base + 3
            docker_cmd = [
                "docker", "run",
                "--name", test_container_name,
                "-d", "--rm",
                "-p", f"{port}:4000",
            ] + volumes + [
                "--privileged",
                "carlasim/carla:0.9.13",
                "sleep", "30"
            ]
            
            result = subprocess.run(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            
            assert result.returncode == 0, f"容器启动失败: {result.stderr.decode()}"
            time.sleep(5)
            
            # 2. 确保目录权限
            check_dir_cmd = [
                "docker", "exec", "-u", "root", test_container_name,
                "bash", "-c",
                "mkdir -p /home/carla/recordings && chown -R carla:carla /home/carla/recordings && chmod 777 /home/carla/recordings"
            ]
            subprocess.run(check_dir_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            
            # 3. 在容器内创建recorder文件（模拟CARLA recorder）
            recorder_filename = "gid:1_sid:1.log"
            recorder_content = "complete workflow test\nframe 1\naction 1\nframe 2\naction 2\n"
            
            create_cmd = [
                "docker", "exec", test_container_name,
                "bash", "-c",
                f"echo '{recorder_content}' > /home/carla/recordings/{recorder_filename}"
            ]
            
            create_result = subprocess.run(
                create_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10
            )
            
            if create_result.returncode != 0:
                pytest.fail(f"无法在容器内创建文件: {create_result.stderr.decode()}")
            
            # 等待文件同步
            time.sleep(1)
            
            # 4. 验证文件在主机上存在
            host_file = test_recorder_dir / recorder_filename
            assert host_file.exists(), "Recorder文件应该在主机上存在"
            
            # 4. 复制到项目output目录并测试save_files
            project_test = tmp_path / "project"
            output_dir = project_test / "data" / "output"
            project_recorder_dir = output_dir / "recorder"
            project_recorder_dir.mkdir(parents=True)
            
            shutil.copy(host_file, project_recorder_dir / recorder_filename)
            
            # 5. 测试save_files
            environment_manager.save_files(project_test)
            
            save_base = project_test / "data" / "save"
            save_dirs = sorted(save_base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if save_dirs:
                latest_save = save_dirs[0]
                saved_recorder = latest_save / "recorder"
                assert saved_recorder.exists(), "recorder应该被保存"
            
            # 6. 测试归档
            method_name = "TestMethod"
            experiment_id = "test_workflow"
            method_dir = project_test / "experiment_results" / method_name / experiment_id
            method_recorder_dir = method_dir / "recorder"
            method_recorder_dir.mkdir(parents=True)
            shutil.copy(host_file, method_recorder_dir / recorder_filename)
            
            with patch('experiments.core.experiment_manager.PROJECT_ROOT', project_test):
                with patch('experiments.core.experiment_manager.ProgressTracker') as mock_progress:
                    with patch('experiments.core.experiment_manager.TimeEstimator') as mock_time:
                        mock_progress_instance = MagicMock()
                        mock_progress_instance.checkpoint_file = str(project_test / "checkpoint.json")
                        mock_progress.return_value = mock_progress_instance
                        
                        mock_time_instance = MagicMock()
                        mock_time_instance.history_file = str(project_test / "time_history.json")
                        mock_time.return_value = mock_time_instance
                        
                        manager = experiment_manager.ExperimentManager(
                            output_base_dir=str(project_test / "experiment_results")
                        )
                        manager.progress_tracker = mock_progress_instance
                        manager.time_estimator = mock_time_instance
                        
                        manager._archive_experiment_run(method_name, experiment_id, method_dir)
            
            # 7. 验证归档结果
            archive_dir = project_test / "data" / "experiment_snapshots" / method_name / experiment_id
            archived_recorder = archive_dir / "results" / "recorder"
            assert archived_recorder.exists(), "recorder应该被归档"
            
            archived_file = archived_recorder / recorder_filename
            assert archived_file.exists(), "recorder文件应该被归档"
            # 文件内容可能末尾有换行符差异，检查主要内容
            archived_content = archived_file.read_text()
            assert "complete workflow test" in archived_content, "文件内容应该正确"
            assert "frame 1" in archived_content, "文件内容应该包含frame 1"
            assert "action 2" in archived_content, "文件内容应该包含action 2"
            
        finally:
            cleanup_container(test_container_name)

