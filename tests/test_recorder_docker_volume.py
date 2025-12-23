"""
单元测试：Recorder Docker Volume映射功能

测试Docker容器启动时的volume映射配置：
1. run_carla.sh脚本中的volume映射
2. script/test.py中run_carla()函数的volume映射
"""

import inspect
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestRunCarlaScriptVolumeMapping:
    """测试run_carla.sh脚本中的volume映射"""
    
    def test_run_carla_sh_has_recorder_volume(self):
        """测试run_carla.sh脚本包含recorder volume映射"""
        script_path = Path(__file__).parent.parent / "script" / "run_carla.sh"
        
        if not script_path.exists():
            pytest.skip("run_carla.sh not found")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # 检查是否包含recorder volume映射
        assert "-v" in content, "应该包含volume映射参数"
        assert "/home/carla/recordings" in content, "应该包含容器内recorder路径"
        assert "recorder" in content, "应该包含recorder相关配置"
        
        # 检查是否创建recorder目录
        assert "mkdir" in content or "RECORDER_DIR" in content, "应该创建recorder目录"
    
    def test_run_carla_sh_recorder_dir_creation(self):
        """测试run_carla.sh脚本中recorder目录的创建逻辑"""
        script_path = Path(__file__).parent.parent / "script" / "run_carla.sh"
        
        if not script_path.exists():
            pytest.skip("run_carla.sh not found")
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # 检查是否在docker run之前创建目录
        lines = content.split('\n')
        docker_run_line = None
        mkdir_line = None
        
        for i, line in enumerate(lines):
            if 'docker run' in line:
                docker_run_line = i
            if 'mkdir' in line and 'recorder' in line.lower():
                mkdir_line = i
        
        if mkdir_line is not None and docker_run_line is not None:
            assert mkdir_line < docker_run_line, "应该在docker run之前创建recorder目录"


class TestRunCarlaFunctionVolumeMapping:
    """测试script/test.py中run_carla()函数的volume映射"""
    
    def test_run_carla_function_imports(self):
        """测试run_carla函数可以正常导入"""
        try:
            import sys
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from script.test import run_carla
            assert callable(run_carla), "run_carla应该是可调用的函数"
        except ImportError as e:
            pytest.skip(f"Cannot import run_carla: {e}")
    
    def test_run_carla_function_has_volume_mapping(self):
        """测试run_carla函数包含volume映射逻辑"""
        try:
            import sys
            import inspect
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from script.test import run_carla
            source = inspect.getsource(run_carla)
            
            # 检查是否包含recorder相关代码
            assert "recorder" in source.lower(), "应该包含recorder相关代码"
            assert "/home/carla/recordings" in source, "应该包含容器内recorder路径"
            assert "volume" in source.lower() or "-v" in source, "应该包含volume映射"
        except ImportError as e:
            pytest.skip(f"Cannot import run_carla: {e}")
    
    def test_run_carla_creates_recorder_dir(self, tmp_path, monkeypatch):
        """测试run_carla函数创建recorder目录"""
        try:
            import sys
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from script.test import OUTPUT_DIR, run_carla
            
            # Mock subprocess.Popen to avoid actually running docker
            mock_popen = MagicMock()
            mock_process = MagicMock()
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process
            
            with patch('script.test.subprocess.Popen', mock_popen):
                with patch('script.test.wait_for_carla_server', return_value=True):
                    # 测试函数不会实际运行docker，但会检查代码逻辑
                    # 这里主要验证代码结构
                    source_code = inspect.getsource(run_carla)
                    assert "makedirs" in source_code or "os.makedirs" in source_code, \
                        "应该创建recorder目录"
        except ImportError as e:
            pytest.skip(f"Cannot test run_carla: {e}")


class TestRecorderVolumePathConsistency:
    """测试recorder volume路径的一致性"""
    
    def test_container_path_consistency(self):
        """测试容器内路径的一致性"""
        container_path = "/home/carla/recordings"
        
        # 检查simulate.py中使用的路径
        simulate_path = Path(__file__).parent.parent / "simulate.py"
        if simulate_path.exists():
            with open(simulate_path, 'r') as f:
                simulate_content = f.read()
            
            # 应该使用容器内路径
            assert container_path in simulate_content, \
                "simulate.py应该使用容器内路径 /home/carla/recordings"
    
    def test_host_path_structure(self):
        """测试主机路径结构的一致性"""
        # 主机路径应该是 {project_root}/data/output/recorder
        # 或者实验特定的 {experiment_dir}/recorder
        
        # 检查config.py中的路径设置
        config_path = Path(__file__).parent.parent / "config.py"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # 应该使用os.path.join来构建路径
            assert "recorder_dir" in config_content, \
                "config.py应该包含recorder_dir配置"
            assert "os.path.join" in config_content or "out_dir" in config_content, \
                "应该使用os.path.join构建recorder_dir路径"


class TestRecorderVolumeMappingIntegration:
    """集成测试：测试volume映射的完整流程"""
    
    def test_volume_mapping_configuration(self):
        """测试volume映射配置的完整性"""
        # 检查所有相关文件是否都正确配置了volume映射
        
        script_path = Path(__file__).parent.parent / "script" / "run_carla.sh"
        test_py_path = Path(__file__).parent.parent / "script" / "test.py"
        
        # 检查run_carla.sh
        if script_path.exists():
            with open(script_path, 'r') as f:
                script_content = f.read()
            
            assert "-v" in script_content, "run_carla.sh应该包含volume映射"
            assert "/home/carla/recordings" in script_content, \
                "run_carla.sh应该映射到 /home/carla/recordings"
        
        # 检查script/test.py
        if test_py_path.exists():
            with open(test_py_path, 'r') as f:
                test_content = f.read()
            
            # 检查run_carla函数
            if "def run_carla" in test_content:
                assert "/home/carla/recordings" in test_content, \
                    "run_carla函数应该包含容器内路径"
                assert "recorder" in test_content.lower(), \
                    "run_carla函数应该包含recorder相关代码"

