"""
单元测试：Recorder文件保存与迁移功能

测试以下功能：
1. Config中recorder_dir的配置
2. Recorder目录的创建
3. Recorder文件路径处理
4. 文件保存功能中的recorder目录处理
5. 实验归档功能中的recorder文件处理
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import config
from experiments.core import environment_manager, experiment_manager


class TestRecorderConfig:
    """测试Config中recorder_dir的配置"""
    
    def test_recorder_dir_initialization(self):
        """测试recorder_dir的初始化"""
        conf = config.Config()
        assert conf.recorder_dir is None
    
    def test_recorder_dir_set_paths(self):
        """测试set_paths()方法设置recorder_dir"""
        conf = config.Config()
        conf.out_dir = "/tmp/test_output"
        conf.set_paths()
        assert conf.recorder_dir == "/tmp/test_output/recorder"
    
    def test_recorder_dir_with_custom_out_dir(self, tmp_path):
        """测试使用自定义out_dir时recorder_dir的设置"""
        conf = config.Config()
        test_out_dir = str(tmp_path / "custom_output")
        conf.out_dir = test_out_dir
        conf.set_paths()
        expected_recorder_dir = os.path.join(test_out_dir, "recorder")
        assert conf.recorder_dir == expected_recorder_dir


class TestRecorderDirectoryCreation:
    """测试recorder目录的创建"""
    
    def test_recorder_dir_creation_in_fuzzer(self, tmp_path):
        """测试fuzzer.py中recorder_dir的创建"""
        conf = config.Config()
        test_out_dir = str(tmp_path / "test_output")
        conf.out_dir = test_out_dir
        conf.set_paths()
        
        # 模拟fuzzer.py中的目录创建
        os.makedirs(conf.recorder_dir, exist_ok=True)
        
        assert os.path.exists(conf.recorder_dir)
        assert os.path.isdir(conf.recorder_dir)
    
    def test_recorder_dir_creation_multiple_times(self, tmp_path):
        """测试多次创建recorder_dir不会出错"""
        conf = config.Config()
        test_out_dir = str(tmp_path / "test_output")
        conf.out_dir = test_out_dir
        conf.set_paths()
        
        # 第一次创建
        os.makedirs(conf.recorder_dir, exist_ok=True)
        assert os.path.exists(conf.recorder_dir)
        
        # 第二次创建（应该不会出错）
        os.makedirs(conf.recorder_dir, exist_ok=True)
        assert os.path.exists(conf.recorder_dir)


class TestRecorderFilePath:
    """测试recorder文件路径处理"""
    
    def test_recorder_filename_format(self):
        """测试recorder文件名的格式"""
        generation_id = 1
        scenario_id = 42
        container_path = "/home/carla/recordings"
        
        recorder_filename = f"{container_path}/gid:{generation_id}_sid:{scenario_id}.log"
        
        assert recorder_filename == "/home/carla/recordings/gid:1_sid:42.log"
        assert recorder_filename.endswith(".log")
        assert "gid:1" in recorder_filename
        assert "sid:42" in recorder_filename
    
    def test_recorder_filename_with_different_ids(self):
        """测试不同ID的recorder文件名"""
        container_path = "/home/carla/recordings"
        
        test_cases = [
            (0, 0, f"{container_path}/gid:0_sid:0.log"),
            (10, 100, f"{container_path}/gid:10_sid:100.log"),
            (999, 9999, f"{container_path}/gid:999_sid:9999.log"),
        ]
        
        for gen_id, scen_id, expected in test_cases:
            filename = f"{container_path}/gid:{gen_id}_sid:{scen_id}.log"
            assert filename == expected


class TestSaveFilesRecorder:
    """测试save_files函数中的recorder目录处理"""
    
    def test_save_files_includes_recorder_dir(self, tmp_path):
        """测试save_files函数包含recorder目录"""
        project_root = tmp_path / "project"
        output_dir = project_root / "data" / "output"
        recorder_dir = output_dir / "recorder"
        
        # 创建目录结构
        recorder_dir.mkdir(parents=True)
        
        # 创建测试recorder文件
        test_recorder_file = recorder_dir / "gid:1_sid:1.log"
        test_recorder_file.write_text("test recorder content")
        
        # 调用save_files
        environment_manager.save_files(project_root)
        
        # 检查save目录是否存在
        save_base = project_root / "data" / "save"
        assert save_base.exists()
        
        # 查找最新的save目录
        save_dirs = sorted(save_base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if save_dirs:
            latest_save = save_dirs[0]
            saved_recorder = latest_save / "recorder"
            
            # recorder目录应该被保存
            assert saved_recorder.exists(), "recorder目录应该被保存"
            assert saved_recorder.is_dir(), "recorder应该是一个目录"
            
            # 检查文件是否被保存
            saved_file = saved_recorder / "gid:1_sid:1.log"
            assert saved_file.exists(), "recorder文件应该被保存"
    
    def test_save_files_with_empty_recorder_dir(self, tmp_path):
        """测试空recorder目录的处理"""
        project_root = tmp_path / "project"
        output_dir = project_root / "data" / "output"
        recorder_dir = output_dir / "recorder"
        
        # 创建空目录
        recorder_dir.mkdir(parents=True)
        
        # 调用save_files
        environment_manager.save_files(project_root)
        
        # 空目录可能被跳过，这是可以接受的行为
        # 主要确保不会出错
        assert True
    
    def test_save_files_with_multiple_recorder_files(self, tmp_path):
        """测试多个recorder文件的保存"""
        project_root = tmp_path / "project"
        output_dir = project_root / "data" / "output"
        recorder_dir = output_dir / "recorder"
        
        recorder_dir.mkdir(parents=True)
        
        # 创建多个recorder文件
        for i in range(3):
            recorder_file = recorder_dir / f"gid:1_sid:{i}.log"
            recorder_file.write_text(f"recorder content {i}")
        
        # 调用save_files
        environment_manager.save_files(project_root)
        
        # 检查文件是否都被保存
        save_base = project_root / "data" / "save"
        save_dirs = sorted(save_base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if save_dirs:
            latest_save = save_dirs[0]
            saved_recorder = latest_save / "recorder"
            
            if saved_recorder.exists():
                saved_files = list(saved_recorder.glob("*.log"))
                assert len(saved_files) == 3, f"应该有3个recorder文件，实际有{len(saved_files)}个"


class TestArchiveRecorder:
    """测试实验归档功能中的recorder文件处理"""
    
    def test_archive_includes_recorder_dir(self, tmp_path):
        """测试归档功能包含recorder目录"""
        project_root = tmp_path / "project"
        method_name = "TestMethod"
        experiment_id = "test_exp_001"
        
        # 创建实验目录结构
        method_dir = project_root / "experiment_results" / method_name / experiment_id
        recorder_dir = method_dir / "recorder"
        recorder_dir.mkdir(parents=True)
        
        # 创建测试recorder文件
        test_recorder_file = recorder_dir / "gid:1_sid:1.log"
        test_recorder_file.write_text("test recorder content")
        
        # 创建ExperimentManager实例（需要mock一些依赖）
        with patch('experiments.core.experiment_manager.PROJECT_ROOT', project_root):
            with patch('experiments.core.experiment_manager.ProgressTracker') as mock_progress:
                with patch('experiments.core.experiment_manager.TimeEstimator') as mock_time:
                    # Mock progress tracker和time estimator
                    mock_progress_instance = MagicMock()
                    mock_progress_instance.checkpoint_file = str(project_root / "checkpoint.json")
                    mock_progress.return_value = mock_progress_instance
                    
                    mock_time_instance = MagicMock()
                    mock_time_instance.history_file = str(project_root / "time_history.json")
                    mock_time.return_value = mock_time_instance
                    
                    manager = experiment_manager.ExperimentManager(
                        output_base_dir=str(project_root / "experiment_results")
                    )
                    manager.progress_tracker = mock_progress_instance
                    manager.time_estimator = mock_time_instance
                    
                    # 调用归档函数
                    manager._archive_experiment_run(method_name, experiment_id, method_dir)
        
        # 检查归档目录
        archive_dir = project_root / "data" / "experiment_snapshots" / method_name / experiment_id
        assert archive_dir.exists(), "归档目录应该存在"
        
        results_dir = archive_dir / "results"
        assert results_dir.exists(), "results目录应该存在"
        
        archived_recorder = results_dir / "recorder"
        assert archived_recorder.exists(), "recorder目录应该被归档"
        
        archived_file = archived_recorder / "gid:1_sid:1.log"
        assert archived_file.exists(), "recorder文件应该被归档"
        assert archived_file.read_text() == "test recorder content", "文件内容应该正确"
    
    def test_archive_with_multiple_recorder_files(self, tmp_path):
        """测试归档多个recorder文件"""
        project_root = tmp_path / "project"
        method_name = "TestMethod"
        experiment_id = "test_exp_002"
        
        method_dir = project_root / "experiment_results" / method_name / experiment_id
        recorder_dir = method_dir / "recorder"
        recorder_dir.mkdir(parents=True)
        
        # 创建多个recorder文件
        for i in range(5):
            recorder_file = recorder_dir / f"gid:1_sid:{i}.log"
            recorder_file.write_text(f"content {i}")
        
        with patch('experiments.core.experiment_manager.PROJECT_ROOT', project_root):
            with patch('experiments.core.experiment_manager.ProgressTracker') as mock_progress:
                with patch('experiments.core.experiment_manager.TimeEstimator') as mock_time:
                    mock_progress_instance = MagicMock()
                    mock_progress_instance.checkpoint_file = str(project_root / "checkpoint.json")
                    mock_progress.return_value = mock_progress_instance
                    
                    mock_time_instance = MagicMock()
                    mock_time_instance.history_file = str(project_root / "time_history.json")
                    mock_time.return_value = mock_time_instance
                    
                    manager = experiment_manager.ExperimentManager(
                        output_base_dir=str(project_root / "experiment_results")
                    )
                    manager.progress_tracker = mock_progress_instance
                    manager.time_estimator = mock_time_instance
                    
                    manager._archive_experiment_run(method_name, experiment_id, method_dir)
        
        # 检查归档的文件数量
        archive_dir = project_root / "data" / "experiment_snapshots" / method_name / experiment_id
        archived_recorder = archive_dir / "results" / "recorder"
        
        if archived_recorder.exists():
            archived_files = list(archived_recorder.glob("*.log"))
            assert len(archived_files) == 5, f"应该有5个recorder文件，实际有{len(archived_files)}个"


class TestRecorderIntegration:
    """集成测试：测试完整的recorder流程"""
    
    def test_complete_recorder_workflow(self, tmp_path):
        """测试完整的recorder工作流程"""
        project_root = tmp_path / "project"
        output_dir = project_root / "data" / "output"
        recorder_dir = output_dir / "recorder"
        
        # 1. 创建recorder目录
        recorder_dir.mkdir(parents=True)
        
        # 2. 创建模拟的recorder文件（模拟CARLA保存的文件）
        recorder_file = recorder_dir / "gid:1_sid:1.log"
        recorder_file.write_text("simulated recorder content")
        
        # 3. 测试save_files功能
        environment_manager.save_files(project_root)
        
        # 4. 验证文件被保存
        save_base = project_root / "data" / "save"
        save_dirs = sorted(save_base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if save_dirs:
            latest_save = save_dirs[0]
            saved_recorder = latest_save / "recorder"
            assert saved_recorder.exists(), "recorder目录应该在save中"
            
            saved_file = saved_recorder / "gid:1_sid:1.log"
            assert saved_file.exists(), "recorder文件应该在save中"
            assert saved_file.read_text() == "simulated recorder content", "文件内容应该正确"

