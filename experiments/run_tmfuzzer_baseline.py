#!/usr/bin/env python3
"""
TM-Fuzzer Baseline Runner
Wraps script/test.py to integrate with experiments framework
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tmfuzzer_quantitative(num_scenarios: int, output_dir: str = "./experiment_results",
                              target: str = "autoware", density: str = "0.4", 
                              town: str = "3", timeout: int = 60):
    """
    Run TM-Fuzzer定量运行（通过估算时间转换为定时运行）
    
    Args:
        num_scenarios: 目标场景数
        output_dir: 输出目录
        target: 测试目标 (autoware/behavior)
        density: 车辆密度
        town: 城镇编号
        timeout: 单个场景超时时间
    """
    # 估算时间（平均50秒/场景，但至少300秒以确保完整运行）
    avg_scenario_time = 50.0
    estimated_duration = max(int(num_scenarios * avg_scenario_time), 300)
    
    print(f"\n{'='*60}")
    print(f"Running TM-Fuzzer Baseline")
    print(f"Target scenarios: {num_scenarios}")
    print(f"Estimated duration: {estimated_duration} seconds (~{estimated_duration/3600:.2f} hours)")
    print(f"{'='*60}\n")
    
    # 获取项目根目录和script目录
    project_root = Path(__file__).parent.parent.resolve()
    script_dir = project_root / "script"
    test_script = script_dir / "test.py"
    original_dir = os.getcwd()
    
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")
    
    try:
        # script/test.py需要从script目录运行，它会自己管理环境
        # 设置PYTHONPATH包含项目根目录
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root) + (':' + env.get('PYTHONPATH', '') if env.get('PYTHONPATH') else '')
        
        # 使用虚拟环境中的python（如果存在）
        venv_python = project_root / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        
        cmd = [
            python_cmd, "test.py",
            target,
            density,
            town,
            str(estimated_duration)
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        print(f"Working directory: {script_dir}")
        print(f"PYTHONPATH: {env['PYTHONPATH']}")
        print(f"Note: script/test.py will call init.sh to manage environment\n")
        
        # 运行命令（从script目录运行，设置PYTHONPATH）
        # script/test.py会自己调用init_environment()来管理docker和环境
        result = subprocess.run(cmd, cwd=str(script_dir), env=env)
        
        if result.returncode == 0:
            print(f"\n✓ TM-Fuzzer completed successfully")
        else:
            print(f"\n[ERROR] TM-Fuzzer exited with code {result.returncode}")
            raise RuntimeError(f"TM-Fuzzer failed with exit code {result.returncode}")
            
    except Exception as e:
        print(f"[ERROR] Failed to run TM-Fuzzer: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to ensure error is visible
    finally:
        os.chdir(original_dir)


def run_tmfuzzer_timed(hours: float, output_dir: str = "./experiment_results",
                       target: str = "autoware", density: str = "0.4",
                       town: str = "3", timeout: int = 60):
    """
    TM-Fuzzer定时运行
    
    Args:
        hours: 运行小时数
        output_dir: 输出目录
        target: 测试目标
        density: 车辆密度
        town: 城镇编号
        timeout: 单个场景超时时间
    """
    duration_seconds = int(hours * 3600)
    
    print(f"\n{'='*60}")
    print(f"Running TM-Fuzzer Baseline")
    print(f"Duration: {hours} hours ({duration_seconds} seconds)")
    print(f"{'='*60}\n")
    
    # 获取script目录的绝对路径
    script_dir = Path(__file__).parent.parent / "script"
    script_dir = script_dir.resolve()
    test_script = script_dir / "test.py"
    original_dir = os.getcwd()
    
    if not test_script.exists():
        raise FileNotFoundError(f"Test script not found: {test_script}")
    
    try:
        # script/test.py需要从script目录运行，它会自己管理环境
        # 设置PYTHONPATH包含项目根目录
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root) + (':' + env.get('PYTHONPATH', '') if env.get('PYTHONPATH') else '')
        
        # 使用虚拟环境中的python（如果存在）
        venv_python = project_root / "venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else "python3"
        
        cmd = [
            python_cmd, "test.py",
            target,
            density,
            town,
            str(duration_seconds)
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        print(f"Working directory: {script_dir}")
        print(f"PYTHONPATH: {env['PYTHONPATH']}")
        print(f"Note: script/test.py will call init.sh to manage environment\n")
        
        # 运行命令（从script目录运行，设置PYTHONPATH）
        # script/test.py会自己调用init_environment()来管理docker和环境
        result = subprocess.run(cmd, cwd=str(script_dir), env=env)
        
        if result.returncode == 0:
            print(f"\n✓ TM-Fuzzer completed successfully")
        else:
            print(f"\n[ERROR] TM-Fuzzer exited with code {result.returncode}")
            raise RuntimeError(f"TM-Fuzzer failed with exit code {result.returncode}")
            
    except Exception as e:
        print(f"[ERROR] Failed to run TM-Fuzzer: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to ensure error is visible
    finally:
        os.chdir(original_dir)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run TM-Fuzzer baseline experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantitative: 100 scenarios
  python run_tmfuzzer_baseline.py --num-scenarios 100
  
  # Timed: 2 hours
  python run_tmfuzzer_baseline.py --hours 2
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--num-scenarios', type=int,
                      help='Number of scenarios to generate')
    group.add_argument('--hours', type=float,
                      help='Duration in hours')
    
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                       help='Output directory (default: ./experiment_results)')
    parser.add_argument('--target', type=str, default='autoware',
                       choices=['autoware', 'behavior'],
                       help='Target ADS system (default: autoware)')
    parser.add_argument('--density', type=str, default='0.4',
                       help='Vehicle density (default: 0.4)')
    parser.add_argument('--town', type=str, default='3',
                       help='CARLA town number (default: 3)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Scenario timeout in seconds (default: 60)')
    
    args = parser.parse_args()
    
    if args.num_scenarios:
        run_tmfuzzer_quantitative(
            num_scenarios=args.num_scenarios,
            output_dir=args.output_dir,
            target=args.target,
            density=args.density,
            town=args.town,
            timeout=args.timeout
        )
    elif args.hours:
        run_tmfuzzer_timed(
            hours=args.hours,
            output_dir=args.output_dir,
            target=args.target,
            density=args.density,
            town=args.town,
            timeout=args.timeout
        )


if __name__ == "__main__":
    main()

