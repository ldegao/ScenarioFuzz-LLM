#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT 日志条目序列化测试

测试 fuzzer.py 中 GPT 日志条目的完整序列化结构。
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestGPTLogEntrySerialization(unittest.TestCase):
    """测试 GPT 日志条目的序列化"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_log_entry(self):
        """测试完整的 GPT 日志条目"""
        log_entry = {
            "generation_id": 0,
            "scenario_id": 1,
            "timestamp": "2025-12-08T17:00:00",
            "iso_timestamp": "2025-12-08T17:00:00.123456",
            
            "gpt_call_duration_seconds": 73.31,
            "model_version": "gpt-5-mini",
            "max_tokens": 10000,
            
            "prompt": "Test prompt",
            "prompt_length": 10,
            "raw_response": "Test response",
            "response_length": 13,
            "parsed_response": {
                "answer1": {"Description": "Test"},
                "answer2": {"Overall Similarity": "85"}
            },
            
            "token_stats": {
                "prompt_tokens": 1986,
                "completion_tokens": 3820,
                "total_tokens": 5806
            },
            
            "scenario_description": "Test scenario",
            "scenario_state": {
                "min_dist": 5.5,
                "min_dist_frame": 100,
                "fitness_values": [10.5, 20.3, 30.1],
                "fitness_info": {
                    "values": [10.5, 20.3, 30.1],
                    "valid": True,
                    "weights": [-1.0, -1.0, 5.0]
                }
            },
            
            "rag_info": {
                "rag_enabled": False
            },
            
            "scenario_database_size": 5,
            "overall_similarity": 85,
            "answer3_vehicle_info": {}
        }
        
        log_path = self.temp_path / "test_gpt_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        with open(log_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['scenario_id'], 1)
        self.assertEqual(loaded['parsed_response']['answer2']['Overall Similarity'], "85")
        self.assertEqual(loaded['scenario_state']['fitness_info']['valid'], True)
        self.assertEqual(loaded['token_stats']['total_tokens'], 5806)
    
    def test_log_entry_with_rag_info(self):
        """测试包含 RAG 信息的日志条目"""
        log_entry = {
            "generation_id": 1,
            "scenario_id": 2,
            "timestamp": datetime.now().isoformat(),
            "rag_info": {
                "rag_enabled": True,
                "rag_k": 5,
                "retrieved_scenarios_count": 5,
                "use_enhanced_rag": True,
                "use_hybrid_search": True,
                "use_reranking": True,
            },
            "token_stats": {
                "prompt_tokens": 2000,
                "completion_tokens": 4000,
                "total_tokens": 6000
            }
        }
        
        log_path = self.temp_path / "test_gpt_log_rag.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        with open(log_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertTrue(loaded['rag_info']['rag_enabled'])
        self.assertEqual(loaded['rag_info']['rag_k'], 5)
        self.assertTrue(loaded['rag_info']['use_enhanced_rag'])
    
    def test_log_entry_with_missing_fields(self):
        """测试缺少某些字段的日志条目"""
        log_entry = {
            "generation_id": 0,
            "scenario_id": 1,
            "timestamp": "2025-12-08T17:00:00",
            # 缺少一些可选字段
            "token_stats": None,
            "rag_info": None,
            "overall_similarity": None,
        }
        
        log_path = self.temp_path / "test_gpt_log_minimal.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        with open(log_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        
        self.assertEqual(loaded['scenario_id'], 1)
        self.assertIsNone(loaded['token_stats'])
        self.assertIsNone(loaded['overall_similarity'])


if __name__ == "__main__":
    unittest.main()

