"""
配置管理器模块
负责加载、验证和管理系统配置
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
        self._apply_defaults()
    
    def _load_config(self):
        """从YAML或JSON文件加载配置"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        file_ext = Path(self.config_path).suffix.lower()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if file_ext in ['.yaml', '.yml']:
                    self.config = yaml.safe_load(f)
                elif file_ext == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_ext}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {str(e)}")
    
    def _validate_config(self):
        """验证配置的有效性"""
        required_sections = ['data', 'models', 'lora', 'dpo', 'rag', 'inference', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置缺少必需的部分: {section}")
        
        # 验证数据配置
        if 'dataset_id' not in self.config['data']:
            raise ValueError("配置缺少 data.dataset_id")
        
        # 验证模型路径配置
        required_model_paths = ['base_model_path', 'embedding_model_path', 'rerank_model_path']
        for path_key in required_model_paths:
            if path_key not in self.config['models']:
                raise ValueError(f"配置缺少 models.{path_key}")
    
    def _apply_defaults(self):
        """为可选配置参数应用默认值"""
        # 数据处理默认值
        self.config['data'].setdefault('filter_banking', True)
        self.config['data'].setdefault('perplexity_threshold', 100.0)
        self.config['data'].setdefault('prompt_template', "问题：{question}\n答案：")
        
        # LoRA默认值
        self.config['lora'].setdefault('rank', 8)
        self.config['lora'].setdefault('alpha', 16)
        self.config['lora'].setdefault('dropout', 0.05)
        self.config['lora'].setdefault('learning_rate', 0.0001)
        self.config['lora'].setdefault('batch_size', 4)
        self.config['lora'].setdefault('epochs', 3)
        
        # DPO默认值
        self.config['dpo'].setdefault('beta', 0.1)
        self.config['dpo'].setdefault('learning_rate', 0.00005)
        self.config['dpo'].setdefault('batch_size', 2)
        self.config['dpo'].setdefault('epochs', 2)
        
        # RAG默认值
        self.config['rag'].setdefault('chunk_max_tokens', 512)
        self.config['rag'].setdefault('retrieval_top_k', 10)
        self.config['rag'].setdefault('rerank_top_n', 5)
        
        # 推理默认值
        self.config['inference'].setdefault('temperature', 0.7)
        self.config['inference'].setdefault('max_tokens', 512)
        self.config['inference'].setdefault('top_p', 0.9)
        
        # 日志默认值
        self.config['logging'].setdefault('level', 'INFO')
        self.config['logging'].setdefault('console', True)
        self.config['logging'].setdefault('file', True)
        self.config['logging'].setdefault('log_file_path', './logs/system.log')
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置键路径，如 "data.dataset_id"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        设置配置值（支持点号分隔的路径）
        
        Args:
            key_path: 配置键路径
            value: 配置值
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy()
    
    def save(self, output_path: Optional[str] = None):
        """
        保存配置到文件
        
        Args:
            output_path: 输出文件路径，默认为原配置文件路径
        """
        save_path = output_path or self.config_path
        file_ext = Path(save_path).suffix.lower()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if file_ext in ['.yaml', '.yml']:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
            elif file_ext == '.json':
                json.dump(self.config, f, ensure_ascii=False, indent=2)
