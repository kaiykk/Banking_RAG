"""
日志记录模块
提供统一的日志记录功能
"""

import logging
import os
from pathlib import Path
from typing import Optional


class Logger:
    """日志记录器类"""
    
    _instances = {}
    
    def __init__(self, name: str, config: dict):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            config: 日志配置字典
        """
        self.name = name
        self.config = config
        self.logger = self._setup_logger()
    
    @classmethod
    def get_logger(cls, name: str, config: Optional[dict] = None):
        """
        获取日志记录器实例（单例模式）
        
        Args:
            name: 日志记录器名称
            config: 日志配置字典
            
        Returns:
            Logger实例
        """
        if name not in cls._instances:
            if config is None:
                config = cls._get_default_config()
            cls._instances[name] = cls(name, config)
        return cls._instances[name]
    
    @staticmethod
    def _get_default_config() -> dict:
        """获取默认日志配置"""
        return {
            'level': 'INFO',
            'console': True,
            'file': True,
            'log_file_path': './logs/system.log',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(self.name)
        
        # 设置日志级别
        level_str = self.config.get('level', 'INFO').upper()
        level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)
        
        # 清除现有处理器
        logger.handlers.clear()
        
        # 日志格式
        formatter = logging.Formatter(
            self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # 控制台处理器
        if self.config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # 文件处理器
        if self.config.get('file', True):
            log_file_path = self.config.get('log_file_path', './logs/system.log')
            
            # 创建日志目录
            log_dir = Path(log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str, *args, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """记录CRITICAL级别日志"""
        self.logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs):
        """记录异常信息"""
        self.logger.exception(message, *args, **kwargs)
