# Memory Monitoring and Optimization Utilities
# For Google Colab Pro training with memory constraints
# Reference: https://huggingface.co/docs/transformers/main/en/performance

import psutil
import GPUtil
import torch
import gc
import logging
from typing import Dict, List, Optional, Tuple
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """
    Comprehensive memory monitoring for GPU training.
    
    This class provides real-time monitoring of GPU and system memory,
    automatic optimization suggestions, and memory leak detection.
    """
    
    def __init__(self, alert_threshold: float = 0.85):
        """
        Initialize memory monitor.
        
        Args:
            alert_threshold: Memory usage threshold for alerts (0.0-1.0)
        """
        self.alert_threshold = alert_threshold
        self.memory_history = []
        self.gpu_history = []
        
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get current system memory information."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "usage_percent": memory.percent / 100,
            "free_gb": memory.free / (1024**3)
        }
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        if not gpu:
            return {"error": "No GPU detected"}
        
        # Get PyTorch GPU memory info
        torch_memory = torch.cuda.memory_stats()
        
        return {
            "total_gb": gpu.memoryTotal / 1024,
            "used_gb": gpu.memoryUsed / 1024,
            "free_gb": gpu.memoryFree / 1024,
            "usage_percent": gpu.memoryUtil,
            "torch_allocated_gb": torch_memory.get("allocated_bytes.all.current", 0) / (1024**3),
            "torch_reserved_gb": torch_memory.get("reserved_bytes.all.current", 0) / (1024**3)
        }
    
    def check_memory_status(self) -> Dict[str, any]:
        """Check current memory status and return alerts if needed."""
        system_mem = self.get_system_memory_info()
        gpu_mem = self.get_gpu_memory_info()
        
        alerts = []
        warnings_list = []
        
        # Check system memory
        if system_mem["usage_percent"] > self.alert_threshold:
            alerts.append(f"System memory usage high: {system_mem['usage_percent']:.1%}")
        
        # Check GPU memory
        if "error" not in gpu_mem:
            if gpu_mem["usage_percent"] > self.alert_threshold:
                alerts.append(f"GPU memory usage high: {gpu_mem['usage_percent']:.1%}")
            
            # Check for potential memory leaks
            if gpu_mem["torch_reserved_gb"] > gpu_mem["torch_allocated_gb"] * 1.5:
                warnings_list.append("Potential memory fragmentation detected")
        
        return {
            "system_memory": system_mem,
            "gpu_memory": gpu_mem,
            "alerts": alerts,
            "warnings": warnings_list,
            "status": "warning" if alerts else "ok"
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get memory optimization suggestions based on current usage."""
        suggestions = []
        status = self.check_memory_status()
        
        gpu_mem = status["gpu_memory"]
        if "error" not in gpu_mem:
            usage = gpu_mem["usage_percent"]
            
            if usage > 0.9:
                suggestions.extend([
                    "Reduce batch size to 1",
                    "Enable gradient checkpointing",
                    "Use 4-bit quantization",
                    "Reduce LoRA rank to 4-8",
                    "Clear GPU cache: torch.cuda.empty_cache()"
                ])
            elif usage > 0.8:
                suggestions.extend([
                    "Consider reducing batch size",
                    "Enable gradient checkpointing",
                    "Monitor for memory leaks"
                ])
            elif usage > 0.7:
                suggestions.append("Memory usage is acceptable but monitor closely")
        
        return suggestions
    
    def log_memory_status(self, step: int = None):
        """Log current memory status."""
        status = self.check_memory_status()
        
        log_msg = f"Memory Status"
        if step is not None:
            log_msg += f" (Step {step})"
        
        logger.info(log_msg)
        logger.info(f"System Memory: {status['system_memory']['usage_percent']:.1%} "
                   f"({status['system_memory']['used_gb']:.1f}GB / "
                   f"{status['system_memory']['total_gb']:.1f}GB)")
        
        if "error" not in status["gpu_memory"]:
            gpu_mem = status["gpu_memory"]
            logger.info(f"GPU Memory: {gpu_mem['usage_percent']:.1%} "
                       f"({gpu_mem['used_gb']:.1f}GB / {gpu_mem['total_gb']:.1f}GB)")
            logger.info(f"PyTorch Allocated: {gpu_mem['torch_allocated_gb']:.1f}GB")
            logger.info(f"PyTorch Reserved: {gpu_mem['torch_reserved_gb']:.1f}GB")
        
        if status["alerts"]:
            for alert in status["alerts"]:
                logger.warning(f"ALERT: {alert}")
        
        if status["warnings"]:
            for warning in status["warnings"]:
                logger.warning(f"WARNING: {warning}")

class MemoryOptimizer:
    """
    Memory optimization utilities for training.
    
    Provides functions to optimize memory usage during training,
    including automatic batch size adjustment and memory cleanup.
    """
    
    @staticmethod
    def clear_memory():
        """Clear all available memory caches."""
        # Clear Python garbage collection
        gc.collect()
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Memory cleared successfully")
    
    @staticmethod
    def get_optimal_batch_size(model, tokenizer, max_length: int = 2048, 
                              target_memory_gb: float = 12.0) -> int:
        """
        Determine optimal batch size based on available memory.
        
        Args:
            model: The model to test
            tokenizer: The tokenizer
            max_length: Maximum sequence length
            target_memory_gb: Target memory usage in GB
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            return 1
        
        # Get available GPU memory
        gpu_mem = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        if not gpu_mem:
            return 1
        
        available_memory_gb = gpu_mem.memoryFree / 1024
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample_gb = 0.5  # Conservative estimate
        
        # Calculate optimal batch size
        optimal_batch_size = int(available_memory_gb * target_memory_gb / 
                                (available_memory_gb * memory_per_sample_gb))
        
        # Ensure minimum batch size of 1
        optimal_batch_size = max(1, min(optimal_batch_size, 8))
        
        logger.info(f"Optimal batch size calculated: {optimal_batch_size}")
        return optimal_batch_size
    
    @staticmethod
    def optimize_model_for_training(model, enable_gradient_checkpointing: bool = True):
        """
        Apply memory optimizations to the model.
        
        Args:
            model: The model to optimize
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
            
        Returns:
            Optimized model
        """
        if enable_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Enable memory efficient attention if available
        if hasattr(model.config, 'use_memory_efficient_attention'):
            model.config.use_memory_efficient_attention = True
            logger.info("Memory efficient attention enabled")
        
        return model

class MemoryCallback:
    """
    Training callback for memory monitoring.
    
    Integrates with HuggingFace Trainer to provide real-time
    memory monitoring and automatic optimization.
    """
    
    def __init__(self, monitor: MemoryMonitor, log_interval: int = 100):
        """
        Initialize memory callback.
        
        Args:
            monitor: MemoryMonitor instance
            log_interval: Steps between memory logging
        """
        self.monitor = monitor
        self.log_interval = log_interval
        self.step_count = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            self.monitor.log_memory_status(self.step_count)
            
            # Check for memory alerts
            status = self.monitor.check_memory_status()
            if status["alerts"]:
                logger.warning("Memory alerts detected - consider optimization")
                suggestions = self.monitor.get_optimization_suggestions()
                for suggestion in suggestions:
                    logger.info(f"SUGGESTION: {suggestion}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        logger.info("Starting memory monitoring")
        self.monitor.log_memory_status(0)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        logger.info("Training completed - final memory status:")
        self.monitor.log_memory_status(self.step_count)
        MemoryOptimizer.clear_memory()

# Convenience functions
def get_memory_info() -> Dict[str, any]:
    """Get comprehensive memory information."""
    monitor = MemoryMonitor()
    return monitor.check_memory_status()

def optimize_memory():
    """Clear memory and apply optimizations."""
    MemoryOptimizer.clear_memory()

def get_optimal_batch_size(model, tokenizer, max_length: int = 2048) -> int:
    """Get optimal batch size for current memory constraints."""
    return MemoryOptimizer.get_optimal_batch_size(model, tokenizer, max_length)

def create_memory_callback(log_interval: int = 100) -> MemoryCallback:
    """Create a memory monitoring callback for training."""
    monitor = MemoryMonitor()
    return MemoryCallback(monitor, log_interval)
