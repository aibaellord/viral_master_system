# setup_gpu.py
import os
import subprocess
import json

def detect_gpu():
    """Detect available GPUs and return configuration."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if has_cuda else 0
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if has_cuda else []
        
        return {
            "has_cuda": has_cuda,
            "gpu_count": gpu_count,
            "gpu_names": gpu_names,
            "cuda_version": torch.version.cuda if has_cuda else None,
            "cudnn_version": torch.backends.cudnn.version() if has_cuda else None
        }
    except ImportError:
        return {"has_cuda": False, "error": "PyTorch not installed"}
    except Exception as e:
        return {"has_cuda": False, "error": str(e)}

def configure_gpu(gpu_info, optimize_for_rtx=False):
    """Generate appropriate configuration file based on detected GPU."""
    config = {
        "use_gpu": gpu_info["has_cuda"],
        "device_count": gpu_info["gpu_count"],
        "precision": "mixed" if optimize_for_rtx else "float32",
        "batch_size": 32 if optimize_for_rtx else 16,
        "parallel_processing": True if gpu_info["gpu_count"] > 1 else False,
        "memory_optimization": optimize_for_rtx,
        "rtx_specific_optimizations": optimize_for_rtx
    }
    
    # Save configuration to file
    with open("config/gpu_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"GPU configuration saved. RTX optimizations: {'enabled' if optimize_for_rtx else 'disabled'}")
    return config

if __name__ == "__main__":
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    # Detect GPU
    gpu_info = detect_gpu()
    print(f"GPU Detection Results: {json.dumps(gpu_info, indent=2)}")
    
    # Check for RTX GPUs
    has_rtx = any("RTX" in gpu_name for gpu_name in gpu_info.get("gpu_names", []))
    
    # Ask user if they want to enable RTX optimizations
    rtx_optimize = False
    if has_rtx:
        response = input("RTX GPU detected. Enable RTX-specific optimizations? (y/n): ")
        rtx_optimize = response.lower() == 'y'
    
    # Configure GPU settings
    configure_gpu(gpu_info, rtx_optimize)

