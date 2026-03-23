#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Jupyter GPU Initialization - Specifically for notebook environments
This handles the environment variable setup that works reliably in Jupyter kernels
"""

import os
import torch
import gc

def force_gpu_setup():
    """Force GPU setup with environment variables that work in Jupyter"""
    
    print("🔧 Force-setting GPU environment for Jupyter kernel...")
    
    # These environment variables work better in Jupyter
    gpu_env_vars = {
        'HSA_OVERRIDE_GFX_VERSION': '11.5.1',
        'HIP_VISIBLE_DEVICES': '0',
        'ROCR_VISIBLE_DEVICES': '0',
        'AMD_SERIALIZE_KERNEL': '1',  # 0 or 1 only
        'TORCH_USE_HIP_DSA': '1',
        'HIP_LAUNCH_BLOCKING': '1',
        'PYTORCH_HIP_ALLOC_CONF': 'max_split_size_mb:128',
        'HIP_FORCE_DEV_KERNARG': '1',  # Additional HIP setting
        'HSA_ENABLE_SDMA': '0'  # Disable SDMA for stability
    }
    
    # Force set environment variables
    for key, value in gpu_env_vars.items():
        os.environ[key] = str(value)
        print(f"  ✓ {key}={value}")
    
    # Clear any existing CUDA context
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    return test_gpu()

def test_gpu():
    """Test GPU with gfx1151 architecture"""
    
    print(f"\n📊 PyTorch version: {torch.__version__}")
    print(f"🔍 CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, using CPU")
        return torch.device("cpu")
    
    print(f"\n🧪 Testing GPU with gfx1151 architecture...")
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        device = torch.device("cuda")
        
        # Test basic tensor operations
        print("  🔸 Testing basic tensor...")
        x = torch.tensor([1.0, 2.0], device=device)
        y = x + 1.0
        
        # Test random tensor (the previously problematic operation)
        print("  🔸 Testing randn...")
        z = torch.randn(2, 2, device=device)
        w = z.sum()
        
        # Test matrix operations
        print("  🔸 Testing matrix ops...")
        a = torch.randn(3, 3, device=device)
        b = a @ a.T
        
        print(f"  ✅ GPU test SUCCESS!")
        print(f"     Basic result: {y.cpu().numpy()}")
        print(f"     Randn sum: {w.item():.4f}")
        print(f"     Matrix shape: {b.shape}")
        
        return device
        
    except Exception as e:
        print(f"  ❌ GPU test failed: {str(e)[:100]}...")
        print("  🔄 Falling back to CPU")
        return torch.device("cpu")

def safe_cuda_operation(operation_func, *args, **kwargs):
    """Safely execute a CUDA operation with fallback"""
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        print(f"⚠️  CUDA operation failed: {e}")
        # Try on CPU
        if 'device' in kwargs:
            kwargs['device'] = torch.device('cpu')
        return operation_func(*args, **kwargs)

# Monkey patch for safety
def safe_randn(*args, **kwargs):
    """Safe randn with automatic CPU fallback"""
    return safe_cuda_operation(torch.randn, *args, **kwargs)

def safe_randn_like(tensor, **kwargs):
    """Safe randn_like with automatic CPU fallback"""
    return safe_cuda_operation(torch.randn_like, tensor, **kwargs)

# Export safe functions
torch.randn_safe = safe_randn
torch.randn_like_safe = safe_randn_like

def get_best_device():
    """Get the best working device for this environment"""
    return force_gpu_setup()

if __name__ == "__main__":
    device = get_best_device()
    print(f"\n🎯 Best device: {device}") 