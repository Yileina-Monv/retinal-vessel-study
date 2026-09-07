"""Self-only limits, decimal GB. No other process or machine setting is changed."""
import ctypes
from ctypes import wintypes
import os
import threading
import time
import subprocess
import psutil

LIMIT=12_000_000_000

def windows_memory_limit():
    if os.name!='nt':raise RuntimeError('This acceptance profile requires Windows')
    class Basic(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',wintypes.DWORD),
            ('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active',wintypes.DWORD),
            ('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[(n,ctypes.c_uint64) for n in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_limit',ctypes.c_size_t),('job_limit',ctypes.c_size_t),
            ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];kernel.CreateJobObjectW.restype=wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    job=kernel.CreateJobObjectW(None,None)
    if not job:raise ctypes.WinError(ctypes.get_last_error())
    info=Extended();info.basic.flags=0x100;info.process_limit=LIMIT
    if not kernel.SetInformationJobObject(job,9,ctypes.byref(info),ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.AssignProcessToJobObject(job,kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    return job  # Keep handle for process lifetime; no kill-on-close flag.

class Budget:
    def __init__(self):
        self.job=windows_memory_limit();self.process=psutil.Process();self.stop=threading.Event()
        self.failure=None;self.peak_rss=0;self.peak_private=0;self.peak_reserved=0;self.peak_allocated=0
        self.peak_device_used=0;self.cuda_enabled=False;self.last_device_sample=0
        self.started=time.perf_counter();self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def sample(self):
        m=self.process.memory_info();self.peak_rss=max(self.peak_rss,m.rss,getattr(m,'peak_wset',0))
        self.peak_private=max(self.peak_private,getattr(m,'private',m.vms))
        if m.rss>10_000_000_000 or getattr(m,'private',0)>10_000_000_000:
            self.failure='RAM soft stop at 10 GB';self.stop.set()
    def watch(self):
        while not self.stop.wait(.1):
            try:
                self.sample()
                if self.cuda_enabled and time.perf_counter()-self.last_device_sample>1:self.device_sample()
            except Exception as e:self.failure=str(e);self.stop.set()
    def device_sample(self):
        output=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
            text=True,timeout=5,creationflags=subprocess.CREATE_NO_WINDOW)
        used=int(output.strip())*2**20;self.peak_device_used=max(self.peak_device_used,used)
        self.last_device_sample=time.perf_counter()
        if used>11_000_000_000:self.failure='Device VRAM soft stop at 11 GB (includes other applications)';self.stop.set()
    def check(self):
        self.sample()
        if self.failure:raise MemoryError(self.failure)
    def cuda(self):
        import torch
        if torch.cuda.device_count()!=1:raise ValueError('One CUDA GPU required')
        # Allocator ceiling below user VRAM limit, and usable on an 8 GB device.
        total=torch.cuda.get_device_properties(0).total_memory
        self.cuda_allocator_limit=min(8_000_000_000,int(total*.75))
        torch.cuda.set_per_process_memory_fraction(self.cuda_allocator_limit/total)
        self.cuda_enabled=True;self.device_sample();self.check()
    def cuda_sample(self):
        import torch
        self.peak_reserved=max(self.peak_reserved,torch.cuda.max_memory_reserved())
        self.peak_allocated=max(self.peak_allocated,torch.cuda.max_memory_allocated())
        self.check()
    def finish(self):
        self.check();self.stop.set();self.thread.join()
        return dict(ram_hard_commit_limit_bytes=LIMIT,ram_soft_stop_bytes=10_000_000_000,
            peak_rss_bytes=self.peak_rss,peak_private_bytes=self.peak_private,
            cuda_allocator_limit_bytes=getattr(self,'cuda_allocator_limit',None),
            peak_cuda_reserved_bytes=self.peak_reserved,peak_cuda_allocated_bytes=self.peak_allocated,
            peak_device_vram_used_bytes=self.peak_device_used,
            seconds=time.perf_counter()-self.started)
