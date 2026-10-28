"""Bounded native-process memory and whole-device NVML memory sampling."""
import threading


class MemorySampler:
    def __init__(self,pid,interval=0.1):
        self.pid=pid;self.interval=interval;self.stop_event=threading.Event()
        self.result={'pid':pid,'interval_seconds':interval,'samples':0,
                     'native_peak_working_set_bytes':0,'native_peak_commit_bytes':0,
                     'sampled_whole_device_used_peak_bytes':None,
                     'gpu_scope':'NVML whole device including desktop/other processes; sampled peak, not per-process GPU allocation',
                     'errors':[]}
        self.thread=threading.Thread(target=self._run,daemon=True)

    def start(self):self.thread.start()

    def finish(self):
        self.stop_event.set();self.thread.join(timeout=3)
        return self.result

    def _run(self):
        import psutil
        process=psutil.Process(self.pid);nvml=None;device=None
        try:
            import pynvml
            pynvml.nvmlInit();nvml=pynvml;device=nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as error:
            self.result['errors'].append('NVML: '+str(error))
        try:
            while not self.stop_event.is_set():
                try:
                    m=process.memory_info()._asdict()
                    self.result['native_peak_working_set_bytes']=max(self.result['native_peak_working_set_bytes'],m.get('peak_wset',m.get('rss',0)))
                    self.result['native_peak_commit_bytes']=max(self.result['native_peak_commit_bytes'],m.get('peak_pagefile',m.get('private',m.get('vms',0))))
                    self.result['samples']+=1
                except psutil.NoSuchProcess:break
                if nvml is not None:
                    try:
                        used=int(nvml.nvmlDeviceGetMemoryInfo(device).used)
                        self.result['sampled_whole_device_used_peak_bytes']=max(self.result['sampled_whole_device_used_peak_bytes'] or 0,used)
                    except Exception as error:
                        if len(self.result['errors'])<10:self.result['errors'].append('NVML sampling: '+str(error))
                self.stop_event.wait(self.interval)
        finally:
            if nvml is not None:nvml.nvmlShutdown()
