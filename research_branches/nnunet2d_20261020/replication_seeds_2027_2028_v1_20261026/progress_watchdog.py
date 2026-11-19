"""CPU-reviewed watchdog proposal; no NAS mtime or wall-clock comparison."""
class ProgressWatchdog:
    def __init__(self,timeout_seconds=1800):
        if timeout_seconds<=0:raise ValueError('positive timeout required')
        self.timeout=timeout_seconds;self.last_token=None;self.last_progress_mono=None;self.last_poll_mono=None
    def observe(self,token,now_monotonic):
        if self.last_poll_mono is not None and now_monotonic<self.last_poll_mono:raise ValueError('monotonic clock moved backward')
        self.last_poll_mono=now_monotonic
        advanced=self.last_token!=token
        if self.last_progress_mono is None or advanced:self.last_token=token;self.last_progress_mono=now_monotonic
        stalled=now_monotonic-self.last_progress_mono
        return {'stop':stalled>=self.timeout,'counter_advanced':advanced,'stall_monotonic_seconds':stalled,'last_token':token,'decision_monotonic':now_monotonic,'uses_NAS_mtime':False,'uses_wall_clock':False}

def train_progress_token(budget):
    return tuple(budget[k] for k in ['epoch','phase','attempts','success','skip','fatal','unknown','train_forwards','backwards','optimizer_calls','monitor_attempts','monitor_complete'])

def inference_progress_token(budget):
    return (budget['forwards'],sum(x['state']=='COMPLETE' for x in budget['images'].values()))
