"""C1 adapter: identical batches and pairs with bounded verified geometry storage."""
from pathlib import Path
import time
from retinal_release.common import ROOT,digest,read
from retinal_c1.pilot import Batches
from retinal_c1.evaluation import geometry
from retinal_c1.audit import OUT as C1_OUT
from local_deployment.cached_data import CachedDataset,Snapshot


class CachedC1Batches(Batches):
    snapshot=None
    geometries=None
    @classmethod
    def configure(cls,keys):
        started=time.perf_counter()
        cls.snapshot=Snapshot(keys);cls.geometries={}
        for key in sorted(set(keys)):
            geo=geometry(key)
            # Training does not need the much larger foreground ownership arrays.
            cls.geometries[key]={k:geo[k] for k in ('pos','bg','bg_offsets')}
            for value in cls.geometries[key].values():value.flags.writeable=False
        return dict(cache_id=cls.snapshot.cache_id,verification_seconds=cls.snapshot.verification_seconds,
                    setup_seconds=time.perf_counter()-started,
                    geometry_bytes=sum(v.nbytes for g in cls.geometries.values() for v in g.values()))
    def _dataset(self):
        return CachedDataset('train',snapshot=self.snapshot,seed=2026,cache_size=0)
    def _geometry(self,key):
        return self.geometries[key]
