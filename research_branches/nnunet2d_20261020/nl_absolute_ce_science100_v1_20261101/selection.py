import numpy as np
def choose_groups(candidates, seed, attempt, crop):
    """Candidates already geometrically validated; private RNG never touches native RNG."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261031,seed,attempt,crop])))
    pool = sorted(candidates, key=lambda c: (c['native_y'],c['native_x']))
    selected = []
    for g in pool:
        negs = sorted(set(map(tuple,g['negatives'])))
        take = min(2,len(negs))
        if not take:
            raise ValueError('invalid candidate with no background')
        ids = sorted(rng.choice(len(negs),take,replace=False))
        selected.append({**g,'negatives':[list(negs[int(i)]) for i in ids]})
    return selected
