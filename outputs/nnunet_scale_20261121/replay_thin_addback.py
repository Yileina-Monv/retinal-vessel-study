"""Replay one pre-specified mask combination; use the same throttling as the audit."""
from thin_recall_audit import root, old, features, read, np, Image, tarfile, io, time, json


def cross(mask):
    expanded = mask.copy()
    expanded[1:] |= mask[:-1]
    expanded[:-1] |= mask[1:]
    expanded[:,1:] |= mask[:,:-1]
    expanded[:,:-1] |= mask[:,1:]
    return expanded


def score(mask, thin, ring, vessel, far, fov):
    hit = cross(mask)
    tp = int((mask & vessel).sum())
    fp = int((mask & ~vessel & fov).sum())
    return dict(thin_d1=int((hit & thin).sum()), thin_d0=int((mask & thin).sum()),
                near_fp=int((mask & ~vessel & ring).sum()), fp=fp,
                dice=2*tp/(2*tp+fp+int((vessel & ~mask).sum())),
                precision=tp/(tp+fp), far_recall=float((mask & far).sum()/far.sum()))


def run():
    rows = []
    with tarfile.open(old/"Train.tar", "r:") as source:
        for fold in range(3):
            folder=root/"eval_teacher"/f"fold{fold}"
            for name in read((old if fold==0 else old/f"fold_{fold}")/"split.json")["dev"]:
                start=time.process_time()
                label=np.array(Image.open(io.BytesIO(source.extractfile(f"Train/Labels/{name}_vessel.png").read())))
                vessel,fov,ring,skeleton,_=features(label)
                thin=ring & skeleton
                far=vessel & ~ring & (label!=127)
                for seed in (2027,2028,2029):
                    with np.load(folder/str(seed)/"correction"/(name+".npz")) as z:
                        native,dual=(np.unpackbits(z[k+"_m0matched"]).reshape(label.shape).view(bool) for k in ("native","dual"))
                    modified=dual | (native & cross(dual))
                    rows.append(dict(image=name,seed=seed,thin_den=int(thin.sum()),ring_area=int(ring.sum()),
                        fov_area=int(fov.sum()),base=score(dual,thin,ring,vessel,far,fov),
                        candidate=score(modified,thin,ring,vessel,far,fov)))
                time.sleep(max(.3,5*(time.process_time()-start)))
    assert len(rows)==162
    (root/"thin_addback_replay.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")

if __name__=="__main__":
    run()
