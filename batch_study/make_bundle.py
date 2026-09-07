"""Package public development inputs and compact real-device evidence; no uploads."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files={}
    def add(path, archive=None):
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        archive=archive or path.relative_to(ROOT).as_posix()
        if Path(archive).is_absolute() or '..' in Path(archive).parts:
            raise ValueError('Unsafe archive path')
        files[archive]=path
    for module in ('retinal_data','retinal_m0','retinal_batchstudy'):
        for path in (ROOT/module).glob('*.py'):
            add(path)
    for name in ('batch_study_v1.json','data_trial_v1.json'):
        add(ROOT/'configs'/name)
    add(ROOT/'data_preparation/selection_summary.json')
    add(ROOT/'data_preparation/试跑规则_v1.md')
    add(ROOT/'FIVES核心数据集下载与验证_2026-09-04.md')
    add(ROOT/'environment/README.md')
    for name in ('smoke_train','smoke_val','overfit'):
        manifest=ROOT/'data_preparation/manifests'/f'{name}.csv'
        add(manifest)
        with manifest.open(encoding='utf-8-sig',newline='') as handle:
            for row in csv.DictReader(handle):
                if row['source_split']!='train':
                    raise ValueError('Only original train inputs may be packaged')
                for role in ('image','label'):
                    path=(ROOT/row[f'{role}_path']).resolve()
                    if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest()!=row[f'{role}_sha256']:
                        raise ValueError('Input path/hash mismatch')
                    add(path)
    for pattern in ('*.txt','*.ps1','*.py'):
        for path in (ROOT/'environment').glob(pattern):
            add(path)
    for name in ('PROTOCOL.md','PORTABLE.md','DECISION_2026-09-05.md','benchmark_native_batch.py','compare_interactions.py'):
        add(ROOT/'batch_study'/name)
    add(ROOT/'batch_study/PORTABLE.md','README.md')
    add(ROOT/'tests/test_batch_study.py')
    evidence=ROOT/'runs/batch_study_v1/rtx5080'
    for name in ('hardware.json','config.json','binding.json','data_check.json','run_order.json'):
        add(evidence/name)
    for path in (evidence/'numerical').glob('*'):
        if path.is_file(): add(path)
    for pattern in ('seed_*/*/result.json','seed_*/*/eval_draw_*.json'):
        for path in evidence.glob(pattern): add(path)
    for path in evidence.glob('initial_seed_*.pt'):
        add(path, f'runs/batch_study_v1/rtx4070/{path.name}')
    for path in (ROOT/'batch_study/results/rtx5080').glob('*'):
        if path.is_file(): add(path)
    for name in ('native_batch_capacity_rtx5080.json','resume_verification_5080.json','numerical_review_5080.json'):
        add(ROOT/'batch_study'/name)
    destination=ROOT/'runs/batch_study_v1/transfer/retinal_batch_study_4070_bundle.zip'
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest={name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in files.items()}
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for name,path in sorted(files.items()): archive.write(path,name)
        archive.writestr('bundle_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None: raise ValueError('Archive CRC verification failed')
        for name,record in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=record['sha256']:
                raise ValueError('Archive member SHA mismatch')
    result={'archive':str(destination),'bytes':destination.stat().st_size,'files':len(files),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'crc_and_all_member_sha256_verified':True,'source_scope':'40 public original-train images with paired labels; no official test/external'}
    (ROOT/'batch_study/transfer_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
