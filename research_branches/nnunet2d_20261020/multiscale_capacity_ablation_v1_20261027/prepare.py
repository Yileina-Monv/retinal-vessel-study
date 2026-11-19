"""Freeze approved protocol, CPU checks and paired real-Train diagnostic fixtures."""
import os, sys, json, time, datetime, copy, shutil, traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from common import *


def prepend_record(title, message, add_authority=None):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = ROOT / 'docs/.continuity_backups' / (stamp + '_nnunet_multiscale')
    backup.mkdir(parents=True, exist_ok=False)
    for name in ['STATUS.md', '项目全局状态与研究决策.md']:
        p = CODE / name; old = p.read_text(); shutil.copy2(p, backup / name)
        head, rest = old.split('\n', 1)
        new = head + '\n\n### ' + title + '\n\n' + message + '\n\n' + rest.lstrip('\n')
        if name == 'STATUS.md' and add_authority is not None:
            if any(m in old for m in MARKERS): raise RuntimeError('authority already exists')
            new += '\n' + MARKERS[0] + '\n' + json.dumps(add_authority, ensure_ascii=False) + '\n' + MARKERS[1] + '\n'
        temp = p.with_name(p.name + '.multiscale_prepend.pending')
        with temp.open('x', encoding='utf-8') as f: f.write(new); f.flush(); os.fsync(f.fileno())
        if p.read_text() != old: raise RuntimeError('concurrent project record change')
        os.replace(temp, p)
    return str(backup)


def freeze():
    STAGE.mkdir(exist_ok=False)
    for folder in ['cpu', 'engineering/local', 'engineering/multi', 'science/local', 'science/multi', 'analysis']:
        (STAGE / folder).mkdir(parents=True, exist_ok=False)
    original = read(N0 / 'protocol.json')
    q = {'phase': PHASE, 'date': '2026-10-27', 'seed': 2027, 'module_seed': 92027,
         'modes': ['local', 'multi'], 'GPU': GPU,
         'user_approval': 'Implement the plan. User approved the immediately preceding complete dual-GPU plan.',
         'classification': 'single-seed controlled architecture exploration on repeatedly used Dev28',
         'scope': 'CPU preparation, independently released engineering, then separate Sol review/release for science; no automatic transition',
         'official_Test_allowed': False, 'baseline': str(N0),
         'module': {'insertion': 'after decoder.stages.7, before final segmentation head',
                    'channels': [32, 8, 24, 32], 'parameters': 2304,
                    'branches': [[1, 1], [3, 1], [3, 'local=1,multi=2']],
                    'norm': 'InstanceNorm2d(eps=1e-5,affine=True,track_running_stats=False)',
                    'activation': 'LeakyReLU(.01,inplace=False)', 'conv_bias': False,
                    'initialization': 'KaimingNormal a=.01; final24->32 projection zero; torch CPU fork_rng seed92027',
                    'formula': 'x + project(concat(branches(reduce(x))))',
                    'meaning': 'same topology/weights/parameter count/MACs, only third-branch dilation differs; not a FR-UNet reproduction'},
         'engineering': {'attempt_cap_per_mode': 32, 'diagnostic_forward_cap_per_mode': 4,
                         'wall_cap_seconds_per_mode': 1800, 'memory_cap_MiB': 10240,
                         'continuous': 16, 'split': [8, 8]},
         'science': {'epochs': 100, 'attempts_per_epoch': 250, 'attempt_cap_per_mode': 25000,
                     'monitor_per_epoch': 50, 'monitor_cap_per_mode': 5000,
                     'batch': 2, 'patch': [1024, 1536], 'loss': 'unchanged official CE+Dice with unchanged deep supervision',
                     'optimizer': {'name': 'SGD', 'lr': .01, 'momentum': .99, 'nesterov': True,
                                   'weight_decay': 3e-5, 'gradient_clip_norm': 12},
                     'scheduler': 'Poly100 exponent.9, official epoch stepping', 'initialization': 'fresh seed2027 official backbone',
                     'producer_seeds': {'train': 2027, 'Train_monitor': 2028},
                     'AMP_consecutive_stop': 5, 'AMP_total_stop': 25, 'replace_skips': False,
                     'wall_cap': None, 'automatic_restart': False, 'native_queued_sampler_exact_resume': False,
                     'input_pairing': 'all25000 IDs/input/target digests and LR match existingN0; mismatch stops before forward',
                     'memory_cap_MiB': 10240},
         'evaluation': {'endpoint': 'unique final100epochs', 'threshold': .5, 'TTA': False,
                        'ensemble': False, 'postprocessing': False, 'primary': 'ordinary image-macro Dice',
                        'contrasts': ['N-Multi minus N0', 'N-Multi minus N-Local', 'N-Local minus N0'],
                        'success': 'both Multi-N0 and Multi-Local Dice95% lower>0; original strict protections also pass',
                        'guards': 'precision/far_recall95% lower>=0; fp_density95% upper<=0; report Recall/coarse/hard-clDice nonadverse means',
                        'all_metrics': 'existing31 metrics, fixed GT-derived effective IDs, denominators and NA',
                        'bootstrap': 'existing1000 whole-image PCG64(20261016) indices shared by effective n',
                        'interpretation': 'single seed, repeatedDev, not independent Test or cross-seed confirmation; negative delivery valid'},
         'train_ids': original['train_ids'], 'Dev_ids': original['Dev_ids'],
         'references': original['references'], 'expected_tiles_by_image': original['inference']['expected_tiles_by_image'],
         'stops': ['source/split/input/LR/initializer drift', 'nonfinite loss/model/optimizer or unhandled gradient',
                   'AMP five consecutive or25 total', 'OOM or reserved memory>10GiB',
                   'engineering/science cap reached', 'interruption/no native queue restoration', 'no auto retry'],
         'no_Dev_before_scientific_final': True, 'no_expansion': 'no new lambda/seed/position/loss/threshold/budget',
         'estimated_hours': {'baseline_measured_per_group': 12.27389896605222, 'two_card_wall_estimate': [12, 16]},
         'literature': ['http${LOCAL_PATH}',
                        'http${LOCAL_PATH}']}
    atomic(STAGE / 'protocol.json', q, new=True)
    text = '''# nnU-Net多尺度与等参数普通卷积：已批准协议V1

N0复用完整连续seed2027。N-Local在GPU0、N-Multi在GPU1，各从相同官方随机主干开始。
选择首个连续复现seed，不按效果选seed。原CE+Dice、深监督、Train110/Dev28、batch2和1024×1536不变。
最终32通道解码后添加32→8降维、1×1/3×3/3×3三分支、拼接24→32零投影的残差模块。
两组新增2304参数、同初始参数；唯一差异第三分支dilation1对2。独立CPU模块种子92027不扰动原随机流。

工程每组最多4零更新前向和32更新尝试，CONT16及独立8+8，每卡累计1800秒/10GiB。
工程执行器绝不自动进入科学训练。Sol必须读取实际结果后发布独立科学放行。
科学每组100epochs/25000尝试/5000 Train监控。AMP跳步占额不补，连续5或累计25停。
每250尝试/epoch完整保存，保留20/40/60/80/100；最终100/.5才Dev28，无TTA/后处理/最佳模型筛选。
全部25000实际训练输入/目标/ID/LR对账N0；历史Train监控仅有ID，无历史像素摘要，不声称其逐像素验证。
原生增强队列不能精确恢复；固定批工程8+8只验证已保存状态，不冒充原生队列可恢复。
中断保留费用和状态，无自动重启。工程权重不进入科学初始化。

主比较Multi−N0和Multi−Local：两项Dice图像配对95%下界均>0且原严格保护通过，才支持受保护的多尺度增益。
所有31指标、分母、NA、负面图像均保留。重复Dev单seed不证明独立泛化；不自动扩种子。
官方Test封存，旧冻结协议、历史负面结论和0/26、0/105人工记录不变。
工程/科学阶段分别绑定当前STATUS、协议、实现、CPU证据和专用release。
用户批准实施整个方案，科学启动必须在工程后由Sol另行审查放行，不能由工程脚本自动衔接。
'''
    (STAGE / 'PROTOCOL.md').write_text(text, encoding='utf-8')
    atomic(STAGE / 'AUTHORIZATION.json', {'user_message': 'Implement the plan.', 'approved_plan': 'nnU-Net 双卡多尺度模块对照实验',
            'CPU': True, 'engineering_requires_independent_release': True,
            'science_requires_separate_post_engineering_Sol_release': True,
            'automatic_phase_transition': False, 'Test': False}, new=True)
    state = {'phase': PHASE, 'stage_path': str(STAGE), 'protocol_sha256': sha(STAGE / 'protocol.json'),
             'user_approved_plan': True, 'engineering_released': False, 'science_released': False,
             'state': 'CPU_PREPARATION', 'GPU': GPU, 'seed': 2027,
             'next': 'CPU/source/input checks; separate Sol engineering release. No automatic science.', 'official_Test_allowed': False}
    backup = prepend_record('用户批准nnU-Net双卡多尺度受控实验：CPU准备中，GPU待独立放行',
        'Sol直接实施。复用连续seed2027 nnU-Net为N0；普通组GPU0、多尺度组GPU1，每组新增2304参数且仅第三分支dilation1/2不同。'
        '原CE+Dice及训练/评价口径不变。先CPU，再单独放行每组4零更新前向/32更新尝试工程；工程后Sol独立审查并另行放行两条100epochs，禁止自动衔接。'
        '科学各25000尝试/5000 Train监控、只final/.5 Dev28，不访问Test；所有旧实验和人工0记录保留。证据 '+str(STAGE)+'。', state)
    atomic(STAGE / 'record_backup.json', {'path': backup}, new=True)


def check_baseline():
    import numpy as np
    from retinal_m0.maples_branch_science_analysis import evaluate_image, metric_values
    from retinal_m0.maples_stratified_science_analysis import hard_cldice
    q = protocol(); end = read(N0 / 'analysis/endpoint_100epochs.json')
    pv = read(N0 / 'prediction_verification.json')['images']; maps = {r['filename']: r for r in pv}
    refs = read(q['references']['science_reference_manifest'])['images']; by = {}
    if [r['filename'] for r in refs] != q['Dev_ids']: raise RuntimeError('Dev ordering mismatch')
    for ref in refs:
        name = ref['filename']; path = N0 / 'dev_predictions' / (name + '.npy')
        if sha(path) != maps[name]['sha256'] or sha(ref['reference_npz']) != ref['reference_npz_sha256']:
            raise RuntimeError('baseline/reference hash drift')
        p = np.load(path, allow_pickle=False)
        if p.dtype != np.float32 or list(p.shape) != ref['native_shape'] or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise RuntimeError('baseline probability invalid')
        with np.load(ref['reference_npz'], allow_pickle=False) as z: arrays = {k: z[k].copy() for k in z.files}
        values = metric_values(evaluate_image(p, arrays, 'B2', threshold=.5))
        values['hard_cldice'] = hard_cldice((p >= .5) & arrays['FOV'].astype(bool), arrays['GT'], arrays['FOV'])['cldice']
        by[name] = values
    maxerr = 0
    for metric, names in end['fixed_ids_by_metric'].items():
        actual = float(np.mean([by[n][metric] for n in names])); want = end['absolute_means']['NNUNET'][metric]
        maxerr = max(maxerr, abs(actual - want))
        if abs(actual - want) > 1e-12: raise RuntimeError('N0 metric mismatch:' + metric)
    return {'images': 28, 'metrics': len(end['fixed_ids_by_metric']), 'max_abs_error': maxerr,
            'seed': 2027, 'ordinary_Dice': end['absolute_means']['NNUNET']['dice']}


def cpu_checks():
    import torch, numpy as np, unittest.mock as mock
    from fusion_model import DetailFusion, make_trainer, seed_cpu, legacy_digest, original_state
    from fast_observer import digest
    from engineering import Counter
    checks = {}
    torch.set_num_threads(2); torch.set_num_interop_threads(1); seed_cpu()
    state = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]): torch.manual_seed(92027); a = DetailFusion('local')
    with torch.random.fork_rng(devices=[]): torch.manual_seed(92027); b = DetailFusion('multi')
    checks['module_rng_preserved'] = torch.equal(state, torch.get_rng_state())
    checks['exact_parameter_match'] = all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    checks['exact2304_parameters_each'] = all(sum(p.numel() for p in m.parameters()) == 2304 for m in [a, b])
    checks['only_dilation_changed'] = a.branches[2][0].dilation == (1, 1) and b.branches[2][0].dilation == (2, 2)
    x = torch.randn(2, 32, 32, 48, requires_grad=True)
    for label, module in [('local', a), ('multi', b)]:
        y = module(x); checks[label + '_identity'] = torch.equal(x, y)
        y.square().mean().backward()
        checks[label + '_zero_projection_expected_gradient'] = module.project.weight.grad.norm() > 0 and all(
            p.grad is not None and p.grad.count_nonzero() == 0 for n, p in module.named_parameters() if not n.startswith('project.'))
        module.zero_grad(set_to_none=True)
        with torch.no_grad(): module.project.weight.fill_(1e-3)
        module(x).square().mean().backward()
        checks[label + '_finite_nonzero_internal_gradients'] = all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in module.parameters())
        module.zero_grad(set_to_none=True)
        checks[label + '_empty_features_finite'] = bool(torch.isfinite(module(torch.zeros_like(x))).all())
    seed_cpu(); t = make_trainer('local', torch.device('cpu'), STAGE / 'cpu/native_producer', BASE)
    checks['original_initializer_matches2027'] = legacy_digest(original_state(t.network)) == read(N0 / 'initialization.json')['model']
    checks['eight_deep_supervision_heads'] = len(t.network.decoder.seg_layers) == 8 and t.enable_deep_supervision
    checks['stateless_normalization'] = not any(True for _ in t.network.buffers())
    checks['SGD_and_Poly100'] = type(t.optimizer).__name__ == 'SGD' and t.lr_scheduler.max_steps == 100
    for scenario, value, foreground in [('empty_GT', 0., False), ('full_GT', 0., True), ('extreme_positive', 1000., False), ('extreme_negative', -1000., True)]:
        logits = [torch.full((2, 2, max(1, 128 // 2**i), max(1, 128 // 2**i)), value, requires_grad=True) for i in range(8)]
        targets = [torch.full((2, 1, v.shape[2], v.shape[3]), int(foreground), dtype=torch.long) for v in logits]
        loss = t.loss(logits, targets); loss.backward()
        checks['official_loss_' + scenario] = bool(torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in logits))
    c = Counter(STAGE / 'cpu/counter_test.json', 'cpu-only-no-GPU')
    c.initialize(); c.start_role('diag')
    denied = 0
    try: c.compute('optimizer')
    except RuntimeError: denied += 1
    for _ in range(4): c.reserve('diag'); c.compute('forward'); c.resolve()
    try: c.reserve('diag')
    except RuntimeError: denied += 1
    c.finish_role()
    for role, n in [('cont', 16), ('split_a', 8), ('split_b', 8)]:
        c.start_role(role)
        for _ in range(n):
            c.reserve('update'); c.compute('forward'); c.compute('backward'); c.compute('optimizer'); c.resolve()
        c.finish_role()
    try: c.reserve('update')
    except RuntimeError: denied += 1
    checks['production_counter_cap_and_forbidden_optimizer'] = denied == 3 and c.read()['successful_updates'] == 32
    d = Counter(STAGE / 'cpu/counter_failure_test.json', 'cpu-only-no-GPU'); d.initialize(); d.start_role('diag')
    for _ in range(4): d.reserve('diag'); d.compute('forward'); d.resolve()
    d.finish_role(); d.start_role('cont'); d.reserve('update'); d.compute('forward'); d.stop('simulated worker failure')
    checks['failed_reserved_attempt_not_refunded'] = d.read()['updates_reserved'] == 1 and d.read()['unknown_updates'] == 1
    checks['counter_restart_refused'] = False
    try: d.initialize()
    except RuntimeError: checks['counter_restart_refused'] = True
    checks['baseline_full31metric_reconciliation'] = check_baseline()
    from analysis import analyze
    fixture = analyze(fixture=True)
    checks['three_group_all31metric_zero_contrast_fixture'] = all(
        v['estimate'] == 0 for metric in fixture['metrics'].values() for v in metric.values())
    # Real official Train-only producer: freeze16 exact augmented batches, never invoke network.
    from seeded_runtime import install_seeded_worker_adapter
    from amp32_runner import bind_train_datasets
    seed_meta = install_seeded_worker_adapter(2027)
    train, monitor = bind_train_datasets(t, set(protocol()['train_ids']))
    loader = val = None; records = []
    directory = STAGE / 'cpu/fixed_train_batches'; directory.mkdir(exist_ok=False)
    expected = [json.loads(l) for l in (N0 / 'attempts.jsonl').read_text().splitlines()][:16]
    try:
        loader, val = t.get_dataloaders()
        for index, row in enumerate(expected):
            batch = next(loader)
            got = {'ids': list(map(str, batch['keys'])), 'input_sha256': digest(batch['data']), 'target_sha256': digest(batch['target'])}
            if any(got[k] != row[k] for k in got): raise RuntimeError('official CPU producer differs from N0 at batch' + str(index + 1))
            path = directory / ('batch_%02d.pt' % (index + 1)); save_torch(path, batch)
            records.append({'index': index + 1, 'path': str(path), 'sha256': sha(path), **got})
        atomic(STAGE / 'cpu/fixed_batches.json', {'source': 'actual official Train110 augmented producer, seed2027/2028',
                'batches': records, 'seed_pools': seed_meta, 'Train_ids': sorted(train.identifiers),
                'Train_monitor_ids': sorted(monitor.identifiers), 'model_forwards': 0}, new=True)
        checks['real16_batches_exact_N0'] = len(records) == 16
        checks['real_worker_seeds'] = seed_meta == [{'pool': 'train', 'worker_count': 1, 'seeds': [2027]},
                                                  {'pool': 'internal_train_monitor', 'worker_count': 1, 'seeds': [2028]}]
    finally:
        for item in [loader, val]:
            if hasattr(item, '_finish'): item._finish()
    checks['CUDA_not_initialized'] = not torch.cuda.is_initialized()
    checks = {k: v if isinstance(v, dict) else bool(v) for k, v in checks.items()}
    atomic(STAGE / 'cpu/checks.json', {'checks': checks, 'passed': all(bool(v) for v in checks.values()),
            'GPU_calls': 0, 'optimizer_updates': 0, 'Dev_model_inference': 0, 'Test_access': False}, new=True)
    if not all(bool(v) for v in checks.values()): raise RuntimeError('CPU check failed:' + repr(checks))


def bind_sources():
    old = read(BASE / 'replication_seed2027_2028_100epochs_v1_20261026/source_binding.json')
    for section in ['all_sources', 'Train_sources']:
        for p, h in old[section].items():
            if sha(p) != h: raise RuntimeError('accepted dependency changed:' + p)
    runtime = dict(old['runtime_sources']); runtime.update({str(p): sha(p) for p in BRANCH.glob('*.py')})
    all_sources = dict(old['all_sources']); all_sources.update(runtime)
    for path in [STAGE / 'PROTOCOL.md', STAGE / 'AUTHORIZATION.json', STAGE / 'cpu/checks.json',
                 STAGE / 'cpu/analysis_fixture/endpoint_100epochs.json']:
        all_sources[str(path)] = sha(path)
    references = {}
    for path in [N0 / n for n in ['protocol.json', 'initialization.json', 'training_completion.json', 'attempts.jsonl',
                                 'monitor.jsonl', 'prediction_verification.json', 'analysis/endpoint_100epochs.json']]: references[str(path)] = sha(path)
    q = protocol()
    for key in ['science_reference_manifest', 'Dev_RGB_manifest']:
        path = Path(q['references'][key]); references[str(path)] = sha(path)
    for row in read(q['references']['science_reference_manifest'])['images']:
        references[row['reference_npz']] = row['reference_npz_sha256']
        path = N0 / 'dev_predictions' / (row['filename'] + '.npy'); references[str(path)] = sha(path)
    fixtures = {str(p): sha(p) for p in (STAGE / 'cpu/fixed_train_batches').glob('*.pt')}
    fixtures[str(STAGE / 'cpu/fixed_batches.json')] = sha(STAGE / 'cpu/fixed_batches.json')
    b = {'phase': PHASE, 'protocol_sha256': sha(STAGE / 'protocol.json'), 'runtime_sources': runtime,
         'all_sources': all_sources, 'Train_sources': old['Train_sources'], 'reference_sources': references,
         'fixture_sources': fixtures, 'versions': old['versions']}
    b['binding_sha256'] = canonical(b); atomic(STAGE / 'source_binding.json', b, new=True)
    atomic(STAGE / 'CPU_REVIEW.json', {'status': 'CPU_ACCEPTED', 'binding_sha256': b['binding_sha256'],
           'checks_sha256': sha(STAGE / 'cpu/checks.json'), 'checks_count': len(read(STAGE / 'cpu/checks.json')['checks']),
           'real_fixed_batches': 16, 'GPU_used': False, 'science_released': False,
           'limitation': 'CPU/synthetic checks do not establish native GPU/AMP behavior or native queue restoration'}, new=True)
    from engineering import Counter
    for mode in GPU:
        key = canonical({'binding': b['binding_sha256'], 'mode': mode, 'phase': 'engineering'})
        Counter(STAGE / 'engineering' / mode / 'budget.json', key).initialize()
    rejections = []
    for mode in GPU:
        for phase in ['engineering', 'science', 'infer']:
            try: validate(mode, phase)
            except RuntimeError as e: rejections.append({'mode': mode, 'phase': phase, 'reason': str(e)})
            else: raise RuntimeError('GPU admitted without phase release')
    atomic(STAGE / 'cpu/authorization_denials.json', {'status': 'PASS', 'real_denials': rejections,
            'GPU_initialized': False}, new=True)
    cli = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    for script, args in [('engineering.py', []), ('engineering.py', ['--run-authorized', '--mode', 'local']),
                         ('science.py', ['--run-authorized', '--mode', 'multi'])]:
        result = subprocess.run([sys.executable, '-B', str(BRANCH / script)] + args,
                                cwd=CODE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.returncode == 0: raise RuntimeError('real unauthorized CLI accepted')
        cli.append({'script': script, 'args': args, 'returncode': result.returncode, 'output': result.stdout[-1600:]})
    atomic(STAGE / 'cpu/CLI_denials.json', {'status': 'PASS', 'cases': cli, 'GPU_used': False}, new=True)
    update_authority(state='CPU_ACCEPTED_AWAITING_SOL_ENGINEERING_RELEASE', binding_sha256=b['binding_sha256'],
                     CPU_review_sha256=sha(STAGE / 'CPU_REVIEW.json'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--freeze', action='store_true'); p.add_argument('--cpu-checks', action='store_true'); a = p.parse_args()
    if a.freeze: freeze()
    elif a.cpu_checks:
        try: cpu_checks(); bind_sources()
        except BaseException as exc:
            atomic(STAGE / 'CPU_FAILURE.json', {'error': repr(exc), 'traceback': traceback.format_exc(), 'GPU_used': False})
            raise
    else: p.error('explicit CPU operation required')
