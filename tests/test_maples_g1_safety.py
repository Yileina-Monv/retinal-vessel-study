"""Isolated CPU fixtures; never touch a production release or ledger."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from retinal_m0.maples_g1_safety import *


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.path=self.root/'ledger.json';self.ids=[str(i) for i in range(32)]
        atomic_json(self.path,{'schema':1,'cap':32,'combinations':{},'autograd_calls':0,'optimizer_step_attempts':0})
        self.ledger=Ledger(self.path,'binding',self.ids)

    def tearDown(self):self.tmp.cleanup()
    def initialize(self):self.ledger.initialize_empty_legacy(self.root/'backup.json')

    def authority(self):
        binding={'schema':2,'source':{'a':'old'},'phase':'G1_ONLY'}
        ready=self.root/'binding.json';release=self.root/'release.json';atomic_json(ready,binding)
        receipt=self.root/'cpu_receipt.json';evidence={}
        for i in range(3):
            p=self.root/f'evidence_{i}.json';atomic_json(p,{'fixture':True});evidence[str(p)]=file_sha(p)
        atomic_json(receipt,{'status':'CPU_VALIDATED_GPU_HELD','binding_sha256':canonical_sha(binding),
            'binding_file_sha256':file_sha(ready),'tests_passed':True,'preflight_passed':True,'evidence':evidence})
        doc={'phase':'G1_ONLY','authorized':True,'gpu_launch_authorized':True,'max_forwards':32,
             'parameter_updates':0,'gpu_uuid':GPU_UUID,'digest_algorithm':DIGEST_ALGORITHM,
             'binding_sha256':canonical_sha(binding),'binding_file_sha256':file_sha(ready),
             'cpu_review_receipt_sha256':file_sha(receipt),
             'ledger_path':str(self.path.resolve()),'release_id':'isolated-test'}
        atomic_json(release,doc)
        state={'state':'RELEASED','phase':'G1_ONLY','release_id':doc['release_id'],
               'binding_sha256':canonical_sha(binding),'release_file_sha256':file_sha(release)}
        text=lambda s:START+'\n'+json.dumps(s)+'\n'+END
        args={'status_text':text(state),'release_path':release,'binding_path':ready,
              'current_binding':binding,'ledger_path':self.path,'review_receipt_path':receipt}
        return args,state,doc,text

    def test_exact_current_release(self):
        args,_,_,_=self.authority();self.assertTrue(verify_authority(**args)['authorized'])

    def test_history_does_not_override_hold(self):
        args,state,_,text=self.authority();state['state']='HOLD'
        args['status_text']=text(state)+'\nOLD GPU_RELEASED true G1_ONLY\n'
        with self.assertRaises(GateError):verify_authority(**args)

    def test_missing_or_duplicate_current_block(self):
        args,_,_,_=self.authority()
        for status in ('historic release true',args['status_text']*2):
            with self.subTest(status=status):
                args2={**args,'status_text':status}
                with self.assertRaises(GateError):verify_authority(**args2)

    def test_all_critical_release_fields_rejected(self):
        args,state,doc,text=self.authority()
        for key in ('phase','authorized','gpu_launch_authorized','max_forwards','parameter_updates',
                    'gpu_uuid','digest_algorithm','binding_sha256','binding_file_sha256',
                    'cpu_review_receipt_sha256','ledger_path','release_id'):
            bad={**doc,key:'wrong'};atomic_json(args['release_path'],bad)
            state['release_file_sha256']=file_sha(args['release_path']);args['status_text']=text(state)
            with self.subTest(field=key),self.assertRaises(GateError):verify_authority(**args)

    def test_raw_release_and_current_digest_binding(self):
        args,_,_,_=self.authority();args['current_binding']['source']['a']='changed'
        with self.assertRaises(GateError):verify_authority(**args)
        args,_,doc,_=self.authority();atomic_json(args['release_path'],{**doc,'extra':'tampered'})
        with self.assertRaises(GateError):verify_authority(**args)

    def test_binding_file_bytes_and_symlink(self):
        args,state,doc,text=self.authority()
        args['binding_path'].write_text(json.dumps(args['current_binding']))
        with self.assertRaises(GateError):verify_authority(**args)
        args,_,_,_=self.authority();link=self.root/'link.json';link.symlink_to(args['binding_path'])
        args['binding_path']=link
        with self.assertRaises(GateError):verify_authority(**args)

    def test_snapshot_and_constructor_are_read_only(self):
        self.initialize();self.ledger.reserve('0');before=self.path.read_bytes()
        Ledger(self.path,'binding',self.ids);ledger_snapshot(self.path)
        self.assertEqual(before,self.path.read_bytes())
        with self.assertRaises(GateError):self.ledger.recover_dead_owner()
        self.assertEqual(before,self.path.read_bytes())

    def test_confirmed_dead_owner_recovery_keeps_charge(self):
        self.initialize();self.ledger.reserve('0')
        with patch('retinal_m0.maples_g1_safety.owner_alive',return_value=False):self.ledger.recover_dead_owner()
        state=ledger_snapshot(self.path);self.assertEqual(state['combinations']['0']['status'],'failed')
        with self.assertRaises(GateError):self.ledger.reserve('1')

    def test_32_cap_and_second_forward_blocked_before_body(self):
        self.initialize();body_calls=0
        for cid in self.ids:
            self.ledger.reserve(cid);self.ledger.before_forward(cid);body_calls+=1
            if cid=='31':
                with self.assertRaises(GateError):
                    self.ledger.before_forward(cid);body_calls+=1
                self.ledger.finish(cid,error='blocked extra forward')
            else:self.ledger.finish(cid)
        state=ledger_snapshot(self.path)
        self.assertEqual(body_calls,32);self.assertEqual(state['forward_starts'],32)
        self.assertEqual(state['blocked_forward_attempts'],1)
        self.assertEqual(sum(v['status']=='failed' for v in state['combinations'].values()),1)
        with self.assertRaises(GateError):self.ledger.reserve('33')
        with self.assertRaises(GateError):self.ledger.reserve('0')

    def test_failure_before_forward_is_charged_and_no_retry(self):
        self.initialize();self.ledger.reserve('0');self.ledger.finish('0',error='input failure')
        state=ledger_snapshot(self.path);self.assertEqual(len(state['combinations']),1)
        self.assertEqual(state['forward_starts'],0)
        for cid in ('0','1'):
            with self.assertRaises(GateError):self.ledger.reserve(cid)

    def test_optimizer_blocked_and_logged(self):
        self.initialize();self.ledger.reserve('0')
        with self.assertRaises(GateError):self.ledger.prohibit_optimizer('0')
        self.assertEqual(ledger_snapshot(self.path)['optimizer_step_attempts'],1)

    def test_autograd_count_persists_and_caps_four(self):
        self.initialize();self.ledger.reserve('0');self.ledger.before_forward('0')
        for _ in range(4):self.ledger.autograd('0')
        with self.assertRaises(GateError):self.ledger.autograd('0')
        self.assertEqual(ledger_snapshot(self.path)['autograd_calls'],4)

    def test_missing_malformed_or_nonempty_legacy_ledger_refused(self):
        missing=self.root/'missing.json'
        with self.assertRaises(GateError):ledger_snapshot(missing)
        old=read_json(self.path);old['combinations']={'0':{}};atomic_json(self.path,old)
        with self.assertRaises(GateError):self.initialize()
        self.assertFalse((self.root/'backup.json').exists())

    def test_no_implicit_reinitialization_or_wrong_binding(self):
        self.initialize();before=self.path.read_bytes()
        with self.assertRaises(GateError):self.ledger.initialize_empty_legacy(self.root/'another.json')
        with self.assertRaises(GateError):Ledger(self.path,'changed',self.ids).reserve('0')
        self.assertEqual(before,self.path.read_bytes())

    def test_exclusive_lock(self):
        lock=self.root/'exclusive.lock'
        with exclusive_lock(lock):
            with self.assertRaises(GateError):
                with exclusive_lock(lock):pass

    def test_nested_nonfinite_and_failure_evidence(self):
        for v in (float('nan'),float('inf'),float('-inf')):
            metrics={'layers':[{'gradient':v}]}
            with self.assertRaises(GateError):finite_tree(metrics)
            safe=safe_evidence(metrics);atomic_json(self.root/'failure.json',safe)
            self.assertIn('NONFINITE',safe['layers'][0]['gradient'])

    def test_duplicate_json_keys_refused(self):
        self.path.write_text('{"cap":32,"cap":64}')
        with self.assertRaises(GateError):read_json(self.path)

    def test_reviewed_cpu_evidence_cannot_be_substituted(self):
        args,_,_,_=self.authority()
        receipt=read_json(args['review_receipt_path']);p=Path(next(iter(receipt['evidence'])))
        atomic_json(p,{'changed':True})
        with self.assertRaisesRegex(GateError,'evidence changed'):verify_authority(**args)


if __name__=='__main__':unittest.main()
