"""Offline regressions for privacy, integrity and public receipt boundaries."""
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.kev4b_publication_v1 import audit as p
from experiments.kev4b_publication_v1.export import redact


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def write_journal(self, rows, *, torn=False):
        raw=b''.join(p.frozen.wire(r)+b'\n' for r in rows)
        with gzip.open(self.root/'supervisor.jsonl.gz','wb') as f:f.write(raw[:-1] if torn else raw)

    def test_private_plain_and_decoded_content_are_rejected(self):
        with self.assertRaises(p.frozen.EvidenceError):p.check_public({'path':'/Users/private/test'})
        raw=b'{"path":"/Users/private/test"}'
        row={'request_base64':base64.b64encode(raw).decode(),'request_sha256':p.frozen.digest(raw)}
        with self.assertRaises(p.frozen.EvidenceError):p.check_public(row)
        with self.assertRaises(p.frozen.EvidenceError):p.check_public({'worker_nonce':'fake-test-only'})
        with self.assertRaises(p.frozen.EvidenceError):p.check_public({'one_use_id':'fake-test-only'})

    def test_redaction_commits_only_known_identity_and_preserves_wire_bytes(self):
        from collections import defaultdict
        counts=defaultdict(int)
        raw=p.frozen.wire({'move':'left'})
        row={'one_use_id':'test-consumed-id','path':'/Volumes/TestAsset/hf/hub',
             'request_base64':base64.b64encode(raw).decode(),'request_sha256':p.frozen.digest(raw)}
        public=redact(row,asset_root='/Volumes/TestAsset',one_use_id='test-consumed-id',counts=counts)
        p.check_public(public)
        self.assertEqual(public['request_base64'],row['request_base64'])
        self.assertEqual(public['path'],p.ALIAS+'/hf/hub')
        self.assertEqual(public['one_use_id'],'sha256:'+p.frozen.digest(b'test-consumed-id'))
        with self.assertRaises(p.frozen.EvidenceError):redact(row,asset_root='/Volumes/TestAsset',one_use_id='other-id',counts=counts)

    def test_torn_and_noncontiguous_compressed_journals_fail(self):
        rows=[{'sequence':1,'event':'worker_exit','monotonic':1,'synthetic':False}]
        self.write_journal(rows,torn=True)
        with self.assertRaises(p.frozen.IncompleteEvidence):list(p.journal(self.root,'supervisor',{}))
        rows[0]['sequence']=2;self.write_journal(rows)
        with self.assertRaises(p.frozen.EvidenceError):list(p.journal(self.root,'supervisor',{}))

    def test_hash_inventory_rejects_extra_files_changed_bytes_and_escape(self):
        f=self.root/'data.json';f.write_text('{}')
        manifest=self.root/'SHA256SUMS';manifest.write_text(p.sha(f)+'  data.json\n')
        p.verify_hashes(self.root)
        f.write_text('{"changed":true}')
        with self.assertRaises(p.frozen.EvidenceError):p.verify_hashes(self.root)
        manifest.write_text(p.sha(f)+'  data.json\n');(self.root/'extra.json').write_text('{}')
        with self.assertRaises(p.frozen.EvidenceError):p.verify_hashes(self.root)
        manifest.write_text('a'*64+'  ../escape.json\n')
        with self.assertRaises(p.frozen.EvidenceError):p.verify_hashes(self.root)

    def supervisor_fixture(self):
        context=b'{"fixture":"explicit synthetic context validation stub"}'
        (self.root/'context-report.json').write_bytes(context)
        limits={'fixture':'synthetic, not campaign evidence'}
        approval={'approval_sha256':'a'*64,'one_use_id':'sha256:'+'b'*64,
                  'freeze_sha256':p.FREEZE_SHA,'limits':limits,'context_report_sha256':p.frozen.digest(context)}
        receipt={'schema_version':1,'publication_only':True,'execution_authorized':False,
                 'original_approval':approval,'original_claim_sha256':'c'*64,'deadline':100}
        (self.root/'authorization-receipt.json').write_text(json.dumps(receipt))
        engine=SimpleNamespace(approval={**approval,'claim_sha256':'c'*64},start_time=12,
                               first_native_load_time=15,end_time=20,finished=True)
        rows=[{'event':'supervisor_start','monotonic':10,'approval':approval,'claim_sha256':'c'*64,
               'freeze_sha256':p.FREEZE_SHA,'limits':limits,'deadline':100}]
        storage={'free_bytes':9*1024**3,'root_bytes':1,'evidence_bytes':1}
        for phase,t,rss,state in [('initial',11,1024,'running'),('periodic',16,2048,'running'),('final',21,None,'exited')]:
            rows.append({'event':'resource_sample','monotonic':t,'sample_phase':phase,'worker_state':state,
                         'current_rss_bytes':rss,'storage':storage.copy()})
        rows.append({'event':'worker_exit','monotonic':22,'returncode':0})
        for i,row in enumerate(rows,1):row.update(sequence=i,synthetic=False)
        return rows,engine,{'protocol':{'limits':limits}}

    def test_public_receipt_is_non_executable_and_requires_full_resource_lifecycle(self):
        rows,engine,prep=self.supervisor_fixture();self.write_journal(rows)
        with patch.object(p.frozen,'verify_context_report'):
            self.assertEqual(p.verify_supervisor(self.root,engine,prep),5)
            for mutate in (lambda r:r[-1].update(returncode=1),
                           lambda r:r[1].update(current_rss_bytes=None),
                           lambda r:r[3].update(sample_phase='periodic'),
                           lambda r:r[2]['storage'].update(free_bytes=1)):
                bad=deepcopy(rows);mutate(bad);self.write_journal(bad)
                with self.assertRaises(p.frozen.EvidenceError):p.verify_supervisor(self.root,engine,prep)
        receipt=json.loads((self.root/'authorization-receipt.json').read_text());receipt['execution_authorized']=True
        (self.root/'authorization-receipt.json').write_text(json.dumps(receipt));self.write_journal(rows)
        with self.assertRaises(p.frozen.EvidenceError):p.verify_supervisor(self.root,engine,prep)

    def test_derived_summary_detects_tampering(self):
        value={'n':3,'mean':2,'median':2,'min':1,'max':3};p.verify_summary(value,[1,2,3])
        value['mean']=20
        with self.assertRaises(p.frozen.EvidenceError):p.verify_summary(value,[1,2,3])

    def test_portable_import_loads_no_model_framework(self):
        code="import experiments.kev4b_publication_v1.audit; import sys; assert not any(n==p or n.startswith(p+'.') for n in sys.modules for p in ('torch','mlx','transformers','kev'))"
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
