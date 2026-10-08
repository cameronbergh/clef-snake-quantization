"""Offline compact decision benchmark protocol/semantics/integrity checks."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from experiments.classifier_benchmark_v2_v1.common import cases,digest,predict,raw_predict,reconstruct,verify_freeze,ROOT
from experiments.classifier_benchmark_v2_v1.analysis import auc,holm,metrics

class ClassifierProtocol(unittest.TestCase):
    def test_complete_locked_cases_and_exact_upstream_question(self):
        rows=cases();self.assertEqual(len(rows),866);self.assertEqual(len({r['case_id'] for r in rows}),866);self.assertEqual(len({r['task_id'] for r in rows}),49)
        r=rows[0];self.assertEqual(r['expected'],'account');self.assertEqual(r['request']['state'],"I lost my phone and can't receive 2FA codes, so I'm locked out of my account.")
        self.assertEqual(list(r['request']),['model','state','questions']);self.assertEqual(list(r['request']['questions']),['decision']);self.assertEqual(list(r['request']['questions']['decision']['criteria']),['billing','tech','sales','account','other'])
    def test_score_uses_distribution_argmax_not_expected_score(self):
        q={'type':'score','criteria':['low','medium','high']};p={'0':.40,'1':.25,'2':.35};native={'answers':{'decision':reconstruct(q,p)}}
        self.assertEqual(native['answers']['decision']['score'],.95);self.assertEqual(predict(q,native),0);self.assertEqual(raw_predict(q,p),0)
    def test_level_order_tie_and_rounded_boundary_disclosure(self):
        q={'type':'score','criteria':['0','1']};p={'0':.49999,'1':.50001};native={'answers':{'decision':reconstruct(q,p)}}
        self.assertEqual(predict(q,native),0);self.assertEqual(raw_predict(q,p),1)
        q={'type':'noul'};p={'true':.49999,'false':.50001};native={'answers':{'decision':reconstruct(q,p)}}
        self.assertTrue(predict(q,native));self.assertFalse(raw_predict(q,p))
    def test_choice_native_tie_respects_question_insertion_order(self):
        q={'type':'choice','criteria':{'z':'Z','a':'A'}};p={'a':.5,'z':.5};native={'answers':{'decision':reconstruct(q,p)}};self.assertEqual(predict(q,native),'z')
    def test_boolean_gold_not_ordinal_and_all_primitives(self):
        for r in cases():
            self.assertEqual(type(r['expected']),{'noul':bool,'score':int,'choice':str}[r['primitive']])
    def test_probability_metrics_and_auc_ties(self):
        rows=[{'raw_probabilities':{'true':.8,'false':.2},'expected':True,'correct':True,'primitive':'noul','prediction':True,'raw_prediction':True},{'raw_probabilities':{'true':.2,'false':.8},'expected':False,'correct':True,'primitive':'noul','prediction':False,'raw_prediction':False}]
        m=metrics(rows);self.assertEqual(m['auc'],1);self.assertAlmostEqual(m['binary_brier'],.04);self.assertAlmostEqual(m['multiclass_brier'],.08);self.assertAlmostEqual(m['top_confidence_ece_10bins'],.2)
        tie=copy.deepcopy(rows);tie[0]['raw_probabilities']['true']=tie[1]['raw_probabilities']['true']=.5;self.assertEqual(auc(tie),.5)
        self.assertEqual(holm([.01,.04,.03,.5]),[.04,.09,.09,.5])
    def test_freeze_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'x').write_bytes(b'original');(p/'freeze.json').write_text(json.dumps({'files':{'x':hashlib.sha256(b'original').hexdigest()}}));verify_freeze(p)
            (p/'x').write_bytes(b'changed')
            with self.assertRaisesRegex(AssertionError,'freeze tamper'):verify_freeze(p)
    def test_fixed_budget_and_deterministic_schedule(self):
        p=json.loads((ROOT/'protocol.json').read_text());self.assertEqual(p['maximum_forward_calls'],6*(866+3));self.assertEqual(p['maximum_scored_calls'],6*866);self.assertEqual(p['maximum_warmup_calls'],18)
        self.assertIn('no automatic retries',p['errors']);self.assertIn('no download',p['storage']);self.assertIn('no globally locked',p['concurrency'])
if __name__=='__main__':unittest.main()

class ClassifierAuditIntegrity(unittest.TestCase):
    def test_complete_offline_evidence_and_native_tamper_rejected(self):
        import gzip,shutil
        from experiments.classifier_benchmark_publication_v1.audit import audit
        from experiments.classifier_benchmark_v2_v1.common import CONDITIONS,sha,write
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prep=root/'preparation';vendor=prep/'vendor/cases';vendor.mkdir(parents=True)
            for name in ['v2.toml','v2.json','hashes.json']:shutil.copyfile(ROOT/'vendor/cases'/name,vendor/name)
            rows=cases();requests=[]
            for r in rows:requests.append({**r,'input_tokens':200,'input_ids_sha256':'a'*64,'truncated':False})
            write(prep/'requests.json',requests)
            f={'condition_order':CONDITIONS,'files':{str(p.relative_to(prep)):sha(p) for p in prep.rglob('*') if p.is_file()}};f['freeze_content_sha256']=digest(f);write(prep/'freeze.json',f)
            warm=[next(r for r in rows if r['primitive']==k) for k in ['choice','noul','score']]
            schedule=[('warmup',r) for r in warm]+[('scored',r) for r in rows]
            for condition in CONDITIONS:
                records=[]
                for i,(phase,r) in enumerate(schedule,1):
                    q=r['request']['questions']['decision'];typ=q['type'];ls=['true','false'] if typ=='noul' else list(q['criteria']) if typ=='choice' else [str(x) for x in range(len(q['criteria']))]
                    probs={k:1/len(ls) for k in ls};response={'model':'clef-flash','answers':{'decision':reconstruct(q,probs)},'usage':{'input_tokens':200,'output_tokens':0}}
                    records.append({'condition':condition,'phase':phase,'inference_id':i,'case_id':r['case_id'],'content_sha256':r['content_sha256'],'request':r['request'],'response':response,'raw_probabilities':probs,'input_ids_sha256':'a'*64,'prediction':predict(q,response),'raw_prediction':raw_predict(q,probs),'latency_ms':1.,'process_peak_rss_bytes':1000000})
                p=root/'records'/f'{condition}.jsonl.gz';p.parent.mkdir(exist_ok=True)
                with gzip.open(p,'wt') as file:file.write('\n'.join(json.dumps(r) for r in records)+'\n')
                write(root/'statuses'/f'{condition}.json',{'state':'complete','records':869,'forward_started':869,'forward_completed':869,'scored':866,'warmup':3})
            events=[{'event':'start','freeze_content_sha256':f['freeze_content_sha256']}]
            for c in CONDITIONS:events.extend([{'event':'worker_start','condition':c},{'event':'worker_complete','condition':c}])
            events.append({'event':'complete','status':{'scored_calls':5196,'warmup_calls':18}});(root/'supervisor.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
            def rehash():write(root/'hashes.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p!=root/'hashes.json'})
            rehash();result,_=audit(root);self.assertEqual(result['forward_completed'],5214)
            p=root/'records'/f'{CONDITIONS[0]}.jsonl.gz'
            with gzip.open(p,'rt') as file:records=[json.loads(l) for l in file]
            records[3]['response']['answers']['decision']['choice']='tampered'
            with gzip.open(p,'wt') as file:file.write('\n'.join(json.dumps(r) for r in records)+'\n')
            rehash()
            with self.assertRaisesRegex(AssertionError,'native answer reconstruction'):audit(root)
