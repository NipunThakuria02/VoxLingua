"""Immutable locked-test evaluation, reload parity and auditable actual predictions."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from voxlingua.audio import make_feature_copy
from voxlingua.features import FEATURE_NAMES, extract_features, validate_feature_columns
from voxlingua.distortion import DistortionService, BASE_ORDER, aligned, predict_probabilities, baseline_predict
from voxlingua.schemas import CLASSES
from .train_models import read_partition
from .build_dataset import digest, write_csv
from .production_gate import source_audio
from .augment import derivative


def score(y,pred):
    per=precision_recall_fscore_support(y,pred,labels=range(6),zero_division=0)
    matrix=confusion_matrix(y,pred,labels=range(6))
    normalized=np.divide(matrix,matrix.sum(axis=1,keepdims=True),out=np.zeros_like(matrix,dtype=float),where=matrix.sum(axis=1,keepdims=True)>0)
    return {'macro_f1':float(np.mean(per[2])),'accuracy':float(accuracy_score(y,pred)),
            'macro_precision':float(np.mean(per[0])),'macro_recall':float(np.mean(per[1])),
            'per_class':{label:{'precision':float(per[0][i]),'recall':float(per[1][i]),'f1':float(per[2][i]),'support':int(per[3][i])} for i,label in enumerate(CLASSES)},
            'confusion_matrix_counts':matrix.tolist(),'confusion_matrix_row_normalized':normalized.tolist()}


def timed(fn,n,warmups=3):
    for _ in range(warmups): fn()
    values=[]
    for _ in range(n):
        start=time.perf_counter();fn();values.append((time.perf_counter()-start)*1000)
    return {'median':float(np.median(values)),'p95':float(np.quantile(values,0.95)),'N':n,'warmups':warmups}


def compare_predictions(original_path, audit_path):
    """Exact IDs/targets/labels, documented reload tolerance for floating scores."""
    with Path(original_path).open(newline='',encoding='utf-8') as f: original=list(csv.DictReader(f))
    with Path(audit_path).open(newline='',encoding='utf-8') as f: audit=list(csv.DictReader(f))
    if len(original)!=len(audit) or not original: raise ValueError('Audit row count differs')
    maximum=0.0
    for old,new in zip(original,audit):
        if set(old)!=set(new): raise ValueError('Audit prediction schema differs')
        for key in ['sample_id','label_id','prediction_id']:
            if old[key]!=new[key]: raise ValueError('Audit IDs/targets/predicted labels differ')
        columns=[k for k in old if k not in ['sample_id','label_id','prediction_id']]
        a=np.array([float(old[k]) for k in columns]);b=np.array([float(new[k]) for k in columns])
        if not np.allclose(a,b,atol=1e-7,rtol=0): raise ValueError('Audit probabilities exceed documented tolerance')
        if len(a): maximum=max(maximum,float(np.max(np.abs(a-b))))
    return {'ids_targets_labels_exact':True,'probabilities_within_tolerance':True,'atol':1e-7,'rtol':0,'max_absolute_score_difference':maximum}


def unseen_severity_probe(service, originals, provenance, seed):
    """Apply the predeclared 25-dB probe to up to three held-out originals per language."""
    rows = []
    counts = {}
    unused = sorted((n for n in provenance['noise'] if n['split']=='test'), key=lambda n:n['noise_id'])
    rng = np.random.default_rng(np.random.SeedSequence([seed,25,2026]))
    raw_root = Path(provenance['raw_root'])
    if unused:
        for original in sorted(originals, key=lambda row:row['original_id']):
            language = original['fleurs_config']
            if counts.get(language,0)>=3:
                continue
            audio,_ = source_audio(raw_root,original)
            for index in rng.permutation(len(unused)):
                noise = unused[int(index)]
                noise_audio,_ = source_audio(raw_root,noise)
                try:
                    altered,meta = derivative(make_feature_copy(audio),'added_noise',rng,
                                               make_feature_copy(noise_audio),target_snr_db=25.0)
                except ValueError:
                    continue
                report = service.predict(extract_features(altered))
                rows.append({'original_id':original['original_id'],'language':language,
                             'noise_id':noise['noise_id'],'applied_snr_db':meta['applied_snr_db'],
                             'predicted_class':report['predicted_class']})
                counts[language] = counts.get(language,0)+1
                break
    if not rows:
        return {'status':'unavailable','reason':'No usable held-out 25-dB noise probes'}
    return {'status':'measured','target_snr_db':25.0,'support':len(rows),
            'correct_added_noise':sum(row['predicted_class']=='added_noise' for row in rows),
            'support_by_language':counts,'predictions':rows}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot',action='store_true')
    mode.add_argument('--production',action='store_true')
    p.add_argument('--audit-id')
    p.add_argument('--timing-repeats',type=int)
    a=p.parse_args(argv)
    kind='production' if a.production else 'pilot'
    if a.timing_repeats is None: a.timing_repeats=200 if a.production else 30
    warmups=20 if a.production else 3
    for identifier in [a.run_id]+([a.audit_id] if a.audit_id else []):
        if identifier in ('','.','..','pilot') or any(c in identifier for c in '/\\:'):
            raise ValueError('Run/audit IDs must be single path components')
    if a.timing_repeats<5: raise ValueError('At least five timing repetitions required')
    model_dir=Path('models/pilot' if a.pilot else 'models')/a.run_id
    folder=Path('results/pilot' if a.pilot else 'results')/a.run_id
    original_result=folder/'metrics.json'
    if original_result.exists() and not a.audit_id: raise ValueError('Original evaluation is sealed; use a new --audit-id for immutable reproduction')
    if a.audit_id:
        if not original_result.exists(): raise ValueError('Audit requires a completed original evaluation')
        folder=folder/'audits'/a.audit_id
        if folder.exists(): raise ValueError('Audit directory exists; never overwrite')
    service=DistortionService.load_run(model_dir,allow_pilot=a.pilot)
    config=json.loads((model_dir/'run_config.json').read_text())
    if config['run_kind']!=kind or config['run_id']!=a.run_id:
        raise ValueError('Model run identity does not match evaluation mode')
    root=Path(config['data_root'])
    provenance=json.loads((root/'manifests/provenance.json').read_text())
    if a.production:
        evidence={
            root/'manifests/provenance.json':config['data_provenance_sha256'],
            root/'manifests/test_lock.json':config['test_lock_sha256'],
            root/'manifests/train_samples.csv':config['train_samples_sha256'],
            root/'manifests/validation_samples.csv':config['validation_samples_sha256'],
            **{root/'features/fs-v1'/f'{split}.csv':stamp for split,stamp in config['partition_feature_sha256'].items()},
        }
        if provenance['run_kind']!='production' or any(digest(path)!=stamp for path,stamp in evidence.items()):
            raise ValueError('Production dataset changed after model fitting')
    # Reload parity occurs on validation before opening locked test features.
    Xv,yv,val_ids,val_meta=read_partition(root,'validation');lookup={i:j for j,i in enumerate(val_ids)}
    probe=np.load(model_dir/'parity_probe.npz',allow_pickle=False)
    selected=[lookup[str(i)] for i in probe['sample_ids']]
    actual=predict_probabilities(service.bases,service.meta,Xv[selected])
    for name,P in actual.items():
        if not np.allclose(P,probe[name],atol=1e-7,rtol=0): raise ValueError('Reload probability mismatch: '+name)
    if not np.array_equal(baseline_predict(service.baseline,Xv[selected])[0],probe['baseline_labels']): raise ValueError('Baseline reload label mismatch')
    lock=json.loads((root/'manifests/test_lock.json').read_text())
    if lock['run_kind']!=kind: raise ValueError('Test-lock kind does not match')
    if a.production and not a.audit_id:
        # Claim the test before reading any held-out bytes. Fail closed if the first run is interrupted.
        claim=Path('results/test-exposures')/(lock['test_samples_sha256']+'.json')
        claim.parent.mkdir(parents=True,exist_ok=True)
        with claim.open('x',encoding='utf-8') as handle:
            json.dump({'run_id':a.run_id,'test_lock_sha256':config['test_lock_sha256']},handle)
    if digest(root/'manifests/test_samples.csv')!=lock['test_samples_sha256'] or digest(root/'features/fs-v1/test.csv')!=lock['test_features_sha256']:
        raise ValueError('Locked test content changed')
    if a.audit_id:
        original=json.loads(original_result.read_text())
        if original['provenance']['artifact_manifest_hash']!=digest(model_dir/'artifact_manifest.json'):
            raise ValueError('Cannot reproduce altered model artifacts')
    with (root/'features/fs-v1/test.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);validate_feature_columns(reader.fieldnames[1:]);features=list(reader)
    with (root/'manifests/test_samples.csv').open(newline='',encoding='utf-8') as f: test_meta={r['sample_id']:r for r in csv.DictReader(f)}
    ids=[r['sample_id'] for r in features]
    if set(ids)!=set(test_meta) or len(ids)!=len(set(ids)): raise ValueError('Test sample-ID alignment failed')
    X=np.array([[r[n] for n in FEATURE_NAMES] for r in features],dtype=np.float32)
    y=np.array([int(test_meta[i]['label_id']) for i in ids])
    if not np.isfinite(X).all(): raise ValueError('Invalid test features')
    probabilities=predict_probabilities(service.bases,service.meta,X)
    predictions={name:P.argmax(axis=1) for name,P in probabilities.items()}
    predictions['threshold_baseline'],coverage=baseline_predict(service.baseline,X)
    folder.mkdir(parents=True,exist_ok=False)
    ids_hash=hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest()
    write_csv(folder/'sample_ids.csv',[{'sample_id':i} for i in sorted(ids)],['sample_id'])
    # One real reference recording for a disclosed, limited batch-1 timing probe.
    with (root/'manifests/originals.csv').open(newline='',encoding='utf-8') as f:
        originals=[r for r in csv.DictReader(f) if r['screening_status']=='kept' and r['split']=='test']
    original=originals[0]
    audio,_=source_audio(Path(provenance['raw_root']),original)
    x=make_feature_copy(audio)
    one=X[:1]
    def function(name,z):
        if name in BASE_ORDER: return aligned(service.bases[name],z)
        if name=='threshold_baseline': return baseline_predict(service.baseline,z)[0]
        if name=='soft_voting': return np.mean([aligned(service.bases[n],z) for n in BASE_ORDER],axis=0)
        return aligned(service.meta,np.hstack([aligned(service.bases[n],z) for n in BASE_ORDER]))
    models={}
    for name,pred in predictions.items():
        rows=[]
        for j,i in enumerate(ids):
            row={'sample_id':i,'label_id':int(y[j]),'prediction_id':int(pred[j])}
            if name in probabilities: row.update({label:float(probabilities[name][j,k]) for k,label in enumerate(CLASSES)})
            rows.append(row)
        prediction_path=folder/(name+'-predictions.csv');write_csv(prediction_path,rows)
        result=score(y,pred)
        classifier=timed(lambda:function(name,one),a.timing_repeats,warmups)
        end_to_end=timed(lambda:function(name,extract_features(x)[None,:]),a.timing_repeats,warmups)
        result.update(sample_ids_sha256=ids_hash,support_rows=len(ids),support_originals=len({test_meta[i]['original_id'] for i in ids}),
                      predictions_path=prediction_path.name,predictions_sha256=digest(prediction_path),
                      latency_ms={'classifier_median':classifier['median'],'classifier_p95':classifier['p95'],
                                  'feature_classify_median':end_to_end['median'],'feature_classify_p95':end_to_end['p95']},
                      train_seconds=config['training_seconds'].get(name),ci95_macro_f1=None,ci_not_run_reason='Optional bootstrap not run')
        if name=='threshold_baseline': result['coverage']=coverage
        models[name]=result
        print(f'{name}: macro F1={result["macro_f1"]:.4f}, n={len(y)} {kind} rows',flush=True)
    metrics={'schema_version':'eval-v1.1','run_id':a.run_id,'run_kind':kind,'created_utc':datetime.now(timezone.utc).isoformat(),'seed':config['seed'],
             'feature_schema_version':'fs-v1','feature_schema_hash':provenance['feature_schema_hash'],
             'dataset':{'sources':['google/fleurs',provenance['noise_corpus']],'noise_corpus':provenance['noise_corpus'],'noise_is_fallback':provenance['noise_is_fallback'],'licenses':provenance['licenses'],
                        'revisions':provenance.get('revisions') or {'fleurs':provenance['speech'][0]['source_revision'],'ESC-10':provenance['noise'][0]['revision']},'originals':provenance['counts'],
                        'languages':sorted({m['language'] for m in test_meta.values()}),'fallback_details':provenance.get('fallback_approval')},
             'provenance':{'code_commit':'no-git','config_hash':digest(model_dir/'run_config.json'),'artifact_manifest_hash':digest(model_dir/'artifact_manifest.json'),
                           'test_lock_hash':digest(root/'manifests/test_lock.json'),'data_provenance_path':str(root/'manifests/provenance.json'),
                           'data_provenance_sha256':digest(root/'manifests/provenance.json'),'citations':provenance['citations'],
                           'speaker_evidence':provenance['speaker_independence'],'original_noise_overlap_audit':'split-disjoint originals/content and noise parents; train OOF group/noise disjoint evidence stored'},
             'deployment':{'model':'stacking','frozen_before_test':True,'production_promotion':False},
             'sample_id_parity':{'sorted_ids_path':'sample_ids.csv','sorted_ids_sha256':ids_hash,'sample_count':len(ids),
                                 'original_count':len({test_meta[i]['original_id'] for i in ids}),'exact_sets_and_labels_equal':True},
             'reload_parity':{'probe_count':len(selected),'probe_ids_sha256':hashlib.sha256(json.dumps(probe['sample_ids'].tolist()).encode()).hexdigest(),
                              'class_support':{CLASSES[k]:int(np.sum(yv[selected]==k)) for k in range(6)},'atol':1e-7,'rtol':0,'probabilities_match':True,'baseline_labels_match':True},
             'timing_protocol':{'hardware':platform.platform()+' '+platform.processor(),'batch_size':1,'warmups':warmups,'timed_calls':a.timing_repeats,
                                'reduced_protocol_reason':'Pilot timing budget' if a.pilot else None,'feature_input_duration_s':len(x)/16000,'workers':2},
             'models':models,'robustness':{'not_run_reasons':{'severity_sweeps':'Optional extended severity sweep deferred'}},
             'limitations':provenance['limitations']+['Semi-synthetic test measures controlled distortions','Single-label conditions overlap in real audio','Test ranking does not select deployment','No bootstrap CI computed']+(['Pilot artifacts cannot be promoted'] if a.pilot else [])}
    if a.production:
        metrics['robustness']['unseen_severity']=unseen_severity_probe(service,originals,provenance,config['seed'])
    if a.audit_id:
        original=json.loads(original_result.read_text())
        parity={}
        for name in models:
            for field in ['macro_f1','accuracy','macro_precision','macro_recall','per_class','confusion_matrix_counts','sample_ids_sha256']:
                if models[name][field]!=original['models'][name][field]: raise ValueError('Deterministic audit metrics differ')
            parity[name]=compare_predictions(original_result.parent/original['models'][name]['predictions_path'],folder/models[name]['predictions_path'])
        metrics['audit_of']=str(original_result)
        metrics['audit_parity']=parity
    (folder/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'results':str(folder/'metrics.json'),'production_promotion':False,'audit':bool(a.audit_id)}),flush=True)


if __name__=='__main__': main()
