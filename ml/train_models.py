"""Actual CART/RF/AdaBoost/XGBoost, grouped OOF stacking and equal voting."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import importlib.metadata
import math
import time
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from voxlingua.features import FEATURE_NAMES, feature_schema, validate_feature_columns
from voxlingua.schemas import CLASSES
from voxlingua.distortion import aligned, predict_probabilities
from .baseline import fit as fit_baseline, predict as predict_baseline
from .build_dataset import digest

ORDER=('cart','random_forest','adaboost','xgboost')


def json_metadata(value):
    """Represent estimator nonfinite defaults as strings, never nonstandard JSON tokens."""
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,dict): return {k:json_metadata(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [json_metadata(v) for v in value]
    return value


def factories(seed):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError('XGBoost native runtime unavailable; all-four-model training is blocked. No algorithm substitution is permitted. '+str(exc)) from exc
    return {'cart':DecisionTreeClassifier(max_depth=8,min_samples_leaf=3,random_state=seed),
            'random_forest':RandomForestClassifier(n_estimators=200,max_features='sqrt',min_samples_leaf=2,n_jobs=2,random_state=seed),
            'adaboost':AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,random_state=seed),n_estimators=150,learning_rate=0.5,random_state=seed),
            'xgboost':XGBClassifier(objective='multi:softprob',n_estimators=300,max_depth=4,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,tree_method='hist',device='cpu',n_jobs=2,eval_metric='mlogloss',random_state=seed)}


def read_partition(root,split):
    if split not in ('train','validation'): raise ValueError('Trainer cannot load the locked test partition')
    path=root/'features/fs-v1'/f'{split}.csv'
    with path.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);validate_feature_columns(reader.fieldnames[1:]);rows=list(reader)
    with (root/'manifests'/f'{split}_samples.csv').open(newline='',encoding='utf-8') as f: metadata={r['sample_id']:r for r in csv.DictReader(f)}
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(metadata): raise ValueError('Feature/manifest ID mismatch')
    X=np.array([[r[n] for n in FEATURE_NAMES] for r in rows],dtype=np.float32)
    if not np.isfinite(X).all(): raise ValueError('Nonfinite feature cache')
    return X,np.array([int(metadata[i]['label_id']) for i in ids]),ids,[metadata[i] for i in ids]


def predict_all(bases,meta,X):
    return predict_probabilities(bases,meta,X)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot',action='store_true')
    mode.add_argument('--production',action='store_true')
    a=p.parse_args(argv)
    kind='production' if a.production else 'pilot'
    a.data=a.data or Path('data')/kind
    provenance_path=a.data/'manifests/provenance.json'
    provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance['run_kind']!=kind: raise ValueError('Dataset kind does not match training mode')
    if a.production:
        lock_path=a.data/'manifests/test_lock.json'
        lock=json.loads(lock_path.read_text(encoding='utf-8'))
        gate=provenance.get('production_gate',{})
        if lock.get('run_kind')!='production' or gate.get('status')!='metadata_gate_passed':
            raise ValueError('Production build and test-lock evidence are required')
        # Read the lock metadata only, never the test feature or manifest rows.
        if (Path('results/test-exposures')/(lock['test_samples_sha256']+'.json')).exists():
            raise ValueError('This production test is already exposed. Reserve a fresh test set.')
    X,y,ids,rows=read_partition(a.data,'train');Xv,yv,val_ids,val_rows=read_partition(a.data,'validation')
    groups=np.array([r['cv_group_id'] for r in rows]);folds=np.array([int(r['fold_id']) for r in rows])
    if set(folds)!={0,1,2}: raise ValueError('Three stored folds required')
    seed=20261115;started=time.perf_counter();oof={n:np.full((len(X),6),np.nan) for n in ORDER};evidence=[]
    for fold in range(3):
        fit=np.flatnonzero(folds!=fold);hold=np.flatnonzero(folds==fold)
        if set(groups[fit])&set(groups[hold]): raise ValueError('CV group leakage')
        if set(y[fit])!=set(range(6)): raise ValueError('Every base fitting fold must contain six classes')
        fit_noise={rows[i].get('noise_source_group_id') for i in fit if rows[i].get('noise_source_group_id')}
        hold_noise={rows[i].get('noise_source_group_id') for i in hold if rows[i].get('noise_source_group_id')}
        if fit_noise&hold_noise: raise ValueError('OOF noise source leakage')
        evidence.append({'fold':fold,'fit_ids':[ids[i] for i in fit],'heldout_ids':[ids[i] for i in hold],'cv_groups_disjoint':True,'noise_sources_disjoint':True})
        for name,model in factories(seed+fold).items():
            model.fit(X[fit],y[fit]);oof[name][hold]=aligned(model,X[hold])
            print(f'OOF fold {fold}: {name}',flush=True)
    if any(not np.isfinite(v).all() for v in oof.values()): raise ValueError('Missing OOF predictions')
    Z=np.hstack([oof[n] for n in ORDER]);meta=LogisticRegression(C=1,max_iter=1000).fit(Z,y)
    bases=factories(seed);train_seconds={}
    for name,model in bases.items():
        t=time.perf_counter();model.fit(X,y);train_seconds[name]=time.perf_counter()-t
    schema=feature_schema()
    if schema['sha256']!=provenance['feature_schema_hash']: raise ValueError('Extractor changed after building data')
    baseline=fit_baseline(X,y,Xv,yv,schema['sha256'])
    run=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-nogit-s'+str(seed)
    models_root=Path('models/pilot' if a.pilot else 'models')
    results_root=Path('results/pilot' if a.pilot else 'results')
    folder=models_root/run;folder.mkdir(parents=True,exist_ok=False)
    for name in ORDER:
        if name=='xgboost': bases[name].save_model(folder/'xgboost.json')
        else: joblib.dump(bases[name],folder/(name+'.joblib'))
    joblib.dump(meta,folder/'stacking_meta.joblib')
    config={'run_id':run,'run_kind':kind,'seed':seed,'data_root':str(a.data.resolve()),'deployment_model':'stacking','base_order':ORDER,
            'voting_weights':'equal','cv_folds':3,'training_seconds':train_seconds,'total_training_seconds':time.perf_counter()-started,
            'train_samples_sha256':digest(a.data/'manifests/train_samples.csv'),'validation_samples_sha256':digest(a.data/'manifests/validation_samples.csv'),
            'noise_is_fallback':provenance['noise_is_fallback'],'noise_corpus':provenance['noise_corpus'],'test_access_during_training':False,
            'base_parameters':{n:m.get_params()|({'estimator':str(m.estimator)} if n=='adaboost' else {}) for n,m in bases.items()}}
    if a.production:
        config['test_lock_sha256']=digest(lock_path)
        config['data_provenance_sha256']=digest(provenance_path)
        config['partition_feature_sha256']={s:digest(a.data/'features/fs-v1'/f'{s}.csv') for s in ('train','validation')}
    # Class-aware probe selection; append remaining deterministic IDs until the bounded count.
    chosen=[]
    for class_id in range(6):
        matches=np.flatnonzero(yv==class_id)
        if len(matches): chosen.append(int(matches[0]))
    for i in range(len(Xv)):
        if len(chosen)>=min(50,len(Xv)): break
        if i not in chosen: chosen.append(i)
    if not chosen: raise ValueError('Validation probe is empty')
    P=predict_all(bases,meta,Xv[chosen]);baseline_labels,_=predict_baseline(baseline,Xv[chosen])
    np.savez(folder/'parity_probe.npz',sample_ids=np.array([val_ids[i] for i in chosen]),**P,baseline_labels=baseline_labels)
    np.savez(folder/'oof_probabilities.npz',sample_ids=np.array(ids),**oof)
    # Independent sanity probes use validation only; report actual outcomes, no imposed accuracy target.
    shuffled=np.random.default_rng(seed).permutation(y)
    probe=RandomForestClassifier(n_estimators=50,n_jobs=2,random_state=seed).fit(X,shuffled)
    shuffle_f1=float(f1_score(yv,probe.predict(Xv),labels=range(6),average='macro',zero_division=0))
    # Original-only metadata is identical across all six labels; never include noise IDs or generator fields.
    def metadata_matrix(metadata_rows):
        # Acquisition metadata lacks original_id; use separately persisted original manifest.
        with (a.data/'manifests/originals.csv').open(newline='',encoding='utf-8') as f: originals={o['original_id']:o for o in csv.DictReader(f)}
        languages=sorted({o['fleurs_config'] for o in originals.values()})
        return np.array([[languages.index(originals[r['original_id']]['fleurs_config']),float(originals[r['original_id']]['duration_s']),int(originals[r['original_id']]['source_sample_rate']),int(originals[r['original_id']]['num_samples'])] for r in metadata_rows])
    Xm=metadata_matrix(rows);Xmv=metadata_matrix(val_rows)
    metadata_probe=RandomForestClassifier(n_estimators=50,n_jobs=2,random_state=seed).fit(Xm,y)
    metadata_f1=float(f1_score(yv,metadata_probe.predict(Xmv),labels=range(6),average='macro',zero_division=0))
    documents={'class_mapping.json':{'classes':list(CLASSES)},'feature_schema.json':schema,'baseline.json':baseline,
               'abstention.json':{'enabled':False,'reason':'Validation abstention tuning deferred'},'run_config.json':config,
               'oof_evidence.json':evidence,'leakage_probes.json':{'label_shuffle_validation_macro_f1':shuffle_f1,'metadata_only_validation_macro_f1':metadata_f1,
               'metadata_allowlist':['original_language','original_duration','source_rate','source_length'],'validation_samples':len(Xv),'interpretation':'Sanity evidence only; elevated scores require investigation, chance macro F1 is not a fixed guarantee'}}
    for name,obj in documents.items(): (folder/name).write_text(json.dumps(json_metadata(obj),indent=2,default=str,allow_nan=False),encoding='utf-8')
    manifest={'run_kind':kind,'python':platform.python_version(),'code_commit':'no-git',
              'library_versions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','librosa','scikit-learn','xgboost','joblib']},
              'files':{f.name:{'sha256':digest(f),'size':f.stat().st_size} for f in folder.iterdir() if f.is_file()}}
    (folder/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2))
    results_root.mkdir(parents=True,exist_ok=True)
    (results_root/(run+'-trained.json')).write_text(json.dumps({'run_id':run,'run_kind':kind,'model_dir':str(folder),'status':'trained_not_evaluated','production_promotion':False},indent=2))
    print(json.dumps({'run_id':run,'models':str(folder),'probe_samples':len(chosen),'test_access':False,'promoted':False}),flush=True)


if __name__=='__main__': main()
