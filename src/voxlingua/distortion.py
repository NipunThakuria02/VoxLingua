"""Trusted local ensemble loading/inference, with fail-closed production promotion."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from .features import FEATURE_NAMES, feature_schema
from .schemas import CLASSES, ServiceError

BASE_ORDER=('cart','random_forest','adaboost','xgboost')


def aligned(model,X):
    out=np.zeros((len(X),6));raw=model.predict_proba(X)
    for j,c in enumerate(model.classes_): out[:,int(c)]=raw[:,j]
    return out


def predict_probabilities(bases,meta,X):
    probabilities={n:aligned(bases[n],X) for n in BASE_ORDER}
    probabilities['soft_voting']=np.mean([probabilities[n] for n in BASE_ORDER],axis=0)
    probabilities['stacking']=aligned(meta,np.hstack([probabilities[n] for n in BASE_ORDER]))
    return probabilities


def baseline_predict(rules,X):
    output=np.zeros(len(X),dtype=int);fired=np.zeros(len(X),dtype=bool)
    for class_id in rules['precedence']:
        rule=rules['rules'][str(class_id)];masks=[]
        for trigger in rule['triggers']:
            x=X[:,FEATURE_NAMES.index(trigger['feature'])]
            masks.append(x>=trigger['threshold'] if trigger['operator']=='>=' else x<=trigger['threshold'])
        mask=np.logical_and.reduce(masks) if rule['combine']=='AND' else np.logical_or.reduce(masks)
        hit=mask&~fired;output[hit]=class_id;fired|=hit
    return output,float(fired.mean())


class DistortionService:
    def __init__(self,bases,meta,baseline,run_id):
        self.bases,self.meta,self.baseline,self.run_id=bases,meta,baseline,run_id

    @classmethod
    def load(cls,models_dir='models'):
        root=Path(models_dir).resolve();pointer=root/'current.json'
        if not pointer.exists(): raise ServiceError('E_MODEL_ARTIFACT_MISSING','No evaluated production run has been promoted.')
        promotion=json.loads(pointer.read_text(encoding='utf-8'))
        run_id=promotion['run_id']
        if not isinstance(run_id,str) or run_id in ('','.','..','pilot') or any(c in run_id for c in '/\\:'):
            raise ServiceError('E_MODEL_ARTIFACT_MISSING','Pilot or unsafe artifact path cannot be promoted.')
        run=(root/run_id).resolve()
        if run.parent!=root or run.name=='pilot':
            raise ServiceError('E_MODEL_ARTIFACT_MISSING','Pilot or unsafe artifact path cannot be promoted.')
        if 'artifact_manifest_sha256' in promotion:
            manifest=run/'artifact_manifest.json'
            if not manifest.is_file() or hashlib.sha256(manifest.read_bytes()).hexdigest()!=promotion['artifact_manifest_sha256']:
                raise ServiceError('E_MODEL_INTEGRITY_FAILED','Promoted model manifest changed.')
        return cls.load_run(run,allow_pilot=False)

    @classmethod
    def load_run(cls,folder: Path,*,allow_pilot=False):
        """Explicit pilot loading is for offline evaluation only; app load never permits it."""
        folder=Path(folder).resolve()
        def read_json(name):
            path=folder/name
            if not path.is_file(): raise ServiceError('E_MODEL_ARTIFACT_MISSING',f'Missing artifact: {name}')
            return json.loads(path.read_text())
        manifest=read_json('artifact_manifest.json')
        required={'cart.joblib','random_forest.joblib','adaboost.joblib','xgboost.json','stacking_meta.joblib','class_mapping.json','feature_schema.json','baseline.json','run_config.json','parity_probe.npz'}
        if not required.issubset(manifest['files']): raise ServiceError('E_MODEL_ARTIFACT_MISSING','Artifact manifest is incomplete.')
        for name,entry in manifest['files'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder) or not path.is_file(): raise ServiceError('E_MODEL_ARTIFACT_MISSING','Artifact path is invalid or missing.')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256'] or path.stat().st_size!=entry['size']:
                raise ServiceError('E_MODEL_INTEGRITY_FAILED',f'Artifact integrity failed: {name}')
        config=read_json('run_config.json')
        if config['run_kind']!='production' and not allow_pilot: raise ServiceError('E_MODEL_ARTIFACT_MISSING','Pilot artifacts are never production models.')
        if read_json('feature_schema.json')['sha256']!=feature_schema()['sha256']:
            raise ServiceError('E_MODEL_SCHEMA_MISMATCH','Model feature schema differs from the running extractor.')
        if read_json('class_mapping.json')['classes']!=list(CLASSES):
            raise ServiceError('E_MODEL_CLASS_ORDER_MISMATCH','Model class order differs from the fixed contract.')
        baseline=read_json('baseline.json')
        if baseline['feature_schema_hash']!=feature_schema()['sha256']:
            raise ServiceError('E_MODEL_SCHEMA_MISMATCH','Baseline feature schema differs.')
        bases={n:joblib.load(folder/(n+'.joblib')) for n in BASE_ORDER if n!='xgboost'}
        from xgboost import XGBClassifier
        xgb=XGBClassifier();xgb.load_model(folder/'xgboost.json');bases['xgboost']=xgb
        meta=joblib.load(folder/'stacking_meta.joblib')
        return cls(bases,meta,baseline,config['run_id'])

    def predict(self,features: np.ndarray) -> dict:
        vector=np.asarray(features,dtype=np.float32)
        if vector.shape!=(84,) or not np.isfinite(vector).all(): raise ServiceError('E_MODEL_SCHEMA_MISMATCH','Expected 84 finite float32 features.')
        start=time.perf_counter();P=predict_probabilities(self.bases,self.meta,vector[None,:]);scores=P['stacking'][0]
        return {'predicted_class':CLASSES[int(np.argmax(scores))],'abstained':False,'class_scores':dict(zip(CLASSES,scores.tolist())),
                'base_predictions':{n:{'top':CLASSES[int(np.argmax(P[n][0]))],'scores':dict(zip(CLASSES,P[n][0].tolist()))} for n in BASE_ORDER},
                'voting_scores':dict(zip(CLASSES,P['soft_voting'][0].tolist())),'final_model':'stacking',
                'latency_ms':(time.perf_counter()-start)*1000,'run_id':self.run_id,'warning':'Most likely modelled condition; not proof of a real-world diagnosis.'}
