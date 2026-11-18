"""Serializable transparent threshold comparator, tuned without test access."""
from itertools import product
import numpy as np
from sklearn.metrics import f1_score
from voxlingua.features import FEATURE_NAMES

RULE_FEATURES={2:['flat_top_run_frac','clip_frac_ge_0p99'],5:['n_short_gaps_rel','max_drop_db','zero_run_frac'],3:['rms_dbfs'],4:['spectral_cutoff_99_hz','hf_lf_ratio_db'],1:['noise_floor_flatness','frame_rms_dbfs_p10']}


def rule_mask(rule,X):
    parts=[]
    for trigger in rule['triggers']:
        column=X[:,FEATURE_NAMES.index(trigger['feature'])]
        parts.append(column>=trigger['threshold'] if trigger['operator']=='>=' else column<=trigger['threshold'])
    return np.logical_and.reduce(parts) if rule['combine']=='AND' else np.logical_or.reduce(parts)


def predict(rules,X):
    labels=np.zeros(len(X),dtype=int);fired=np.zeros(len(X),dtype=bool)
    for class_id in rules['precedence']:
        hit=rule_mask(rules['rules'][str(class_id)],X)&~fired
        labels[hit]=class_id;fired|=hit
    return labels,float(fired.mean())


def fit(X,y,Xv,yv,schema_hash):
    rules={}
    for class_id,names in RULE_FEATURES.items():
        grids=[]
        for name in names:
            thresholds=sorted(set(float(v) for v in np.quantile(X[:,FEATURE_NAMES.index(name)],[0.1,0.25,0.5,0.75,0.9])))
            grids.append([{'feature':name,'operator':op,'threshold':v} for op in ('>=','<=') for v in thresholds])
        best=-1;chosen=None
        for triggers in product(*grids):
            rule={'combine':'AND' if class_id==4 else 'OR','triggers':list(triggers)}
            score=f1_score(y==class_id,rule_mask(rule,X),zero_division=0)
            if score>best: best=score;chosen=rule
        rules[str(class_id)]=chosen
    result={'schema_version':'baseline-v1','feature_schema_hash':schema_hash,'rules':rules,'precedence':[2,5,3,4,1],
            'default_class':'reference','quantiles':[0.1,0.25,0.5,0.75,0.9],'threshold_fit_partition':'train','precedence_partition':'validation',
            'threshold_objective':'one-vs-rest F1','precedence_objective':'macro F1','tie_break':'canonical feature/operator/threshold grid order; stated precedence wins ties',
            'missing_policy':'Reject nonfinite or wrong-schema X'}
    forward=f1_score(yv,predict(result,Xv)[0],labels=range(6),average='macro',zero_division=0)
    reverse={**result,'precedence':list(reversed(result['precedence']))}
    backward=f1_score(yv,predict(reverse,Xv)[0],labels=range(6),average='macro',zero_division=0)
    return reverse if backward>forward else result
