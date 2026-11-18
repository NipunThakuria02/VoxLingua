"""Unblocked base learners only, train/validation diagnostics; never an ensemble substitute."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import f1_score
from .train_models import read_partition


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data/pilot'));a=p.parse_args(argv)
    X,y,_,_=read_partition(a.data,'train');Xv,yv,_,_=read_partition(a.data,'validation')
    seed=20261115
    models={'cart':DecisionTreeClassifier(max_depth=8,min_samples_leaf=3,random_state=seed),
            'random_forest':RandomForestClassifier(n_estimators=200,max_features='sqrt',min_samples_leaf=2,n_jobs=2,random_state=seed),
            'adaboost':AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2,random_state=seed),n_estimators=150,learning_rate=0.5,random_state=seed)}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Train/validation diagnostics only; full four-base ensemble BLOCKED by XGBoost Application Control',
            'test_features_opened':False,'parameters_tuned':False,'train_samples':len(X),'validation_samples':len(Xv),'models':{}}
    for name,m in models.items():
        start=time.perf_counter();m.fit(X,y);pred=m.predict(Xv)
        report['models'][name]={'fit_seconds':time.perf_counter()-start,'validation_macro_f1':float(f1_score(yv,pred,labels=range(6),average='macro',zero_division=0)),
                                'probability_shape':list(m.predict_proba(Xv).shape)}
        print(name,report['models'][name],flush=True)
    path=Path('results/pilot/base-model-diagnostics-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2));print('Diagnostics:',path)


if __name__=='__main__': main()
