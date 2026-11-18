"""Read authentic promoted evaluation data; never train or silently expose a pilot."""
import csv
import hashlib
import json
from pathlib import Path


def run_settings(config: dict) -> dict:
    return {key: config[key] for key in ('base_parameters','voting_weights','cv_folds','deployment_model')
            if key in config}


def read_pilot_research(run_id: str, models_dir='models', results_dir='results') -> dict:
    """Explicit read-only pilot evidence; never reads or writes the production pointer."""
    try:
        if not isinstance(run_id,str) or not run_id or run_id in ('.','..') or any(c in run_id for c in '/\\:'):
            raise ValueError('Invalid pilot run identifier')
        root=(Path(models_dir)/'pilot').resolve();results=(Path(results_dir)/'pilot').resolve()
        folder=(root/run_id).resolve();result_folder=(results/run_id).resolve()
        if not folder.is_relative_to(root) or not result_folder.is_relative_to(results):
            raise ValueError('Pilot path escapes local artifacts')
        config=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
        metrics=json.loads((result_folder/'metrics.json').read_text(encoding='utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite metrics: '+x)))
        if config.get('run_kind')!='pilot' or metrics.get('run_kind')!='pilot' or metrics.get('run_id')!=run_id:
            raise ValueError('Evidence is not the selected pilot')
        if metrics['provenance']['artifact_manifest_hash']!=hashlib.sha256((folder/'artifact_manifest.json').read_bytes()).hexdigest():
            raise ValueError('Pilot evaluation/model provenance mismatch')
        evidence=[(v['predictions_path'],v['predictions_sha256']) for v in metrics['models'].values()]
        parity=metrics['sample_id_parity']
        ids_path=(result_folder/parity['sorted_ids_path']).resolve()
        if not ids_path.is_relative_to(result_folder): raise ValueError('Pilot ID evidence escapes run')
        with ids_path.open(newline='',encoding='utf-8') as handle: ids=[r['sample_id'] for r in csv.DictReader(handle)]
        ids_digest=hashlib.sha256(json.dumps(sorted(ids),separators=(',',':')).encode()).hexdigest()
        if ids_digest!=parity['sorted_ids_sha256']: raise ValueError('Pilot sample ID evidence changed')
        for relative,digest in evidence:
            path=(result_folder/relative).resolve()
            if not path.is_relative_to(result_folder) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Pilot prediction evidence changed')
        return {'status':'available','reason':None,'run_id':run_id,'metrics':metrics,'settings':run_settings(config)}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {'status':'empty','reason':str(exc),'metrics':None}


def read_production_research(models_dir='models',results_dir='results') -> dict:
    try:
        pointer=Path(models_dir)/'current.json'
        if not pointer.exists(): return {'status':'empty','reason':'No evaluated production run has been promoted.','metrics':None}
        promotion=json.loads(pointer.read_text(encoding='utf-8'))
        return read_production_run(promotion['run_id'],models_dir,results_dir,promotion.get('metrics_sha256'))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {'status':'empty','reason':str(exc),'metrics':None}


def read_production_run(run_id,models_dir='models',results_dir='results',metrics_sha256=None):
    """Read an explicit evaluated run so promotion can verify it before writing the pointer."""
    root=Path(models_dir).resolve();results=Path(results_dir).resolve()
    try:
        if not isinstance(run_id,str) or not run_id or run_id in ('.','..') or any(c in run_id for c in '/\\:') or run_id=='pilot':
            raise ValueError('Invalid production run identifier')
        folder=(root/run_id).resolve();result_folder=(results/run_id).resolve()
        if not folder.is_relative_to(root) or not result_folder.is_relative_to(results): raise ValueError('Run path escapes local artifacts')
        config=json.loads((folder/'run_config.json').read_text(encoding='utf-8'))
        payload=(result_folder/'metrics.json').read_bytes()
        if metrics_sha256 is not None and metrics_sha256!=hashlib.sha256(payload).hexdigest():
            raise ValueError('Promoted evaluation metadata changed')
        metrics=json.loads(payload,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON constant: '+x)))
        if config.get('run_kind')!='production' or metrics.get('run_kind')!='production' or metrics.get('run_id')!=run_id:
            raise ValueError('Research results must belong to the promoted production run')
        manifest=folder/'artifact_manifest.json'
        if metrics['provenance']['artifact_manifest_hash']!=hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError('Evaluation/model provenance mismatch')
        for value in metrics['models'].values():
            pred=(result_folder/value['predictions_path']).resolve()
            if not pred.is_relative_to(result_folder) or hashlib.sha256(pred.read_bytes()).hexdigest()!=value['predictions_sha256']:
                raise ValueError('Saved prediction evidence changed')
        return {'status':'available','reason':None,'run_id':run_id,'metrics':metrics,'settings':run_settings(config)}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {'status':'empty','reason':str(exc),'metrics':None}
