import hashlib
import json
from pathlib import Path
import pytest
from voxlingua.distortion import DistortionService
from voxlingua.features import feature_schema
from voxlingua.schemas import CLASSES,ServiceError


def fixture(root):
    root.mkdir()
    files={'cart.joblib':b'not loaded','random_forest.joblib':b'not loaded','adaboost.joblib':b'not loaded',
           'xgboost.json':b'not loaded','stacking_meta.joblib':b'not loaded','parity_probe.npz':b'not loaded',
           'class_mapping.json':json.dumps({'classes':list(CLASSES)}).encode(),
           'feature_schema.json':json.dumps(feature_schema()).encode(),
           'run_config.json':json.dumps({'run_kind':'production','run_id':'fixture'}).encode(),
           'baseline.json':json.dumps({'feature_schema_hash':feature_schema()['sha256']}).encode()}
    for name,data in files.items(): (root/name).write_bytes(data)
    rehash(root)
    return root


def rehash(root):
    files={p.name:{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.iterdir() if p.name!='artifact_manifest.json'}
    (root/'artifact_manifest.json').write_text(json.dumps({'files':files}))


@pytest.mark.parametrize('kind,code',[('missing','E_MODEL_ARTIFACT_MISSING'),('tamper','E_MODEL_INTEGRITY_FAILED'),('schema','E_MODEL_SCHEMA_MISMATCH'),('classes','E_MODEL_CLASS_ORDER_MISMATCH')])
def test_independent_artifact_failure_stages(tmp_path,kind,code):
    root=fixture(tmp_path/'run')
    if kind=='missing': (root/'cart.joblib').unlink()
    if kind=='tamper': (root/'cart.joblib').write_bytes(b'changed')
    if kind=='schema':
        (root/'feature_schema.json').write_text(json.dumps({'sha256':'wrong'}));rehash(root)
    if kind=='classes':
        (root/'class_mapping.json').write_text(json.dumps({'classes':list(reversed(CLASSES))}));rehash(root)
    with pytest.raises(ServiceError) as exc: DistortionService.load_run(root)
    assert exc.value.code==code


def test_pilot_artifacts_cannot_be_promoted(tmp_path):
    root=fixture(tmp_path/'run')
    (root/'run_config.json').write_text(json.dumps({'run_kind':'pilot','run_id':'fixture'}));rehash(root)
    with pytest.raises(ServiceError,match='never production'): DistortionService.load_run(root)
