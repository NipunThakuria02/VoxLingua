import numpy as np
import pytest
from ml.augment import derivative,screen_reference
from ml.build_dataset import validate_samples
from ml.train_models import read_partition
from voxlingua.audio import quantize_int16
from voxlingua.schemas import CLASSES


def test_reference_quantization_is_identity():
    x=quantize_int16(np.sin(np.arange(48000)*0.17)*0.3)
    y,meta=derivative(x,'reference',np.random.default_rng(1))
    assert np.array_equal(x,y) and meta=={}


def test_group_completeness_uses_original_not_shared_speaker():
    rows=[]
    for oid in ['recording1','recording2']:
        for label in CLASSES:
            rows.append({'original_id':oid,'sample_id':oid+'|'+label,'label':label,'split':'train','fold_id':0,'cv_group_id':'speaker1'})
    validate_samples(rows,[])
    with pytest.raises(ValueError,match='six'):
        validate_samples(rows[:-1],[])


def test_trainer_refuses_test_partition_before_reading(tmp_path):
    with pytest.raises(ValueError,match='locked test'):
        read_partition(tmp_path,'test')


def test_noise_determinism_and_applied_snr():
    x=quantize_int16(np.sin(np.arange(48000)*0.17)*0.3)
    noise=np.random.default_rng(2).normal(0,0.1,80000)
    a,am=derivative(x,'added_noise',np.random.default_rng(9),noise)
    b,bm=derivative(x,'added_noise',np.random.default_rng(9),noise)
    assert np.array_equal(a,b) and am==bm
    assert np.max(np.abs(a))<=0.9901 and 0<=am['applied_snr_db']<=20


def test_screening_and_clipping_actual_metadata():
    x=quantize_int16(np.sin(np.arange(48000)*0.17)*0.3)
    assert screen_reference(x)==[]
    y,meta=derivative(x,'clipping',np.random.default_rng(3))
    assert np.mean(np.abs(y)>=meta['clip_ceiling']-1e-6)==meta['applied_clip_fraction']
    assert meta['applied_clip_fraction']>=0.002
    assert 'reference_already_clipped' in screen_reference(np.ones(48000))
