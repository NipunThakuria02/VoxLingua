from copy import deepcopy
import pytest
from ml.production_gate import validate_production_acquisition


def fixture():
    plan={'reserved_pilot_sources':{'speech':[{'source_sha256':'a'*64,'content_sha256':'b'*64}],
                                  'noise':[{'source_sha256':'c'*64,'content_sha256':'d'*64,'noise_source_group_id':'pilot-parent'}],
                                  'original_ids':['pilot-original'],'sentence_ids':['pilot-sentence']}}
    acquisition={'run_kind':'production','noise_corpus':'MUSAN','noise_is_fallback':False,'licenses':{'FLEURS':'CC-BY-4.0','MUSAN':'CC-BY-4.0'},'revisions':{'FLEURS':'test-only','MUSAN':'test-only'},
                 'speech':[{'source':'google/fleurs','source_sha256':'1'*64,'content_sha256':'2'*64,'official_split':'train','original_id':'fresh-original','sentence_id':'fresh-sentence'}],
                 'noise':[{'source_sha256':'3'*64,'content_sha256':'4'*64,'split':'train','noise_source_group_id':'fresh-parent'}]}
    return acquisition,plan


def test_fresh_production_metadata_can_pass_without_promoting_or_training():
    candidate,plan=fixture()
    result=validate_production_acquisition(candidate,plan)
    assert result['status']=='metadata_gate_passed' and result['not_training_acceptance']


@pytest.mark.parametrize('field,value',[('source_sha256','a'*64),('content_sha256','b'*64),('original_id','pilot-original'),('sentence_id','pilot-sentence')])
def test_pilot_originals_and_equivalent_audio_cannot_reenter_production(field,value):
    candidate,plan=fixture();candidate['speech'][0][field]=value
    with pytest.raises(ValueError,match='Pilot'): validate_production_acquisition(candidate,plan)


def test_pilot_esc10_approval_does_not_approve_production_fallback():
    candidate,plan=fixture();candidate.update(noise_corpus='ESC-10',noise_is_fallback=True)
    with pytest.raises(ValueError,match='MUSAN'): validate_production_acquisition(candidate,plan)


def test_noise_parent_cannot_cross_production_partitions():
    candidate,plan=fixture();other=deepcopy(candidate['noise'][0]);other.update(split='test',source_sha256='5'*64,content_sha256='6'*64)
    candidate['noise'].append(other)
    with pytest.raises(ValueError,match='parent crosses'): validate_production_acquisition(candidate,plan)
