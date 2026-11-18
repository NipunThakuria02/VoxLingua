import pytest

from voxlingua.language import apply_upload_refinement, upload_candidate


def ranked(first, second, score):
    return [{'label': first, 'score': 0.55}, {'label': second, 'score': score}]


@pytest.mark.parametrize('first, target, score, token', [
    ('Kumaoni', 'Hindi', 0.10, 50276),
    ('Chhattisgarhi', 'Hindi', 0.20, 50276),
    ('Khariboli', 'Hindi', 0.03, 50276),
    ('Malayalam', 'Tamil', 0.045, 50287),
    ('Kannada', 'Tamil', 0.23, 50287),
])
def test_refinement_exposes_original_scores(first, target, score, token):
    top5 = ranked(first, target, score)
    result = {'top_label': first, 'top_score': top5[0]['score'], 'top5': top5}
    apply_upload_refinement(result, upload_candidate(top5), token, 0.8)
    assert result['top_label'] == target
    assert result['top_score'] == score
    assert result['top5'] == ranked(first, target, score)
    assert result['upload_refinement']['vaani_first'] == first


@pytest.mark.parametrize('first, second, score', [
    ('Telugu', 'Tamil', 0.22), ('Bengali', 'Hindi', 0.25), ('Hindi', 'Tamil', 0.28),
    ('Malayalam', 'Tamil', 0.044), ('Chhattisgarhi', 'Bengali', 0.24),
])
def test_candidate_preserves_other_or_weak_predictions(first, second, score):
    assert upload_candidate(ranked(first, second, score)) is None


@pytest.mark.parametrize('token, score', [(50287, 0.699), (50299, 0.96)])
def test_secondary_model_must_pick_the_candidate_above_its_gate(token, score):
    top5 = ranked('Kannada', 'Tamil', 0.2)
    result = {'top_label': 'Kannada', 'top_score': 0.55, 'top5': top5}
    apply_upload_refinement(result, upload_candidate(top5), token, score)
    assert result['top_label'] == 'Kannada'
    assert result['upload_refinement']['applied'] is False
