import pytest
from ml.evaluate import compare_predictions


def test_audit_float_precision_preserves_exact_labels(tmp_path):
    a=tmp_path/'original.csv';b=tmp_path/'audit.csv'
    a.write_text('sample_id,label_id,prediction_id,reference\na,0,0,0.5\n')
    b.write_text('sample_id,label_id,prediction_id,reference\na,0,0,0.5000000000000001\n')
    assert compare_predictions(a,b)['ids_targets_labels_exact']
    b.write_text('sample_id,label_id,prediction_id,reference\na,0,1,0.5\n')
    with pytest.raises(ValueError,match='labels'): compare_predictions(a,b)
    b.write_text('sample_id,label_id,prediction_id,reference\na,0,0,0.51\n')
    with pytest.raises(ValueError,match='tolerance'): compare_predictions(a,b)
