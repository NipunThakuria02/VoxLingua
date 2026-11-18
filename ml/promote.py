"""Verify a completed production evaluation and publish its local model pointer."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from voxlingua.distortion import BASE_ORDER, DistortionService
from voxlingua.features import feature_schema
from voxlingua.research import read_production_run

from .build_dataset import digest


def promote(run_id: str, models: Path = Path('models'), results: Path = Path('results')) -> Path:
    pointer = models / 'current.json'
    if pointer.exists():
        raise FileExistsError('A production model is already promoted')
    data = read_production_run(run_id, models, results)
    if data['status'] != 'available':
        raise ValueError(data['reason'])
    run_folder = models / run_id
    evidence = results / run_id
    metrics = data['metrics']
    config_path = run_folder / 'run_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    parity = metrics['reload_parity']
    samples = metrics['sample_id_parity']
    methods = {*BASE_ORDER, 'soft_voting', 'stacking', 'threshold_baseline'}
    if (
        config['run_id'] != run_id
        or config['deployment_model'] != 'stacking'
        or metrics['deployment']['model'] != 'stacking'
        or metrics['deployment']['frozen_before_test'] is not True
        or metrics['feature_schema_hash'] != feature_schema()['sha256']
        or metrics['provenance']['config_hash'] != digest(config_path)
        or metrics['provenance']['test_lock_hash'] != config['test_lock_sha256']
        or set(metrics['models']) != methods
        or parity['probabilities_match'] is not True
        or parity['baseline_labels_match'] is not True
        or parity['probe_count'] < 6
        or len(parity['class_support']) != 6
        or min(parity['class_support'].values()) < 1
        or samples['exact_sets_and_labels_equal'] is not True
        or samples['sample_count'] != 6 * samples['original_count']
        or samples['original_count'] < 1
    ):
        raise ValueError('Production evaluation does not satisfy promotion gates')

    ids_path = (evidence / samples['sorted_ids_path']).resolve()
    if not ids_path.is_relative_to(evidence.resolve()):
        raise ValueError('Sample ID evidence escapes run directory')
    with ids_path.open(newline='', encoding='utf-8') as handle:
        ids = [row['sample_id'] for row in csv.DictReader(handle)]
    expected = hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()
    if len(set(ids)) != len(ids) or len(ids) != samples['sample_count'] or expected != samples['sorted_ids_sha256']:
        raise ValueError('Sample ID evidence changed')
    target_labels = None
    for entry in metrics['models'].values():
        with (evidence / entry['predictions_path']).open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        targets = {row['sample_id']: row['label_id'] for row in rows}
        if (
            entry['sample_ids_sha256'] != expected or len(rows) != len(ids)
            or set(targets) != set(ids) or not set(targets.values()).issubset(set('012345'))
            or (target_labels is not None and targets != target_labels)
        ):
            raise ValueError('Model predictions do not share locked test IDs and labels')
        target_labels = targets
    DistortionService.load_run(run_folder)
    payload = {
        'run_id': run_id,
        'artifact_manifest_sha256': digest(run_folder / 'artifact_manifest.json'),
        'metrics_sha256': digest(evidence / 'metrics.json'),
    }
    # Publish the complete pointer atomically, with the platform's no-replace operation.
    with TemporaryDirectory(dir=models) as directory:
        staged = Path(directory) / 'current.json'
        staged.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
        if os.name == 'nt':
            staged.rename(pointer)
        else:
            os.link(staged, pointer)
    return pointer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    print(f'Promoted {args.run_id}: {promote(args.run_id)}')


if __name__ == '__main__':
    main()
