"""Release restoration preserves existing artifacts and validates before writing."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from scripts import setup_local


def test_saved_extractor_identity_survives_repository_export():
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads((root / 'models/runtime-bundle.json').read_text())
    extractor = root / 'src/voxlingua/features.py'
    assert hashlib.sha256(extractor.read_bytes()).hexdigest() == metadata['feature_extractor_sha256']
    assert '/src/voxlingua/features.py -text' in (root / '.gitattributes').read_text()


def bundle_fixture(tmp_path, monkeypatch, members):
    root = tmp_path / 'checkout'
    (root / 'models').mkdir(parents=True)
    archive = tmp_path / 'runtime.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        for name, content in members.items():
            handle.writestr(name, content)
    (root / 'models/runtime-bundle.json').write_text(json.dumps({
        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'run_id': 'fixture',
    }))
    monkeypatch.setattr(setup_local, 'ROOT', root)
    return root, archive


def test_bundle_restoration_is_repeatable(tmp_path, monkeypatch):
    root, archive = bundle_fixture(tmp_path, monkeypatch, {'models/current.json': '{}'})
    setup_local.restore_bundle(archive)
    setup_local.restore_bundle(archive)
    assert (root / 'models/current.json').read_text() == '{}'


def test_changed_bundle_fails_before_writing(tmp_path, monkeypatch):
    root, archive = bundle_fixture(tmp_path, monkeypatch, {'models/current.json': '{}'})
    archive.write_bytes(archive.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA-256'):
        setup_local.restore_bundle(archive)
    assert not (root / 'models/current.json').exists()


def test_existing_artifacts_and_unsafe_paths_are_preserved(tmp_path, monkeypatch):
    root, archive = bundle_fixture(tmp_path, monkeypatch, {
        'models/new.json': '{}', 'models/current.json': '{}',
    })
    (root / 'models/current.json').write_text('different')
    with pytest.raises(ValueError, match='differs'):
        setup_local.restore_bundle(archive)
    assert not (root / 'models/new.json').exists()
    root, archive = bundle_fixture(tmp_path / 'unsafe', monkeypatch, {'../escape.txt': 'unsafe'})
    with pytest.raises(ValueError, match='Invalid'):
        setup_local.restore_bundle(archive)
    assert not (root.parent / 'escape.txt').exists()
