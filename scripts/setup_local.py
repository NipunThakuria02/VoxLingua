"""Restore verified local runtime artifacts and configure an existing upstream snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def restore_bundle(bundle: Path) -> None:
    """Validate the entire archive before writing any runtime files."""
    expected = json.loads((ROOT / 'models/runtime-bundle.json').read_text(encoding='utf-8'))
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('Runtime bundle SHA-256 differs from models/runtime-bundle.json')
    pending = []
    with zipfile.ZipFile(bundle) as archive:
        if sum(item.file_size for item in archive.infolist()) > 50 * 1024**2:
            raise ValueError('Runtime archive exceeds its expected size allowance')
        for item in archive.infolist():
            destination = (ROOT / item.filename).resolve()
            if (item.is_dir() or not destination.is_relative_to(ROOT)
                    or item.filename.split('/')[0] not in {'models', 'results'}):
                raise ValueError('Invalid runtime archive entry')
            content = archive.read(item)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError(f'Existing runtime file differs: {item.filename}')
            pending.append((destination, content))
    for destination, content in pending:
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as handle:
                handle.write(content)
    print(f'Restored verified production run {expected["run_id"]}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='VoxLingua-Runtime-Models.zip')
    parser.add_argument('--upstream', type=Path, help='Existing verified upstream model snapshot root')
    args = parser.parse_args(argv)
    if args.upstream and not (args.upstream / 'verified-downloads.json').is_file():
        parser.error('--upstream must contain verified-downloads.json and the pinned snapshots')
    if args.bundle:
        restore_bundle(args.bundle.resolve())
    (ROOT / 'results').mkdir(exist_ok=True)
    for source in (ROOT / 'data/setup').glob('*.json'):
        destination = ROOT / 'results' / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    if args.upstream:
        configuration = ROOT / 'configs/runtime.local.yaml'
        settings = yaml.safe_load((ROOT / 'configs/runtime.yaml').read_text(encoding='utf-8'))
        if configuration.exists():
            settings = yaml.safe_load(configuration.read_text(encoding='utf-8'))
        settings['upstream_models'] = str(args.upstream.resolve())
        settings['language_enabled'] = True
        configuration.write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
        print('Enabled verified local language snapshots in configs/runtime.local.yaml')
    print('Start with: python -m streamlit run app.py')


if __name__ == '__main__':
    main()
