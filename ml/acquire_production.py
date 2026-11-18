"""Acquire a bounded, disjoint FLEURS and OpenSLR MUSAN subset on Volume D."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np
import soundfile as sf

from .acquire_pilot import archive_subset, fleurs_table, speech_entry
from .download_models import Redirects, digest, fetch
from .production_gate import speech_identity

MIB = 1024 ** 2
MUSAN_MIRRORS = {
    'eu': 'https://openslr.trmal.net/resources/17/musan.tar.gz',
    'elda': 'https://openslr.elda.org/resources/17/musan.tar.gz',
    'cn': 'https://openslr.magicdatatech.com/resources/17/musan.tar.gz',
}
NOISE_CAP = 64 * MIB
SPEECH_CAP = 22 * MIB
TRANSFER_CAP = NOISE_CAP + 15 * (SPEECH_CAP + 6 * MIB)
REPACK_REVISION = '76f9882cfa4475efe11508ac9aa32722f84ca5b7'
REPACK_PATH = 'noise/train-00001-of-00002.parquet'
REPACK_BYTES = 319_889_413
REPACK_SHA256 = '8178d69f8b052c5986c0a7d30c0f718885132a1c7cdae97a7c8f1a3e2523972c'
REPACK_URL = f'https://huggingface.co/datasets/corypaik/musan/resolve/{REPACK_REVISION}/{REPACK_PATH}'
APPROVED_REPACK_CAP = 726 * MIB
NOISE_BUCKETS = [('train', 0), ('train', 1), ('train', 2), ('validation', None), ('test', None)]



def official_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != 'https' or parsed.hostname not in {urlsplit(url).hostname for url in MUSAN_MIRRORS.values()}
        or parsed.username or parsed.password or parsed.port not in (None, 443)
    ):
        raise ValueError('Refusing transfer outside official OpenSLR')


class OpenSLRRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        official_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def retain_noise(root: Path, name: str, payload: bytes, rows: list, parent_hashes: set) -> None:
    path = PurePosixPath(name)
    if 'noise' not in path.parts or path.suffix.casefold() != '.wav' or not 0 < len(payload) <= 5 * MIB:
        return
    parent = 'musan:' + '/'.join(path.parts[path.parts.index('noise'):])
    original_hash = hashlib.sha256(payload).hexdigest()
    if '..' in path.parts or original_hash in parent_hashes or any(row['noise_source_group_id'] == parent for row in rows):
        return
    try:
        with sf.SoundFile(io.BytesIO(payload)) as handle:
            if handle.samplerate != 16000 or handle.channels != 1 or handle.frames < 16000:
                return
            values = handle.read(frames=240000, dtype='float32')
    except sf.LibsndfileError:
        return
    if not np.isfinite(values).all() or float(np.mean(values.astype('float64')**2)) < 1e-6:
        return

    split, fold = NOISE_BUCKETS[len(rows) % len(NOISE_BUCKETS)]
    relative = f'noise/noise-{len(rows):03d}.wav'
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = io.BytesIO()
    sf.write(encoded, values, 16000, format='WAV', subtype='PCM_16')
    retained = encoded.getvalue()
    if destination.exists():
        if destination.read_bytes()!=retained:
            raise ValueError('Existing noise excerpt differs from verified shard')
    else:
        with destination.open('xb') as handle:
            handle.write(retained)
    parent_hashes.add(original_hash)
    rows.append({'noise_id': parent, 'noise_source_group_id': parent, 'corpus': 'MUSAN',
                 'category': 'noise', 'split': split, 'fold_id': fold,
                 'source_relpath': relative, 'archive_member': name,
                 'parent_sha256': original_hash, 'parent_offset_s': 0,
                 'source_sha256': hashlib.sha256(retained).hexdigest(), 'license': 'CC-BY-4.0'})


def repack_noise_subset(root: Path, deadline: float) -> tuple[list, dict]:
    import pyarrow.parquet as parquet

    shard = root / 'musan-noise-00001.parquet'
    receipt = fetch(build_opener(Redirects()), REPACK_URL, shard, REPACK_BYTES, REPACK_SHA256, deadline)
    print(f'Verified MUSAN parquet SHA-256: {REPACK_SHA256}', flush=True)
    table = parquet.ParquetFile(shard)
    if not {'path', 'audio'}.issubset(table.schema_arrow.names):
        raise ValueError('Pinned MUSAN shard lacks parent paths or audio')
    rows, hashes = [], set()
    for batch in table.iter_batches(batch_size=4, columns=['path', 'audio']):
        for row in batch.to_pylist():
            path, audio = row['path'], row['audio']
            if isinstance(path, str) and isinstance(audio, dict) and isinstance(audio.get('bytes'), bytes):
                retain_noise(root, path, audio['bytes'], rows, hashes)
            if len(rows) == 15:
                return rows, {'repository':'corypaik/musan','revision':REPACK_REVISION,
                              'shard':REPACK_PATH,'shard_url':REPACK_URL,'shard_sha256':REPACK_SHA256,
                              'distribution':'Community Parquet repack of MUSAN, with original noise paths',
                              'transfer_bytes':0 if receipt['reused'] else REPACK_BYTES}
    raise ValueError('Pinned MUSAN shard did not yield 15 distinct usable original noise parents')


def noise_subset(root: Path, deadline: float, mirror: str = 'eu', *,
                 local_archive: Path | None = None) -> tuple[list, dict]:
    """Read a bounded network prefix or a provisioned, checksum-verified archive."""
    opener = build_opener(OpenSLRRedirects())
    rows, transferred = [], 0
    prefix_hash = hashlib.sha256()
    parent_hashes = set()

    class Prefix:
        def read(self, size):
            nonlocal transferred
            if local_archive is None and transferred >= NOISE_CAP:
                raise ValueError('No sufficient MUSAN noise within the official archive prefix cap')
            if time.monotonic() >= deadline:
                raise TimeoutError('Production acquisition deadline reached')
            limit = size if local_archive is not None else min(size, NOISE_CAP - transferred)
            data = response.read(limit)
            transferred += len(data)
            prefix_hash.update(data)
            return data

    url = MUSAN_MIRRORS[mirror]
    request = Request(url, headers={'Range': f'bytes=0-{NOISE_CAP-1}'})
    stream = local_archive.open('rb') if local_archive is not None else opener.open(request, timeout=25)
    with stream as response:
        etag = None
        if local_archive is None:
            official_url(response.url)
            etag = response.headers.get('ETag')
        with tarfile.open(fileobj=Prefix(), mode='r|gz') as archive:
            for member in archive:
                if not member.isfile():
                    continue
                path = PurePosixPath(member.name)
                if 'noise' not in path.parts or path.suffix.casefold() != '.wav' or not 0 < member.size <= 5 * MIB:
                    continue
                retain_noise(root, member.name, archive.extractfile(member).read(), rows, parent_hashes)
                if len(rows) == 15:
                    break
    if len(rows) < 15:
        raise ValueError('Insufficient usable distinct MUSAN noise sources')
    return rows, {'archive_url': url, 'etag': etag,
                  'compressed_prefix_sha256': prefix_hash.hexdigest(),
                  'compressed_bytes_read': transferred,
                  'transfer_bytes': transferred if local_archive is None else 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('data/raw/production'))
    parser.add_argument('--reservation-plan', type=Path, default=Path('results/production-plan.json'))
    parser.add_argument('--musan-mirror', choices=MUSAN_MIRRORS, default='eu')
    parser.add_argument('--approved-repack', action='store_true', help='Use the explicitly approved 726 MiB pinned repack route')
    parser.add_argument('--recover-with-local-speech',type=int,metavar='PRIOR_BATCH_MIB',help='Reuse complete audio and charge a conservative bound for prior transfers')
    parser.add_argument('--resume-first-prefix', action='store_true', help='Resume the interrupted first FLEURS prefix and reserve its maximum 28 MiB')
    parser.add_argument('--musan-archive', type=Path, help='Already provisioned official archive')
    parser.add_argument('--musan-sha256', help='Trusted SHA-256 for the provisioned archive')
    args = parser.parse_args(argv)
    root = args.destination.resolve()
    plan = json.loads(args.reservation_plan.read_text(encoding='utf-8'))
    preflight = json.loads(Path('results/preflight.json').read_text(encoding='utf-8'))
    revision = preflight['upstream']['sources']['fleurs']['revision']
    ancestor = next(path for path in root.parents if path.is_dir())
    if shutil.disk_usage(ancestor).free < 2 * 1024 ** 3:
        raise ValueError('Production acquisition requires at least 2 GiB free')
    if args.approved_repack and args.musan_archive is not None:
        parser.error('Select one MUSAN source')
    if args.musan_archive is not None:
        if not args.musan_sha256 or digest(args.musan_archive)!=args.musan_sha256.lower():
            raise ValueError('Provisioned archive must match the supplied trusted SHA-256')
    elif args.musan_sha256:
        parser.error('--musan-sha256 requires --musan-archive')
    deadline = time.monotonic() + 30 * 60
    root.mkdir(parents=True, exist_ok=True)
    previous_fleurs_bound = 0
    speech_cap = SPEECH_CAP
    recover = args.recover_with_local_speech is not None
    if recover:
        if (not args.approved_repack or args.resume_first_prefix or (root/'acquisition.json').exists()
            or not (root/'first-prefix-recovery.json').is_file()
            or not (root/'musan-noise-00001.parquet').is_file()):
            raise ValueError('Recovery requires the interrupted approved batch and its verified shard')
        prior = args.recover_with_local_speech*MIB
        # This includes the verified shard and the two earlier FLEURS attempts.
        if not REPACK_BYTES < prior < APPROVED_REPACK_CAP:
            raise ValueError('Supply the conservative total prior transfer bound in MiB')
        previous_fleurs_bound = prior - REPACK_BYTES
        with (root/'local-speech-recovery.json').open('x',encoding='utf-8') as handle:
            json.dump({'prior_batch_transfer_upper_bound':prior},handle)
        speech_cap = 20*MIB
    elif args.resume_first_prefix:
        if not args.approved_repack or (root/'acquisition.json').exists():
            parser.error('First-prefix recovery requires the unfinished approved repack acquisition')
        existing = root/'speech'
        first = existing/plan['target']['languages'][0]/'train'
        if not (root/'musan-noise-00001.parquet').is_file() or not first.is_dir() or any(
            not file.is_relative_to(first) for file in existing.rglob('*.wav')
        ):
            raise ValueError('Recovery expects a finished shard and files from only the first FLEURS prefix')
        # The earlier prefix used at most 6 MiB of metadata plus 22 MiB of archive data,
        # with one possible lookahead byte each. Charge that entire upper bound.
        previous_fleurs_bound = 28 * MIB + 2
        with (root/'first-prefix-recovery.json').open('x', encoding='utf-8') as handle:
            json.dump({'previous_fleurs_transfer_upper_bound':previous_fleurs_bound}, handle)
        speech_cap = 20 * MIB
    elif any(root.iterdir()):
        raise ValueError('Production acquisition requires an empty destination')
    if args.approved_repack:
        noise, noise_receipt = repack_noise_subset(root, deadline)
    else:
        noise, noise_receipt = noise_subset(root, deadline, args.musan_mirror, local_archive=args.musan_archive)
    if args.musan_archive is not None:
        noise_receipt['archive_sha256'] = args.musan_sha256.lower()
    transferred = REPACK_BYTES if args.resume_first_prefix or recover else noise_receipt['transfer_bytes']
    observed = transferred
    transferred += previous_fleurs_bound
    print(f'MUSAN: {len(noise)} verified parent files. {transferred/MIB:.2f} MiB charged to batch.', flush=True)
    reservations = plan['reserved_pilot_sources']
    blocked_bytes = {item['source_sha256'] for item in reservations['speech']}
    blocked_sentences = set(reservations['sentence_ids'])
    speech = []

    def eligible(row):
        _, sentence, _ = speech_identity(row)
        if row['source_sha256'] in blocked_bytes or sentence in blocked_sentences:
            return False
        blocked_bytes.add(row['source_sha256'])
        blocked_sentences.add(sentence)
        return True

    for config in plan['target']['languages']:
        for split, count in [('train', 32), ('validation', 10), ('test', 10)]:
            if time.monotonic() >= deadline:
                raise TimeoutError('Production acquisition deadline reached')
            directory = root/'speech'/config/split
            local_files = sorted(directory.glob('*.wav')) if recover else []
            minimum = 8 if split=='train' else 4
            # Reserve the largest possible response before issuing a request.
            maximum = (6*MIB+1) if len(local_files)>=minimum else (6*MIB+1+speech_cap)
            if transferred + maximum > APPROVED_REPACK_CAP and args.approved_repack:
                raise ValueError('Remaining approved budget cannot cover the next bounded request')
            if len(local_files)>=minimum:
                table,used,_ = fleurs_table(build_opener(Redirects()),config,split,revision)
                rows = [speech_entry(table[f.name],config,split,revision,f.relative_to(root),f.read_bytes(),
                                     'validated local WAV from interrupted pinned archive + pinned TSV')
                        for f in local_files if f.name in table]
                rows = [row for row in rows if eligible(row)]
                if len(rows)<minimum:
                    raise ValueError('Too few verified local WAVs after matching the pinned metadata')
            else:
                rows, used = archive_subset(root, config, split, count, revision, cap=speech_cap,
                                            eligible=eligible, minimum=minimum)
            transferred += used
            observed += used
            speech.extend(rows)
            print(f'FLEURS {config}/{split}: {len(rows)} candidates, cumulative {transferred/MIB:.2f} MiB', flush=True)
    receipt = {
        'run_kind': 'production', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'speech': speech, 'noise': noise, 'noise_corpus': 'MUSAN', 'noise_is_fallback': False,
        'revisions': {'FLEURS': revision, 'MUSAN': noise_receipt},
        'licenses': {'FLEURS': 'CC-BY-4.0', 'MUSAN': 'CC-BY-4.0'},
        'citations': {'FLEURS': 'https://huggingface.co/datasets/google/fleurs', 'MUSAN': 'https://www.openslr.org/17/'},
        'transfer_bytes': transferred, 'observed_transfer_bytes': observed,
        'interrupted_prefix_transfer_upper_bound_bytes': previous_fleurs_bound, 'transfer_cap_bytes': APPROVED_REPACK_CAP if args.approved_repack else TRANSFER_CAP,
        'limitations': ['Bounded archive-prefix subset. Actual retention depends on the reference screen.',
                        'Environmental noise only. No verified speaker identifiers.'],
    }
    if transferred > receipt['transfer_cap_bytes']:
        raise ValueError('Acquisition exceeded the approved transfer cap')
    with (root / 'acquisition.json').open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, indent=2)


if __name__ == '__main__':
    main()
