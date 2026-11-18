"""Fail-closed metadata gate before any production features or training exist."""
import hashlib
import json
from pathlib import Path
import unicodedata

from voxlingua.audio import decode_audio
from voxlingua.schemas import InputLimits


def source_audio(root: Path, row: dict):
    """Check physical bytes and decoded identity before using a source."""
    root = root.resolve()
    path = (root / row['source_relpath']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source path escapes the acquisition directory')
    if path.stat().st_size > InputLimits().max_bytes:
        raise ValueError('Source exceeds WAV input size limit')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row['source_sha256']:
        raise ValueError('Source file hash changed')
    audio = decode_audio(data)
    header = f'{audio.sample_rate}:{audio.n_channels}:{audio.samples.shape}'.encode()
    content = hashlib.sha256(header + audio.samples.astype('<f4').tobytes()).hexdigest()
    if 'content_sha256' in row and row['content_sha256'] != content:
        raise ValueError('Source decoded hash changed')
    return audio, content


def speech_identity(row: dict) -> tuple[str, str, str]:
    path = unicodedata.normalize('NFC', row['source_relpath'])
    identity = [row['source_revision'], row['config'], row['official_split'], path, row['source_sha256']]
    original = 'fleurs:' + hashlib.sha256(json.dumps(identity, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    normalized = ' '.join(unicodedata.normalize('NFC', row['transcription']).split())
    text_hash = hashlib.sha256(normalized.encode()).hexdigest()
    return original, row['config'] + ':' + text_hash, text_hash


def validate_production_acquisition(candidate: dict, plan: dict) -> dict:
    """Reject pilot reuse, unapproved corpus changes and split leakage; never downloads."""
    if candidate.get('run_kind')!='production': raise ValueError('Production run_kind is required')
    if candidate.get('noise_corpus')!='MUSAN' or candidate.get('noise_is_fallback') is not False:
        raise ValueError('Current production plan requires genuine MUSAN; pilot fallback approval does not extend to production')
    if not candidate.get('licenses') or not candidate.get('revisions'):
        raise ValueError('Verified production licenses and source revisions are required')
    speech=candidate.get('speech',[]);noise=candidate.get('noise',[])
    if not speech or not noise: raise ValueError('Production speech and noise sources are required')
    reserve=plan['reserved_pilot_sources']
    reserved_hashes={r[k] for kind in ('speech','noise') for r in reserve[kind] for k in ('source_sha256','content_sha256')}
    reserved_ids=set(reserve['original_ids']);reserved_sentences=set(reserve['sentence_ids'])
    reserved_parents={r['noise_source_group_id'] for r in reserve['noise']}
    content_splits={};parent_splits={};parent_hash_splits={};speech_content=set()
    for kind,rows in [('speech',speech),('noise',noise)]:
        for row in rows:
            for key in ('source_sha256','content_sha256'):
                value=row.get(key)
                if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('Source and canonical content SHA-256 values are required')
                if value in reserved_hashes: raise ValueError('Pilot byte/content hash cannot enter production')
            split=row.get('official_split') if kind=='speech' else row.get('split')
            if split not in ('train','validation','test'): raise ValueError('Official production split is required')
            content=row['content_sha256']
            if content in content_splits and content_splits[content]!=split: raise ValueError('Production content crosses partitions')
            content_splits[content]=split
            if kind=='speech':
                if row.get('source')!='google/fleurs': raise ValueError('Current production speech plan requires FLEURS')
                if row.get('original_id') in reserved_ids or row.get('sentence_id') in reserved_sentences:
                    raise ValueError('Pilot original/sentence cannot enter production')
                if not row.get('original_id') or not row.get('sentence_id'): raise ValueError('Recording and sentence identities are required')
                if row.get('split',split)!=split: raise ValueError('Official partition membership changed')
                if content in speech_content: raise ValueError('Duplicate production original recording')
                speech_content.add(content)
            else:
                parent=row.get('noise_source_group_id')
                if not parent or parent in reserved_parents: raise ValueError('Distinct production noise parent is required')
                if parent in parent_splits and parent_splits[parent]!=split: raise ValueError('Noise parent crosses partitions')
                parent_splits[parent]=split
                if parent_hash:=row.get('parent_sha256'):
                    if parent_hash in reserved_hashes or parent_hash_splits.get(parent_hash,split)!=split:
                        raise ValueError('Noise parent bytes overlap a reserved source or another partition')
                    parent_hash_splits[parent_hash]=split
    return {'status':'metadata_gate_passed','speech_originals':len(speech),'noise_sources':len(noise),
            'not_training_acceptance':True,'remaining':'Physical source hashes, CV assignment, screening, completeness, reload/evaluation and promotion require their own verification'}
