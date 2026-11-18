"""Reviewed upstream Vaani inference from verified local snapshots only."""
from __future__ import annotations
import gc
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
import time
import numpy as np
from .schemas import ServiceError

VAANI_REVISION='df5a1394996671df3234ee23257d3259a40058fd'
BASE_REVISION='41f01f3fe87f28c78e2fbf8b568835947dd65ed9'

# First-decoder-token language IDs from the pinned Whisper generation config.
WHISPER_LANGUAGE_IDS = {'Hindi': 50276, 'Tamil': 50287}
UPLOAD_REFINEMENT_RULES = {
    'Chhattisgarhi': ('Hindi', 0.03),
    'Khariboli': ('Hindi', 0.03),
    'Kumaoni': ('Hindi', 0.03),
    'Kannada': ('Tamil', 0.045),
    'Malayalam': ('Tamil', 0.045),
}
WHISPER_MINIMUM_SCORES = {'Hindi': 0.80, 'Tamil': 0.70}


def upload_candidate(top5: list[dict]) -> dict | None:
    """Choose an eligible alternative from the unchanged Vaani ranking."""
    rule = UPLOAD_REFINEMENT_RULES.get(top5[0]['label']) if top5 else None
    if rule is None:
        return None
    label, minimum = rule
    return next((item for item in top5[1:] if item['label'] == label and item['score'] >= minimum), None)


def apply_upload_refinement(result: dict, candidate: dict, whisper_id: int, whisper_score: float) -> None:
    """Record the second opinion and update the selected label when both gates pass."""
    label = candidate['label']
    applied = whisper_id == WHISPER_LANGUAGE_IDS[label] and whisper_score >= WHISPER_MINIMUM_SCORES[label]
    result['upload_refinement'] = {
        'model_id': 'openai/whisper-large-v3-turbo', 'revision': BASE_REVISION,
        'candidate': candidate['label'], 'vaani_first': result['top5'][0]['label'],
        'whisper_top_token_id': whisper_id, 'whisper_top_score': whisper_score,
        'applied': applied,
    }
    if applied:
        result['top_label'] = candidate['label']
        result['top_score'] = candidate['score']


class LanguageService:
    def __init__(self, root: Path | str = 'models/upstream', prefer_gpu: bool = True):
        self.root=Path(root)
        self.prefer_gpu=prefer_gpu
        self.model=None
        self.processor=None
        self.device=None
        self.labels=None
        self._whisper = None
        self._lock = RLock()

    def _verify(self):
        receipt=self.root/'verified-downloads.json'
        if not receipt.exists():
            raise ServiceError('E_LANG_UNAVAILABLE','Approved upstream snapshots have not finished downloading.')
        records=json.loads(receipt.read_text())['files']
        required={("ARTPARK-IISc/Vaani-LID_v0",VAANI_REVISION,n) for n in ['model.safetensors','config.json','preprocessor_config.json','modeling_whisper_lid.py']}
        required|={("openai/whisper-large-v3-turbo",BASE_REVISION,n) for n in ['model.safetensors','config.json']}
        recorded={(r['repository'],r['revision'],Path(r['path']).name):r for r in records}
        if not required.issubset(recorded):
            raise ServiceError('E_LANG_UNAVAILABLE','Download receipt lacks required pinned files.')
        for key in required:
            record=recorded[key]
            path=self.root/key[0].split('/')[-1]/key[1]/key[2]
            if not path.is_file() or path.stat().st_size!=record['bytes']:
                raise ServiceError('E_MODEL_INTEGRITY_FAILED','Upstream file is missing or its size changed.')
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
            if h.hexdigest()!=record['sha256']:
                raise ServiceError('E_MODEL_INTEGRITY_FAILED','Upstream file integrity check failed.')

    def _load(self):
        if self.model is not None: return
        self._verify()
        # Set before importing HF. All nested from_pretrained calls must stay offline.
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
        os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
        os.environ.setdefault('HF_HOME',str(self.root/'.hf-runtime'))
        os.environ.setdefault('HF_MODULES_CACHE',str(self.root/'.hf-modules'))
        import torch
        from transformers import AutoConfig, AutoModelForAudioClassification, WhisperFeatureExtractor
        torch.set_num_threads(2)
        folder=self.root/'Vaani-LID_v0'/VAANI_REVISION
        config=AutoConfig.from_pretrained(folder,trust_remote_code=True,local_files_only=True)
        config.whisper_model=str((self.root/'whisper-large-v3-turbo'/BASE_REVISION).resolve())
        self.device='cuda' if self.prefer_gpu and torch.cuda.is_available() else 'cpu'
        dtype=torch.float16 if self.device=='cuda' else torch.float32
        try:
            model=AutoModelForAudioClassification.from_pretrained(folder,config=config,trust_remote_code=True,local_files_only=True,torch_dtype=dtype,low_cpu_mem_usage=True)
            self.model=model.to(self.device).eval()
            self.processor=WhisperFeatureExtractor.from_pretrained(folder,local_files_only=True)
            self.labels={int(i):label for i,label in self.model.config.id2label.items()}
        except torch.cuda.OutOfMemoryError as exc:
            self.unload()
            raise ServiceError('E_LANG_OOM','Vaani does not fit available GPU memory.') from exc
        except Exception as exc:
            self.unload()
            raise ServiceError('E_LANG_FAILED',f'Local Vaani loading failed: {type(exc).__name__}: {exc}') from exc

    def identify(self, x16k: np.ndarray, *, refine_upload: bool = False) -> dict:
        if x16k.ndim!=1 or not np.isfinite(x16k).all() or not 16000<=len(x16k)<=240000:
            raise ServiceError('E_AUDIO_TOO_SHORT','Language input must be finite mono 16-kHz audio of 1–15 seconds.')
        with self._lock:
            return self._identify(x16k, refine_upload)

    def _identify(self, x16k: np.ndarray, refine_upload: bool) -> dict:
        import torch
        start=time.perf_counter()
        self._load()
        inputs=self.processor(x16k,sampling_rate=16000,return_tensors='pt')
        features=inputs['input_features'].to(self.device,dtype=next(self.model.parameters()).dtype)
        try:
            with torch.inference_mode(): scores=self.model(input_features=features).logits.float().softmax(dim=-1)[0].cpu().numpy()
        except torch.cuda.OutOfMemoryError as exc:
            raise ServiceError('E_LANG_OOM','Vaani inference ran out of GPU memory.') from exc
        order=np.argsort(-scores)[:5]
        result = {'top_label':self.labels[int(order[0])],'top_score':float(scores[order[0]]),
                'top5':[{'label':self.labels[int(i)],'score':float(scores[i]),'native_script':None} for i in order],
                'device':'cuda-fp16' if self.device=='cuda' else 'cpu-fp32',
                'model_id':'ARTPARK-IISc/Vaani-LID_v0','revision':VAANI_REVISION,
                'warnings':['Upstream model scores, not calibrated probabilities. Unsupported languages and code-switching are not reliably detected.']}
        candidate = upload_candidate(result['top5']) if refine_upload else None
        if candidate:
            try:
                token_id, score = self._whisper_language(features)
                apply_upload_refinement(result, candidate, token_id, score)
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                result['warnings'].append('Whisper second opinion unavailable. Showing the Vaani ranking.')
                result['upload_refinement_error'] = type(exc).__name__
        result['inference_ms'] = (time.perf_counter()-start)*1000
        return result

    def _whisper_language(self, features) -> tuple[int, float]:
        """Read the pinned Whisper model's first decoder step, restricted to language tokens."""
        import torch
        if self._whisper is None:
            from transformers import WhisperForConditionalGeneration
            folder = self.root / 'whisper-large-v3-turbo' / BASE_REVISION
            self._whisper = WhisperForConditionalGeneration.from_pretrained(
                folder, local_files_only=True, dtype=features.dtype,
            ).to(self.device).eval()
        start = self._whisper.config.decoder_start_token_id
        if start != 50258:
            raise ValueError('Unexpected Whisper language vocabulary')
        token = torch.tensor([[start]], device=self.device)
        with torch.inference_mode():
            logits = self._whisper(input_features=features, decoder_input_ids=token, use_cache=False).logits
            score, index = logits[0, 0, 50259:50359].float().softmax(dim=0).max(dim=0)
        return 50259 + int(index.item()), float(score.item())

    def unload(self):
        with self._lock:
            self.model=self.processor=self._whisper=None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except ImportError: pass
