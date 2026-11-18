# Dataset sources and preparation

## Speech

[Google FLEURS](https://huggingface.co/datasets/google/fleurs) supplies multilingual read speech under CC BY 4.0. The pinned revision is 70bb2e84b976b7e960aa89f1c648e09c59f894dd. The selected configurations are en_us, hi_in, ta_in, bn_in and te_in. Official training, validation and test assignments are preserved.

The production acquisition produced 185 candidate WAVs. Screening and derivative checks retained 153 originals. Source/content hashes identify recordings, while transcription identity detects reused sentences.

## Environmental noise

[MUSAN / OpenSLR SLR17](https://www.openslr.org/17/), by Snyder, Chen and Povey, is licensed CC BY 4.0. Production used the environmental noise category from a checksum-verified [community repack](https://huggingface.co/datasets/corypaik/musan).

The repack revision is 76f9882cfa4475efe11508ac9aa32722f84ca5b7. The shard is noise/train-00001-of-00002.parquet with SHA-256 8178d69f8b052c5986c0a7d30c0f718885132a1c7cdae97a7c8f1a3e2523972c. Fifteen distinct noise parents were assigned to training, validation and test as 9/3/3.

The development pilot used [ESC-10](https://github.com/karolpiczak/ESC-50) environmental sound under CC BY 3.0. The pilot's 40 originals and 240 examples remain separate from production.

## Screening and cleaning

1. Check source SHA-256, decoded-content identity, WAV format, finite samples and 1-15-second duration.
2. Assign splits and training folds by original recording. Keep all derivatives together and separate noise parents across splits and folds.
3. Mean-downmix model copies to mono and polyphase-resample to 16 kHz. Preserve decoded native-rate audio for signal measurements.
4. Quantize the distortion input copy to the PCM16 sample grid.
5. Reject reference clips with clipping fraction above 0.0005, an interior zero run of at least 320 samples, peak below -30 dBFS, or RMS below -45 dBFS.
6. Detect duplicate source/content identities and exclude reserved pilot recordings and sentence identities from production.
7. Retain an original only when it produces all six valid conditions.
8. Extract the same 84 finite float32 features used during inference.

Preparation preserves amplitude and does not perform peak normalization or speech denoising.

## Controlled transformations

| Condition | Preparation |
| --- | --- |
| Reference | Prepared original |
| Added noise | MUSAN noise at sampled 0-20 dB injected SNR with headroom control |
| Clipping | Amplification and peak capping with applied-fraction verification |
| Low volume | Sampled 15-35 dB attenuation |
| Frequency filtering | Fourth-order low-pass, high-pass or band-pass filtering with stop-band checks |
| Interruption | Short gaps of roughly 30-300 ms in active speech |

The exact seeded transformations and checks are in ml/augment.py and ml/build_dataset.py.

## Local data layout

Raw acquisitions belong in data/raw/production. Prepared manifests and features belong in data/production. Raw audio and generated features stay git-ignored. sources.json records names, licenses and revisions. setup/ contains the pinned download and reservation metadata.

The production split contains 76/36/41 originals and 456/216/246 examples for training/validation/test. A separate language check uses ten expected examples from each of the five languages.
