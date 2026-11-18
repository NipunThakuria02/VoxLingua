# VoxLingua
### Local Voice Intelligence through Ensemble Learning

VoxLingua analyzes a short speech recording to identify its language, classify its recording condition, and display acoustic measurements. The application combines language recognition, recording analysis, and interactive audio visualization in one interface.

## Features

- Language Lab with WAV upload, microphone capture, language rankings, and a rotating language-region globe.
- Voice Lab with waveform playback, spectrograms, voiced pitch, signal measurements, and recording-condition classification.
- Research Lab for exploring saved experiments and model outputs.
- Methodology views with system architecture and ensemble learning diagrams.
- A shared recording session across the labs and local audio processing.

## Ensemble Learning

CART, Random Forest, AdaBoost, and XGBoost classify six recording conditions using 84 acoustic features. Soft voting combines their scores equally, while grouped stacking uses a logistic regression combiner.

The six conditions are reference, added noise, clipping, low volume, frequency filtering, and interruption. Language recognition uses pretrained ARTPARK-IISc Vaani-LID and an OpenAI Whisper second opinion for selected Hindi and Tamil predictions.

## Datasets

- [Google FLEURS](https://huggingface.co/datasets/google/fleurs) supplies English, Hindi, Tamil, Bengali, and Telugu speech.
- [MUSAN](https://www.openslr.org/17/) supplies environmental noise for the production training workflow.
- [ESC-10](https://github.com/karolpiczak/ESC-50) supplies environmental noise for the development pilot.

Preparation verifies source hashes, screens reference recordings, assigns original recordings to splits, generates controlled audio transformations, and extracts a shared feature representation. Source revisions, licenses, and preparation details are documented in [data/README.md](data/README.md).

## Tech Stack

| Area | Technologies |
| --- | --- |
| Language | Python |
| Interface | Streamlit, HTML, CSS, JavaScript |
| Visualization | Plotly, Canvas API |
| Audio capture | Web Audio API, AudioWorklet |
| Audio processing | NumPy, SciPy, librosa, SoundFile |
| Machine learning | scikit-learn, XGBoost |
| Language inference | PyTorch, Transformers |
| Model storage | Joblib, Safetensors |
| Configuration | YAML, JSON |
| Testing | pytest, Streamlit AppTest |

## Project Organization

~~~text
frontend/        Interface modules, styles, globe and recorder
src/voxlingua/   Audio processing and inference services
ml/             Dataset preparation, training and evaluation
data/           Dataset documentation and pinned source metadata
models/         Local runtime artifacts and snapshot metadata
configs/        Runtime configuration
static/         Local fonts and license notices
scripts/        Local setup utilities
tests/          Automated checks
docs/           Third-party attribution
~~~
