---
library_name: transformers
tags:
- language
- speech
- SpeechLID
license: mit
datasets:
- ARTPARK-IISc/Vaani
metrics:
- accuracy
- f1
base_model:
- openai/whisper-large-v3-turbo
---

# Model Card for Model ID
Indic Spoken Language Classifier supporting 42 languages.


# Model Details
The model consists of a whisper-large-v3-turbo encoder followed by a pooled-attention module and finally a linear classifier layer. Trained end-to-end on ARTPARK-IISc/Vaani dataset, capable of classifying 42 Indic languages.
The list of supported languages include: 
```python
["Surjapuri", 
"Marathi", 
"Assamese", 
"Haryanvi", 
"Halbi", 
"Malayalam", 
"Maithili", 
"Wancho", 
"Chhattisgarhi", 
"Punjabi", 
"Magahi", 
"Nepali", 
"Garhwali", 
"Garo", 
"Khortha", 
"Sumi", 
"Bajjika", 
"Marwari", 
"Telugu", 
"Nagamese", 
"Tulu", 
"Odia", 
"Urdu", 
"Kumaoni", 
"Kannada", 
"Tamil", 
"Sambalpuri", 
"Bengali", 
"Rajasthani", 
"English", 
"Malvani", 
"Chakma", 
"Surgujia", 
"Kokborok", 
"Khariboli", 
"Hindi", 
"Kurukh", 
"Angika", 
"Sadri", 
"Bhojpuri", 
"Konkani", 
"Gujarati"]
```



# How to Get Started with the Model
```python
from transformers import pipeline

pipe = pipeline(
    "audio-classification",
    model="ARTPARK-IISc/Vaani-LID_v0",
    trust_remote_code=True,
    # device=-1 # uncomment this line to run on cpu
)

out = pipe("path/to/16kHz/mono-channel/wav/file")
print(out)
```

## Evaluation
---
metrics:
- accuracy
model-index:
  results:
  - task:
    type: speech-classification
    dataset:
      name: FLEURS
      split: test
    metrics:
    - name: Accuracy
      value: 0.71
    
    dataset:
      name: Kathbath
      split: test
    metrics:
    - name: Accuracy
      value: 0.63

    dataset:
      name: Vaani
      split: test
    metrics:
    - name: Accuracy
      value: 0.77

---
# Citation

If you use this model, please cite the following:
```
@misc{pulikodan2026vaanicapturinglanguagelandscape,
      title={VAANI: Capturing the language landscape for an inclusive digital India}, 
      author={Sujith Pulikodan and Abhayjeet Singh and Agneedh Basu and Nihar Desai and Pavan Kumar J and Pranav D Bhat and Raghu Dharmaraju and Ritika Gupta and Sathvik Udupa and Saurabh Kumar and Sumit Sharma and Vaibhav Vishwakarma and Visruth Sanka and Dinesh Tewari and Harsh Dhand and Amrita Kamat and Sukhwinder Singh and Shikhar Vashishth and Partha Talukdar and Raj Acharya and Prasanta Kumar Ghosh},
      year={2026},
      eprint={2603.28714},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.28714}, 
}
```