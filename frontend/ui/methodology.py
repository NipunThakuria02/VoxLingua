"""Short explanations and an accurate three-branch architecture illustration."""
from html import escape
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

SECTIONS={
 'Architecture':('01','ONE INPUT / THREE PERSPECTIVES','Three independent branches, one shared recording.'),
 'Datasets':('02','SPEECH + CONTROLLED CONDITIONS','Real speech, deliberate transformations, traceable sources.'),
 'Feature extraction':('03','WAVE / SPECTRUM / VECTOR','Measurable properties, expressed as 84 sound features.'),
 'Ensemble learning':('04','FOUR MODELS / TWO COMBINATIONS','Independent predictions, equal voting and learned stacking.'),
 'Evaluation & privacy':('05','EVIDENCE / LOCAL BY DESIGN','Held-out evaluation, honest limits and upstream attribution.'),
}

ARCHITECTURE_BRANCHES = {
    'Language identification': (
        'Audio to language ranking',
        'A mono 16 kHz copy goes to Vaani-LID, which ranks the supported languages. On selected Hindi and Tamil confusions, Whisper checks the alternative. The report shows the selected language and Vaani’s original scores.',
        {'LANGUAGE IDENTIFICATION'},
        'M180 210 L270 90 H710 L820 210 H830',
    ),
    'Distortion classification': (
        'Audio to recording condition',
        'The shared extractor describes the recording with 84 features. CART, Random Forest, AdaBoost and XGBoost score six conditions. Soft voting averages their scores, while stacking combines them with a trained logistic model for the final prediction.',
        {'84 SOUND FEATURES', 'CART · RF · ADA · XGB', 'VOTE / STACK'},
        'M180 210 H830',
    ),
    'Acoustic measurements': (
        'Audio to waveform, spectrum and pitch',
        'The original signal supplies the waveform, spectrum, digital level and clipping measurements. A separate mono copy supplies median voiced pitch. The report can leave pitch or estimated SNR unavailable when the signal provides too little evidence.',
        {'SIGNAL ANALYSIS'},
        'M180 210 L270 330 H710 L820 210 H830',
    ),
}

def architecture_html(reduced,selected):
    _, _, active, selected_path = ARCHITECTURE_BRANCHES[selected]
    pulse='' if reduced else f'<circle class="pulse" r="3" fill="#c3d1bc"><animateMotion dur="3.4s" repeatCount="indefinite" path="{selected_path}"/></circle>'
    def node(x,y,w,title,sub):
        focused=title in active
        opacity='1' if focused or title in ('RECORDING','UNIFIED REPORT') else '.28'
        edge='#c3d1bc' if focused else '#424642'
        return f'<g tabindex="0" opacity="{opacity}"><title>{escape(sub)}</title><rect x="{x}" y="{y}" width="{w}" height="60" rx="8" fill="#151515" stroke="{edge}" stroke-width="{2 if focused else 1}"/><text x="{x+w/2}" y="{y+25}" text-anchor="middle" fill="#ededeb" font-size="13">{escape(title)}</text><text x="{x+w/2}" y="{y+43}" text-anchor="middle" fill="#999" font-size="9">{escape(sub)}</text></g>'
    svg='<svg viewBox="0 0 1000 400" role="img" aria-label="Microphone or upload through validation to three independent analysis branches, then a unified report" preserveAspectRatio="xMidYMid meet" style="display:block;width:100%;height:100vh;font-family:Arial,sans-serif">'
    for _, _, _, path in ARCHITECTURE_BRANCHES.values():
        stroke, width = ('#c3d1bc', 2) if path == selected_path else ('#343b35', 1)
        svg+=f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{width}"/>'
    svg+=node(20,180,160,'RECORDING','Validated local audio')+node(270,60,440,'LANGUAGE IDENTIFICATION','Vaani-LID · optional Whisper second opinion')
    svg+=node(270,180,175,'84 SOUND FEATURES','Shared amplitude-preserving extractor')+node(480,180,170,'CART · RF · ADA · XGB','Four parallel classical classifiers')+node(685,180,115,'VOTE / STACK','Aligned six-class scores')
    svg+=node(270,300,440,'SIGNAL ANALYSIS','Original audio · native rate · deterministic measures')+node(830,180,150,'UNIFIED REPORT','Independent results + warnings')
    svg+=pulse+'<style>@media(prefers-reduced-motion:reduce){.pulse{display:none}}</style></svg>'
    return '<html><body style="margin:0;overflow:hidden;background:#090909" data-selected-branch="'+escape(selected)+'">'+svg+'</body></html>'


def flow_diagram(label, steps):
    nodes = ''.join(
        f'<li><strong>{escape(title)}</strong><span>{escape(detail)}</span></li>'
        for title, detail in steps
    )
    return (
        f'<section class="flow-lane"><p class="flow-label">{escape(label)}</p>'
        f'<ol class="flow-steps" role="list">{nodes}</ol></section>'
    )


def render(reduced=False):
    section=st.session_state.get('method_section')
    if section=='Evaluation, privacy & about':
        section=st.session_state.method_section='Evaluation & privacy'
    if section in SECTIONS:
        if st.button('← All methodology',key='method_back'): st.session_state.method_section=None;st.rerun()
        number,tag,description=SECTIONS[section]
        st.markdown(f'<div class="eyebrow">{escape(tag)}</div>',unsafe_allow_html=True)
        st.title(section);st.markdown(description)
        if section=='Architecture':
            selected=st.selectbox('Explore a branch',list(ARCHITECTURE_BRANCHES),key='architecture_branch')
            components.html(architecture_html(reduced,selected),height=390)
            heading, detail, _, _ = ARCHITECTURE_BRANCHES[selected]
            st.markdown(f'<section class="branch-explanation"><h3>{escape(heading)}</h3><p>{escape(detail)}</p></section>',unsafe_allow_html=True)
        elif section=='Datasets':
            st.subheader('A transformation is a label—not a diagnosis.')
            st.markdown('FLEURS supplies real read speech. Each retained original contributes one reference and five controlled variants: added noise, clipping, low volume, frequency filtering and interruption.')
            st.markdown('Original recordings and noise sources are partitioned **before augmentation**. Every original keeps exactly six samples. Speaker independence is not claimed without verified speaker metadata.')
            try:
                provenance=json.loads((Path(__file__).parents[1]/'data/pilot/manifests/provenance.json').read_text(encoding='utf-8'))
                count=int(provenance['retained_originals']);rows=int(provenance['rows']);noise=str(provenance['noise_corpus'])
                st.info(f'The completed development pilot contains {count} originals / {rows} samples. Its explicitly disclosed noise source is {noise}. Pilot findings do not establish production reliability. Production preparation remains pending.')
            except (OSError,ValueError,KeyError,TypeError):
                st.info('Pilot data counts are unavailable. Actual source counts appear only when verified manifests are present.')
            st.markdown('[FLEURS · CC BY 4.0](https://huggingface.co/datasets/google/fleurs) · [MUSAN · intended production source · CC BY 4.0](https://www.openslr.org/17/) · [ESC-10 pilot fallback · CC BY 3.0](https://github.com/karolpiczak/ESC-50)')
        elif section=='Feature extraction':
            st.subheader('From pressure waves to a compact description.')
            cols=st.columns(3)
            for col,title,body in zip(cols,['AMPLITUDE & TIME','FREQUENCY & MFCC','84 FEATURES'],['Level, crest, near-peak occupancy and internal gaps describe how the signal changes.','Spectral shape and Mel-frequency summaries describe how energy is distributed.','A single versioned extractor supplies the same finite feature order during training and inference.']):
                with col: st.markdown('### '+title);st.write(body)
            st.caption('No filename, language label, source ID, duration or injected SNR enters the classifier feature vector. Pitch and estimated SNR are separate acoustic outputs.')
        elif section=='Ensemble learning':
            st.caption('Unit 5 · Combining Models')
            st.markdown(
                '<ul class="ensemble-models" role="list" aria-label="The four base classifiers">'
                '<li><strong>CART</strong><span>Single decision tree</span></li>'
                '<li><strong>Random Forest</strong><span>Bagging</span></li>'
                '<li><strong>AdaBoost</strong><span>Boosting</span></li>'
                '<li><strong>XGBoost</strong><span>Gradient boosting</span></li></ul>',
                unsafe_allow_html=True,
            )
            voting,stacking=st.tabs(['Soft voting','Grouped stacking'])
            with voting:
                st.markdown(flow_diagram('Analyse a recording', (
                    ('Four score vectors', 'One six-class prediction from each base model.'),
                    ('Equal-weight average', 'Each model contributes 25% to every class score.'),
                    ('Predicted condition', 'Choose the class with the highest average score.'),
                )),unsafe_allow_html=True)
            with stacking:
                training=flow_diagram('Offline training', (
                    ('Grouped folds', 'Keep every sample from the same original in one fold.'),
                    ('Out-of-fold scores', 'Predict each fold with models fitted on the other folds.'),
                    ('Fit the combiner', 'Train logistic regression on the held-out scores.'),
                ))
                inference=flow_diagram('Analyse a recording', (
                    ('Refitted base models', 'Each model, refitted on all training data, supplies scores.'),
                    ('Fitted combiner', 'Logistic regression combines the four score vectors.'),
                    ('Predicted condition', 'Choose the class with the highest stacking score.'),
                ))
                st.markdown(training+inference,unsafe_allow_html=True)
            st.caption('Research Lab compares both methods against the individual classifiers and a threshold baseline. Stacking supplies the app’s final condition prediction.')
        else:
            st.subheader('Useful evidence, clearly bounded.')
            st.markdown('The locked test is never used for model training or threshold tuning. Controlled semi-synthetic evaluation and real microphone checks are different evidence. A small pilot does not establish real-world reliability.')
            st.markdown('Audio is processed locally with no default disk retention or external inference. Clear removes the application session’s recording and results. Microphone access starts only on your action; browser processing may affect measurements.')
            st.markdown('A language-region pin illustrates where a language is commonly spoken. It never identifies your location, origin or identity. Code-switching is unsupported, model scores are uncalibrated, and multiple audio problems can coexist.')
            st.markdown('[Vaani-LID_v0 · ARTPARK-IISc · MIT](https://huggingface.co/ARTPARK-IISc/Vaani-LID_v0) — pretrained upstream model, with the requested Pulikodan et al. VAANI citation. [Whisper large-v3-turbo · OpenAI · MIT](https://huggingface.co/openai/whisper-large-v3-turbo) is its underlying model dependency.')
            st.caption('Not a certified audio-quality, medical or forensic tool. Full source attribution is retained with the project.')
        return
    st.markdown('<div class="eyebrow">THE KNOWLEDGE HUB</div>',unsafe_allow_html=True);st.title('Behind the signal.')
    st.markdown('<p class="lede">A closer look at the data, models and decisions that shape every result.</p>',unsafe_allow_html=True)
    groups=[list(SECTIONS)[:3],list(SECTIONS)[3:]]
    for group in groups:
        for col,title in zip(st.columns(len(group),gap='medium'),group):
            number,drawing,description=SECTIONS[title]
            with col:
                st.markdown(f'<div class="method-card"><div class="drawing">{drawing}</div><h3>{escape(title)}</h3><p>{description}</p></div>',unsafe_allow_html=True)
                if st.button('Explore '+title+'  →',key='method_'+number,use_container_width=True): st.session_state.method_section=title;st.rerun()
