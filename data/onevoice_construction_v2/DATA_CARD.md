# DATA CARD — OneVoice Construction v2

## Status
Synthetic bootstrap / pre-review.

## Primary intended use
Construction-domain specialization for an offline VI↔EN speech translation system:
terminology normalization, contextual biasing, intent/domain classification, MT adaptation experiments,
safety semantic evaluation, TTS/audio generation manifests, and ASR error analysis.

## Corpus design
The corpus distinguishes:
- core vs augmented data,
- train/dev/test,
- standard vs colloquial vs code-switch variants,
- safety-critical vs normal rows,
- semantic frame and slot labels,
- official-source scope vs synthetic translations.

## Safety caveat
No row in this dataset is a substitute for a project-specific method statement, safety plan, regulation,
or instruction from a competent person. Safety wording must be reviewed before deployment.

## Real-data requirement
The final OneVoice evaluation set should contain real noisy recordings from target worksites.
Synthetic data should be used to bootstrap/adapt the system, not to claim final field performance.
