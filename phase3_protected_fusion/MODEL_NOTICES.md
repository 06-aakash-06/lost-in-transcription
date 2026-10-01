# Final ASR system notices

Creator of this adaptation: Aakash, using the independently implemented training and fusion code in this repository. Modified models: Whisper large-v3-turbo with domain LoRA merged into pretrained weights; MERaLiON-3-3B-ASR with decoder LoRA merged into pretrained weights. Inference uses a deterministic protected word fusion rule. No endorsement by original model developers is implied.

Whisper: Copyright OpenAI, MIT licence. Full licence accompanies this package as WHISPER_LICENSE.txt.

MERaLiON is developed by the Agency for Science Technology And Research (A*STAR), Singapore. Copyright 2026 AGENCY FOR SCIENCE TECHNOLOGY AND RESEARCH. The development of this ASR system was assisted by MERaLiON, an AI model developed by A*STAR. The full upstream MERaLiON public licence, including its Gemma Terms of Use annex, is preserved under models/meralion/. The MIT grant and applicable upstream terms govern the adaptation; this notice supplements rather than replaces those terms.

The custom MERaLiON model/processor code, original tokenizer conventions, processor files and merged model weights are bundled for local offline inference. The new fusion/entrypoint code is provided under the MIT terms in the accompanying WHISPER_LICENSE.txt, with copyright 2026 Aakash for those modifications. All models and code are provided without warranties. Public model sources: openai/whisper-large-v3-turbo and MERaLiON/MERaLiON-3-3B-ASR on Hugging Face.
