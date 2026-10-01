# Model and runtime notices

Base: `openai/whisper-large-v3-turbo`, public OpenAI Whisper checkpoint, MIT license. Bundle the upstream model card and license notice with the merged weights if the downloaded snapshot supplies them.

Inference uses Transformers, PyTorch, NumPy, SoundFile, Librosa and Pandas from the official Lost in Transcription runtime. Training uses PEFT; merged inference does not require PEFT.
