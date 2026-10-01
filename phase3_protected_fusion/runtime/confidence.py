"""Teacher-forced acoustic log probabilities, never reference-conditioned.

Token probabilities are aggregated by geometric mean into whitespace words.
Only full <=30s audio is scored. Longer clips retain the anchor or use proven
sequence/support safeguards; their first 30s is never used as full-clip evidence.
"""
import math,re
import torch

def token_to_words(text,offsets,token_logp):
    spans=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
    values=[[] for _ in spans]
    for (start,end),lp in zip(offsets,token_logp):
        if start==end:continue
        for i,(s,e) in enumerate(spans):
            if start<e and end>s:values[i].append(float(lp))
    return [sum(x)/len(x) if x else None for x in values]

@torch.inference_mode()
def score_pair(engine,wave,texts):
    if len(wave)>30*16000:return [[None]*len(text.split()) for text in texts]
    tokenizer=engine.processor.tokenizer
    tokenizer.set_prefix_tokens(language=engine.language,task='transcribe',predict_timestamps=False)
    encoded=[tokenizer(text,return_offsets_mapping=True,add_special_tokens=True) for text in texts]
    features=engine.processor.feature_extractor(wave,sampling_rate=16000,return_tensors='pt').input_features.to(engine.device,dtype=engine.dtype)
    model=engine.model.get_base_model() if hasattr(engine.model,'get_base_model') else engine.model
    encoder=model.model.encoder(features,return_dict=True).last_hidden_state
    result=[]
    for text,tokens in zip(texts,encoded):
        ids=tokens['input_ids']
        if len(ids)>model.config.max_target_positions:
            result.append([None]*len(text.split()));continue
        decoder=torch.tensor([ids[:-1]],device=engine.device)
        hidden=model.model.decoder(input_ids=decoder,encoder_hidden_states=encoder,use_cache=False,return_dict=True).last_hidden_state
        labels=torch.tensor(ids[1:],device=engine.device)
        scores=[]
        for start in range(0,len(labels),64):
            logits=model.proj_out(hidden[:,start:start+64]).float()[0]
            lp=logits.gather(1,labels[start:start+64,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)
            if not torch.isfinite(lp).all():raise FloatingPointError('nonfinite confidence scores')
            scores.extend(lp.cpu().tolist())
        result.append(token_to_words(text,tokens['offset_mapping'][1:],scores))
    return result
