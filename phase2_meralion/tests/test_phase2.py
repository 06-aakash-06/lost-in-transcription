import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from transformers import Gemma2Config,Gemma2ForCausalLM
from peft import LoraConfig,get_peft_model
from phase2_meralion.training.train_lora import loss_from_hidden
from phase2_meralion.analyze import correct_positions,aggregate

class Phase2Tests(unittest.TestCase):
    def small_model(self):
        return Gemma2ForCausalLM(Gemma2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=128)).eval()
    def test_chunked_loss_matches_gemma_causal_loss(self):
        torch.manual_seed(1);decoder=self.small_model()
        ids=torch.tensor([[2,3,4,5,6,1]])
        labels=ids.clone();labels[:,:3]=-100
        expected=decoder(input_ids=ids,labels=labels).loss
        from types import SimpleNamespace
        model=SimpleNamespace(text_decoder=decoder)
        hidden=decoder.model(input_ids=ids).last_hidden_state[:,2:-1,:]
        actual=loss_from_hidden(model,hidden,ids[:,3:],chunk=2)
        self.assertTrue(torch.allclose(expected,actual,atol=1e-6),(expected,actual))
    def test_decoder_inner_peft_preserves_embeddings_and_gradients(self):
        decoder=self.small_model();decoder.requires_grad_(False)
        decoder.model=get_peft_model(decoder.model,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
        decoder.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});decoder.train()
        self.assertIs(decoder.base_model.embed_tokens,decoder.get_input_embeddings())
        out=decoder(input_ids=torch.tensor([[2,3,4,1]]),labels=torch.tensor([[-100,3,4,1]]),use_cache=False)
        out.loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in decoder.parameters() if p.requires_grad))
        self.assertTrue(all('lora_' in name for name,p in decoder.named_parameters() if p.requires_grad))
    def test_word_complementarity_tracks_reference_positions(self):
        self.assertEqual(correct_positions('satu dua tiga','satu dua'),[True,True,False])
        self.assertEqual(correct_positions('satu dua','oh satu dua'),[True,True])
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS required')
    def test_nonfinite_mps_example_recomputed_on_cpu(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from phase2_meralion.training import train_lora as training
        decoder=self.small_model();decoder.requires_grad_(False)
        decoder.model=get_peft_model(decoder.model,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj']))
        decoder.to(device='mps',dtype=torch.bfloat16).train()
        params=[p for p in decoder.parameters() if p.requires_grad]
        for p in params:p.data=p.data.float();p.grad=torch.ones_like(p)
        for p in params:p.data.add_(0.000123)  # not representable in BF16
        weights=[p.detach().cpu().clone() for p in params]
        # Rotary frequencies normally retain FP32 in pretrained loading.
        for name,buf in decoder.named_buffers():
            if buf.is_floating_point():buf.data=buf.float()+0.000123
        buffers={name:buf.detach().cpu().clone() for name,buf in decoder.named_buffers()}
        identities=[id(p) for p in params]
        model=SimpleNamespace(text_decoder=decoder,config=SimpleNamespace(speech_token_index=63))
        cached={'input_ids':torch.tensor([[2,63,4]]),'speech':torch.zeros(1,1,16)}
        original=training.supervised_loss;calls=[]
        def injected(*args):
            calls.append(args[3])
            if len(calls)==1:return torch.tensor(float('nan'),device='mps')
            return original(*args)
        with patch.object(training,'supervised_loss',side_effect=injected):
            value,fallback=training.safe_backward(model,cached,[5,1],params,'mps',torch.bfloat16,2,16)
        self.assertTrue(fallback);self.assertTrue(value>0);self.assertEqual(calls,['mps','cpu'])
        self.assertEqual(identities,[id(p) for p in decoder.parameters() if p.requires_grad])
        self.assertTrue(all(torch.equal(p.detach().cpu(),w) for p,w in zip(params,weights)))
        self.assertTrue(all(buf.dtype==buffers[name].dtype and torch.equal(buf.detach().cpu(),buffers[name]) for name,buf in decoder.named_buffers()))
        self.assertTrue(all(p.device.type=='mps' and p.dtype==torch.float32 and torch.isfinite(p.grad).all() for p in params))

if __name__=='__main__':unittest.main()
