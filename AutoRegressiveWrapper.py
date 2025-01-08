import torch
from torch import nn
import torch.nn.functional as F


def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])  
    val, ind = torch.topk(logits, k)         
    probs = torch.full_like(logits, float('-inf')) 
    probs.scatter_(1, ind, val)  
    return probs  


class AutoRegressiveWrapper(nn.Module):
    def __init__(self, net, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.model = net
        self.max_seq_len = net.max_seq_len  

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1.0, filter_thres=0.9):
        self.model.eval()
        device = start_tokens.device  

        
        if start_tokens.dim() == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape 
        prev_out = start_tokens  

        for _ in range(seq_len):
            x = prev_out[:, -self.max_seq_len:]  
            logits = self.model(x)[:, -1, :]  
            
            filtered_logits = top_k(logits, thres=filter_thres)  
            probs = F.softmax(filtered_logits / temperature, dim=-1)  
            predicted_char_token = torch.multinomial(probs, 1) 
            
            out = torch.cat((prev_out, predicted_char_token), dim=-1)  
            prev_out = out  

            if eos_token is not None and (predicted_char_token == eos_token).all():  
                break

        out = out[:, t:]  
        if start_tokens.dim() == 1:
            out = out.squeeze(0)
        return out

    def forward(self, x):
        xi = x[:, :-1]  
        xo = x[:, 1:]   
        
        out = self.model(xi)  
        logits_reorg = out.view(-1, out.size(-1))  
        targets_reorg = xo.reshape(-1)  
        
        loss = F.cross_entropy(logits_reorg, targets_reorg)  
        return loss
