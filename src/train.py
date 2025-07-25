import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024   #input序列最大长度(每个batch输入的最大token数)
    vocab_size: int = 50257  #词表大小，表示要预测的token种类数
    n_layer: int = 12      #有几个decoder_block
    n_head:int = 12        #多头注意力的头数  
    n_embd:int = 768     #每个token的向量长度

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        #3*n_embd 是因为直接算出q,k,v三个矩阵
        #每个矩阵为embd_size的原因是因为进行完attention后需要进行残差连接加上输入的矩阵，所以需要和输入矩阵的维度保持一致
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)  

        #投影层
        #每个头的矩阵大小是block_size * head_size 。 经过q*k.transpose后变为block_size*block_size，在经过与v矩阵乘法做矩阵乘法变为 block_size*head_size
        #之后我们把n_head个矩阵拼接，n个head_size拼为n_embd。
        # 但是我们不想生硬的拼接，所以拼接后又添加一个linear将他加权混合在一起，这也就是投影层   
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", 
                             torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #[B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class MOE(nn.Module):
    def __init__(self, config, num_experts=8, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = 4 * config.n_embd
        self.experts = nn.ModuleList([MLP(config) for _ in range(num_experts)])
        self.gate = nn.Linear(config.n_embd, num_experts)

    
    def forward(self, x):
        #x : [B, T, C]
        B, T, C = x.shape
        #x_flat : [B*T, C]
        x_flat = x.reshape(-1, C)

        gate_scores = self.gate(x_flat)
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)  #[B*T, top_k]

        top_k_gates = F.softmax(top_k_scores, dim=-1)  # (B*T, top_k)

        # MoE输出初始化
        moe_output = torch.zeros_like(x_flat)   #(B*T, C)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # (B*T,1)
            for j in range(self.num_experts):
                indices_j = (expert_idx == j).nonzero(as_tuple=False).view(-1)  # 在第i个序列(有top_k个序列，每个序列为(B*T,1) ) 中，第j个专家执行哪些token
                if indices_j.numel() == 0:   #若没有token被分配给第j个专家，直接跳过
                    continue
                input_j = x_flat[indices_j]   #所有分配给第j个专家的token的输入向量
                output_j = self.experts[j](input_j)  #送入第j个专家(mlp)，得出输出
                gate_j = top_k_gates[indices_j, i].unsqueeze(1)  # 取出这些token在gating对应的权重(softmax的结果)来做加权
                moe_output[indices_j] += gate_j * output_j
    
        return moe_output.view(B, T, C)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.moe = MOE(config, num_experts=2)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  #word token embeddings: 一个查找表，用于将输入的token映射到对应的向量表示
            wpe = nn.Embedding(config.block_size, config.n_embd),  #word position embeddings: 一个查找表, 给每个位置分配一个向量
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #存放n_layer个配置为config的子模块Block
            ln_f = nn.LayerNorm(config.n_embd), #对每个token的输出做归一化
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  #decoder_block最后的ffn的输出是和输入矩阵维度一样的。之后经过lm_head映射到对于每个token的打分来预测输出
        
        #lm_head是为每一个token对于词表中的所有token都打分，发生在所有encoder_block之后
        #对于训练过程。每一个token都打分选出预测的的token和原本的target计算损失(eg. 对于"我喜欢你": '我'的target是'喜欢', '喜欢'的target是'火锅'， '火锅'的target是<EOF>)
        #对于推理过程，我们只需要关注最后一个token的输出就好了。预测最后一个token的输出，接入之前的输出后继续推理

        #权重共享
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 
        pos_emb = self.transformer.wpe(pos)   
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if(targets == None):
            loss = targets
        else :
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

        

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

import tiktoken
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"epoch = {len(self.tokens) // (B * T) } batches")

        self.current_position = 0


    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model = GPT(GPTConfig())
model = model.to(device)

if __name__ == "__main__":    
    train_loader = DataLoaderLite(B=4, T=32)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    for i in range(50):
        x, y = train_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        torch.no_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        print(f"strep {i} : loss : {loss.item()}")

    import sys; sys = exit(0)