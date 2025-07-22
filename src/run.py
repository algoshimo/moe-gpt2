from train import model, device

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


num_return_sequences = 1
max_length = 30


torch.manual_seed(42)
torch.cuda.manual_seed(42)

enc = tiktoken.get_encoding('gpt2')
print("please input : >>> ", end="")
text = input()
tokens = enc.encode(text)
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

while x.size(1) < max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:,-1,:]   #logit原本为[B, T, vocab_size]， 现在对于第二维的token只取最后一个token，变为[B, vocab_size]
        probs = F.softmax(logits, dim=-1)  #dim=-1:在最后一维做softmax

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  #在最后一位取出最大的50个值及其坐标(这个坐标指这个token在词表中的坐标)
        ix = torch.multinomial(topk_probs, 1)  #按概率分布从每一行采样1个下标(注意不是取最大概率的下标)。 ix : [B, 1]
        xcol = torch.gather(topk_indices, -1, ix)  #根据采样到的下标从topk_indices中查找到这个token在词表中对应的位置
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)