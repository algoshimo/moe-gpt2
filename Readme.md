🧠MOE-GPT2
======

MOE-GPT2 replicates the GPT-2 architecture and replaces the original GPT-2 MLP with a sparse MoE (Mixture of Experts) architecture.

✨ Features
✅ Fully compatible with Hugging Face GPT-2 tokenizer and config

✅ Drop-in replacement of MLP with sparse MoE

✅ Configurable number of experts and top-k routing

✅ Support for loading pretrained GPT-2 weights

✅ Training and evaluation scripts included


🚀 Quick start
```
git clone https://github.com/yourusername/moe-gpt2.git
cd moe-gpt2

conda create --name moe-gpt2 python=3.10
conda activate moe-gpt2

python -m pip install -r requirements.txt

python src/train.py
python src/run.py
```

you can find the original GPT-2 architecture in the"gpt2" branch, which loads the pre-trained weight supported by huggingface
```
git checkout gpt2

python src/run.py
```

📌 TODO
 Support MoE layer sharing
 Expert dropout and regularization
 Inference-Ready Model Export
