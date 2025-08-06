---
pipeline_tag: robotics
tags:
- smolvla
library_name: lerobot
datasets:
- lerobot/svla_so101_pickplace
---

## SmolVLA: A vision-language-action model for affordable and efficient robotics

Resources and technical documentation:

[SmolVLA Paper](https://huggingface.co/papers/2506.01844)

[SmolVLA Blogpost](https://huggingface.co/blog/smolvla)

[Code](https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/smolvla/modeling_smolvla.py)

[Train using Google Colab Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb#scrollTo=ZO52lcQtxseE)

[SmolVLA HF Documentation](https://huggingface.co/docs/lerobot/smolvla)

Designed by Hugging Face.

This model has 450M parameters in total.
You can use inside the [LeRobot library](https://github.com/huggingface/lerobot).

Before proceeding to the next steps, you need to properly install the environment by following [Installation Guide](https://huggingface.co/docs/lerobot/installation) on the docs.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
python lerobot/scripts/train.py \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

Example of finetuning the smolvla neural network with pretrained VLM and action expert
intialized from scratch:
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=lerobot/svla_so101_pickplace \
  --batch_size=64 \
  --steps=200000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```