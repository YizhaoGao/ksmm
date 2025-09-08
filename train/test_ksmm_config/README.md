# Test KSMM original config

This folder contains tests that use the vit config from ksmm original repo to test if it's a majic number

## Export a VIT layer
```bash
    python export_weights.py --input ckpt/smolvla/model.safetensors --layer model._orig_mod.vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.fc1.weight --output ckpt/vit_fc1_weights0.pth
```

The majic number for vit fc1 in ksmm is "6,64,64,1" "1,768,192,2", which is used for [1536, 384]. When using smolvla, the fc1 is [3072, 768]. 


## Train with random input/output
```bash 
    python train_with_random.py \
        --weight_path ../ckpt/vit_fc1_weights0.pth\
        --patterns "[(12,64,64,1),(2,768,192,2)]" 
```
Compression ratio: 3.69x
Loss 0.819313 -> 0.6059

```bash 
    python train_with_random.py \
        --weight_path ../ckpt/vit_fc1_weights0.pth\
        --patterns "[(6,64,64,2),(1,768,192,4)]" 
```
Compression ratio: 3.69x
Loss 0.8187 -> 0.605272


```bash 
    python train_with_random.py \
        --weight_path ../ckpt/vit_fc1_weights0.pth\
        --patterns "[(6,128,128,1),(1,1536,384,2)]" 
```
Compression ratio: 1.85x
Loss 0.629393 -> 0.260881