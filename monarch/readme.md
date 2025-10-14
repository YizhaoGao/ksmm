# Compare with Monarch

## Experiment with a vit layer

### Monarch
- Monarch nblocks=2 val_loss=0.137379 compression = 1.60x
- Monarch nblocks=4 val_loss=0.431451 compression = 3.19x 
- Monarch nblocks=6 val_loss=0.576538 compression = 4.77x 
- Monarch nblocks=8 val_loss=0.660523 compression = 6.35x 


### Ksmm config
- [(6,128,128,1),(2,768,192,2)] val_loss=0.616252 compression = 3.43x
- [(6,64,64,2),(1,768,192,4)] val_loss=0.642746 compression = 3.69x
- [(6,128,128,1),(1,1536,384,2)] val_loss=0.260995 compression = 1.85x