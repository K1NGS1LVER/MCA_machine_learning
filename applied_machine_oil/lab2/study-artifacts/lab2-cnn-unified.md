# Lab 2 — Unified CNN: explain-the-lab (slowed down)

Goal (learner): **explain the lab and what they did. Deep understanding later.**

## Review queue
- [ ] 9-shape chain
- [ ] valid vs same
- [ ] in-trailing = channels, out-trailing = filters
- [ ] params formula (conv3=48120)

## Session log (slow mode, one fact per message)

### 2026-09-30

- Tk, Ch: confirmed (trailing = channels)
- L, Net: layer = function tensor in→out; network = chain
- K, C, Pd, Sl: kernel reused; filters = output maps; valid/same; pooling halves H,W, C unchanged
- Fl: flatten unrolls, counts preserved, 0 params. `(3,2,4) → (24,)` OK.
- Dd: all-to-all weighted sum, 120 inputs → each of 84 sees all 120
- Sm: makes scores sum to 1 (probabilities)
- LN (tail): (1,1,120) → flatten → (120,) → Dense(84) → (84,) → Dense(10)+softmax → (10,)

## Explained in plain words (defence-ready)
Lab 2 takes three views of CNNs and explains them step by step.

1. **TissueMNIST (grayscale)** — loaded medical images, normalized to 0-1. Showed hand-written
   convolution vs Keras Conv2D: they match (error ~1e-7). Compared fixed filters (Sobel/blur)
   vs trained filters (learned after training) vs random (before training). Trained a small CNN,
   printed shapes at each layer, looked at feature maps and a 32-D latent vector.

2. **RGB image (sons_of_anarchy.jpg)** — loaded color (3 channels). Applied filters per channel
   and a 3-channel filter. Showed random vs learned features; optionally MobileNetV2 gave
   early/mid/late learned features and a pooled latent vector.

3. **LeNet-5 on MNIST** — padded 28×28 to 32×32. Built classic LeNet (conv+pool, conv+pool, conv,
   flatten, dense 84, dense 10 softmax). Ran baseline, did bounded KerasTuner (4 trials),
   rebuilt winner from fresh weights and retrained, reported test accuracy and confusion matrix.
   Also traced activations and the latent vector.

The core idea: a **layer** is a function (tensor in → tensor out). **Conv** slides small kernels
across the grid (reuse) to find patterns; each filter produces one output map (C_out = filters).
**Pooling** shrinks the grid (halves), keeps channels same, has 0 params. **Flatten** unrolls the
volume into a vector. **Dense** connects everything to everything. **Softmax** turns scores into
probabilities (sum to 1). For LeNet: input (32,32,1), conv1 gives (28,28,6), pool1 (14,14,6),
conv2 (10,10,16), pool2 (5,5,16), conv3 (1,1,120), flatten (120,), dense84 (84,), dense10 (10,).

**Flatten (clear):** Unrolls the 3D grid into a straight 1D line. Same values, same order, only shape changes. Count = H×W×C. No weights → 0 params. Example: (2,2,3) → (12,).

Say "next slowly" for Dense, or "explain lab" for full answer.

next slowly
