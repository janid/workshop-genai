# Workshop: Generative AI with Diffusion Models

*Based on NVIDIA DLI materials "Generative AI with Diffusion Models"*
*Adapted for Google Colab (T4 GPU) and local environments*
*Format: 3 × 90 minutes | 2 presenters: Theory (30 min) + Demo (30 min) + Practice Time (30 min)*

*Prerequisite: Workshop "Rapid Application Development using Large Language Models"*
*or equivalent knowledge of deep learning and neural network architectures*

---

## Part 1 — U-Nets and Forward Diffusion (90 min)

### Theory (30 min) | slides: slides1, slides2

**A brief history of generative models** *(slides1: 8-19)*
- From early AI chatbots to deep dreaming
- GANs: generator + discriminator, the adversarial game
- Why GANs are hard to train: mode collapse, instability

**U-Net architecture** *(slides1: 22-25)*
- From encoder-decoder (autoencoder) to U-Net
- Skip connections: why they preserve spatial detail
- Channel dimensions through the network

**Transposed convolution** *(slides1: 26-36)*
- How to upsample with learned weights
- Stride and padding in transposed convolution
- The checkerboard artifact (preview, covered in depth in Part 2)

**The experiment: FashionMNIST** *(slides1: 45-47)*
- Dataset overview and the generation hypothesis
- Our U-Net architecture for 16x16 grayscale images

**Forward diffusion** *(slides2: 3-11)*
- Thermodynamic inspiration: ink dissolving in water
- Forward diffusion step by step: adding noise at each timestep
- The noise schedule: beta values from 0.0001 to 0.02
- Closed-form forward sampling: jumping directly to any timestep

### Demo (30 min)

Instructor-led, presented end to end:

- Setting up the environment: `torch`, `torchvision`, `matplotlib`
  (pre-installed on Colab, local install cell provided at the top of the
  notebook)
- Loading and visualizing FashionMNIST (16x16 grayscale)
- Building a U-Net from scratch: `DownBlock`, `UpBlock`, skip connections
- Training a simple denoiser without time conditioning, observing its
  limitations (fixed noise level, no time information)
- Implementing the noise schedule (linear beta schedule)
- Visualizing the forward diffusion process: from a clean image to pure
  noise, recursive vs. closed-form
- Implementing closed-form forward sampling `q(x_0, t)`
- First DDPM training run with time conditioning (2 epochs)

### Practice Time (30 min)

Students rebuild key pieces themselves, with hidden solutions for
self-checking:

1. **`UpBlock`** (7 min): rebuild the up-path block from memory, including
   why the skip connection doubles the input channel count.
2. **`add_noise`** (7 min): rebuild the noise-mixing function from memory.
3. **Closed-form forward diffusion** (9 min): rebuild `q(x_0, t)` from
   memory.
4. **Put it together** (7 min): assemble a mini U-Net using their own
   `UpBlock`, then run their own `q(x_0, t)` and visualize the noise
   increasing with `t`.

---

## Part 2 — Reverse Diffusion and Optimizations (90 min)

### Theory (30 min) | slides: slides2, slides3

**Reverse diffusion** *(slides2: 12-16)*
- Why reversing forward diffusion step by step is intractable directly
- Using Bayes' rule to derive the reverse process
- The noise prediction objective: predicting the added noise, not the image

**Model training and time conditioning** *(slides2: 17-23)*
- Training loop: sample a timestep, add noise, predict noise, compute MSE loss
- Adding time to the U-Net: time embeddings via broadcasting

**The checkerboard problem** *(slides3: 3-6)*
- Stride artifacts in transposed convolution
- Kernel size 2 with stride 2 as a fix

**Group Normalization** *(slides3: 7-12)*
- Batch norm review and its limitations for small batches
- Group norm: normalizing within channel groups
- Instance norm and layer norm as special cases

**Sinusoidal positional embeddings for time** *(slides3: 20-24)*
- Why one-hot encoding for timesteps is inefficient
- Unit circle analogy: encoding time as angles
- Multiple frequencies: the sinusoidal embedding formula

**Deeper networks** *(slides3: 25-28)*
- Adding more convolution layers to each Up/Down block
- Residual connections: stabilizing gradients in deep networks

### Demo (30 min)

Instructor-led, presented end to end:

- Recap from Part 1: dataset, noise schedule, basic U-Net classes
  (same code, fresh kernel)
- Implementing the reverse diffusion process: `reverse_q(x_t, t, e_t)`
- Generating FashionMNIST samples from pure noise with the basic model
  (3 epochs)
- Improving the U-Net step by step:
  - `GELUConvBlock`: GroupNorm + GELU (fixes small-batch instability,
    better gradients than ReLU)
  - `RearrangePoolBlock`: einops-based pooling (no information loss,
    unlike MaxPool)
  - `SinusoidalPositionEmbedBlock`: richer time encoding via multiple
    frequencies
  - `ResidualConvBlock`: deeper per-block networks with skip connections
- Assembling the optimized U-Net (`UpBlockOpt` uses kernel=2, stride=2 to
  remove the checkerboard artifact directly)
- Training the optimized model (3 epochs) and comparing against the basic
  model
- Loading a 50-epoch pre-trained checkpoint from Google Drive for a
  three-way quality comparison (architecture and training time, isolated)

### Practice Time (30 min)

1. **`reverse_q`** (7 min): rebuild the reverse diffusion step from memory.
2. **`GELUConvBlock`** (7 min): rebuild Conv2d + GroupNorm + GELU from
   memory.
3. **`UpBlockOpt`** (9 min): build the optimized up-path block using their
   own `GELUConvBlock`, including the kernel=2/stride=2 fix.
4. **Put it together** (7 min): assemble a mini optimized U-Net using
   their own `UpBlockOpt`, train for 1 epoch, and generate samples with
   their own `reverse_q`.

---

## Part 3 — Conditional Generation and CLIP (90 min)

### Theory (30 min) | slides: slides4, slides5

**Adding context to the U-Net** *(slides4: 3-6)*
- The problem with unconditional generation: no control over the output
- Adding context embeddings alongside time embeddings
- How context and time embeddings are multiplied into the feature maps

**Classifier-free guidance** *(slides4: 7-14)*
- Why training a separate classifier is expensive
- The dropout trick: randomly drop context during training (Bernoulli mask)
- Weighted reverse diffusion: combining conditional and unconditional predictions
- Guidance weight w: from negative (opposite class) to large positive (exaggerated class)
- Visual results at different w values

**The flowers dataset** *(slides4: 15-16)*
- 3 classes: daisy, sunflower, rose
- Images cropped and resized to 32x32 RGB

**CLIP** *(slides5: 3-10)*
- Matching text to images: is it possible?
- Cosine similarity as a measure of vector alignment
- CLIP training: contrastive loss on 400M image-text pairs
- What CLIP learns: a shared embedding space for text and images

**From class labels to CLIP context** *(slides5: 11-16)*
- One-hot class label → CLIP text encoding (512 features)
- The same Bernoulli mask dropout trick applies
- The model architecture does not change, only the context vector changes

### Demo (30 min)

Instructor-led, presented end to end:

- `UNetCFG`: the same optimized architecture from Part 2, plus context
  embeddings. Transfer weights from the Part 2 50-epoch checkpoint, then
  fine-tune with class conditioning (3 epochs)
- Generating specific clothing categories at different guidance weights,
  observing the effect of w from unconditional (w=0) to strongly
  exaggerated (w=2.0)
- CLIP: loading the model, comparing text encodings of flower descriptions
  with a cosine similarity heatmap
- Loading the pre-trained CLIP-conditioned flowers model from Google Drive
  (trained before the workshop, 100 epochs, about 30 min on a T4, not run
  live)
- Generating flowers from free-form text: "A round white daisy", "A bright
  yellow sunflower"
- Open floor: inviting the room to call out their own text prompts and
  generating live

### Practice Time (30 min)

This session's Practice Time combines ideas from all three parts:

1. **CFG formula** (7 min): rebuild the classifier-free guidance
   combination as a standalone function, from memory.
2. **CLIP text encoding** (7 min): rebuild a tokenize + encode + normalize
   function, from memory.
3. **Negative prompt guidance** (9 min): a new synthesis task, not pure
   recall. Push generation away from a negative text prompt instead of
   the unconditional prediction, using the same formula shape as Task 1.
4. **Put it together** (7 min): pick a positive/negative prompt pair,
   generate with both standard CFG and negative-prompt guidance, and
   compare side by side.

---

## Wrap-up | slides6

Slides6 covers: VAE, latent diffusion, state-of-the-art models (DALL-E, Stable Diffusion),
tiled upsampling, and trustworthy AI.

These slides are suitable for the closing discussion after the three practical parts,
time permitting. Recommended selection:

- **Latent diffusion** *(slides6: 10)*: how Stable Diffusion uses a VAE to work in latent space
  instead of pixel space (explains why it is faster and higher resolution than our pixel-space models)
- **State of the art** *(slides6: 8-9)*: how diffusion + VAE + CLIP + GAN all combine in modern systems
- **Trustworthy AI** *(slides6: 14-16)*: licenses, content credentials, responsible generation

---

## Optional Homework (end of workshop or take-home)

Three self-contained options, up to 60 minutes each, no GPU required. See
`homework/` for full instructions and starter notebooks.

- **Option A: Toy 2D Diffusion** - rebuild the full pipeline on 2D points
  instead of images.
- **Option B: Conditional Constellation Generator** - 2D points again, but
  combines all three parts: forward diffusion, reverse diffusion, and
  classifier-free guidance.
- **Option C: Flower Prompt Gallery and Curation** - lighter on code,
  reuses the pretrained Part 3 flowers model, focused on prompt design
  and write-up.

---

*Credits: Workshop materials adapted from NVIDIA Deep Learning Institute (DLI) course*
*"Generative AI with Diffusion Models". Original materials © NVIDIA Corporation.*
