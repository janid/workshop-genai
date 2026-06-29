# Workshop: Generative AI with Diffusion Models

*Based on NVIDIA DLI materials "Generative AI with Diffusion Models"*
*Adapted for Google Colab (T4 GPU) and local environments*
*Format: 3 × 90 minutes | 2 presenters: Theory (30 min) + Practical (60 min)*

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

### Practical (60 min)

- Setting up the Colab environment: `torch`, `torchvision`, `einops`
- Loading and visualizing FashionMNIST (16x16 grayscale)
- Building a U-Net from scratch: DownBlock, UpBlock, skip connections
- Training a simple denoiser without time conditioning: observing its limitations
- Implementing the noise schedule (linear beta schedule)
- Visualizing the forward diffusion process: from a clean image to pure noise
- Implementing closed-form forward sampling `q(x_0, t)`
- First DDPM training run with time conditioning (2 epochs)

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

### Practical (60 min)

- Implementing the reverse diffusion process: `reverse_q(x_t, t, e_t)`
- Generating FashionMNIST samples from pure noise with the basic model
- Improving the U-Net step by step:
  - `GELUConvBlock`: GroupNorm + GELU (fixes checkerboard, better gradients)
  - `RearrangePoolBlock`: einops-based pooling (no information loss)
  - `SinusoidalPositionEmbedBlock`: richer time encoding
  - `ResidualConvBlock`: deeper per-block networks with skip connections
- Training the improved model (3 to 5 epochs)
- Comparing generation quality: basic model vs. optimized model

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

### Practical (60 min)

- Implementing classifier-free guidance on FashionMNIST
- Generating specific clothing categories (shirts, trousers, sneakers)
- Visualizing the effect of different guidance weights on generation quality
- Loading the flowers dataset from Google Drive *(placeholder cell)*
- Loading pre-trained flowers model weights from Google Drive *(placeholder cell)*
- Generating flower images with the pre-trained model (daisy, sunflower, rose)
- Short training demo on the flowers dataset (5 to 10 epochs): watching the model learn
- CLIP-conditioned diffusion: replacing class labels with CLIP text encodings
- Generating flowers from free-form text: "A round white daisy", "A bright yellow sunflower"

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

*Credits: Workshop materials adapted from NVIDIA Deep Learning Institute (DLI) course*
*"Generative AI with Diffusion Models". Original materials © NVIDIA Corporation.*
