# Workshop: Generative AI with Diffusion Models

*Based on NVIDIA DLI materials "Generative AI with Diffusion Models"*
*Adapted for Google Colab (T4 GPU) and local environments*
*Format: 3 × 90 minutes | 2 presenters: Theory (30 min) + Practical (60 min)*

*Prerequisite: Workshop "Rapid Application Development using Large Language Models"*
*or equivalent knowledge of deep learning and neural network architectures*

---

## Part 1 — U-Nets and Forward Diffusion (90 min)

### Theory (30 min)

- The image generation problem: why sampling from complex distributions is hard
- U-Net architecture: encoder path, decoder path, and skip connections
- Why skip connections matter: preserving spatial information across scales
- The diffusion idea: gradually adding noise until the image becomes pure noise
- Forward diffusion as a Markov chain: the noise schedule (beta values)
- Closed-form forward sampling: jumping directly to any noise level without iteration
- The DDPM framework: what the model learns to predict

### Practical (60 min)

- Setting up the Colab environment: `torch`, `torchvision`, `einops`
- Loading and visualizing FashionMNIST (16x16 grayscale)
- Building a U-Net from scratch: DownBlock, UpBlock, skip connections
- Training a simple denoiser (no time conditioning) and observing its limitations
- Implementing the noise schedule (linear beta schedule)
- Visualizing the forward diffusion process: from a clean image to pure noise
- Implementing closed-form forward sampling: `q(x_0, t)`
- DDPM training with time conditioning: first proper training run (2 epochs)

---

## Part 2 — Reverse Diffusion and Optimizations (90 min)

### Theory (30 min)

- Reverse diffusion: learning to remove noise step by step
- The noise prediction objective: why we predict noise instead of the image
- Reverse diffusion sampling: from pure noise to a clean image
- Architectural improvements that make diffusion models work well in practice:
  - Group Normalization: normalizing within groups of channels
  - GELU activation: smoother than ReLU for generative tasks
  - Rearrange pooling: efficient spatial downsampling with einops
  - Sinusoidal positional embeddings: encoding the timestep into the network
  - Residual connections: stable gradients in deep networks

### Practical (60 min)

- Implementing the reverse diffusion process: `reverse_q(x_t, t, e_t)`
- Generating FashionMNIST samples from pure noise with the basic model
- Improving the U-Net step by step:
  - Replacing activations with GELUConvBlock (GroupNorm + GELU)
  - Replacing pooling with RearrangePoolBlock (einops-based)
  - Adding sinusoidal time embeddings: SinusoidalPositionEmbedBlock
  - Wrapping convolutions in ResidualConvBlock
- Training the improved model (3 to 5 epochs)
- Comparing generation quality: basic vs. optimized U-Net

---

## Part 3 — Conditional Generation and CLIP (90 min)

### Theory (30 min)

- Why unconditional generation is not enough: we want control
- Class conditioning: giving the model a label to guide generation
- Classifier-free guidance (CFG): training with and without conditioning
  in the same model using a dropout trick
- The guidance weight w: interpolating between conditional and unconditional generation
- From class labels to text: using CLIP embeddings as conditioning signals
- CLIP recap: joint text-image embedding space (introduced in Workshop 1)
- Text-to-image diffusion: replacing one-hot class vectors with CLIP text encodings

### Practical (60 min)

- Implementing classifier-free guidance on FashionMNIST:
  generating specific clothing categories (shirts, trousers, sneakers)
- Visualizing the effect of different guidance weights on generation quality
- Loading the flowers dataset from Google Drive *(placeholder cell)*
- Loading pre-trained flowers model weights from Google Drive *(placeholder cell)*
- Generating flower images with the pre-trained model: daisy, sunflower, rose
- Short training demo on the flowers dataset (5 to 10 epochs): watching the model learn
- CLIP-conditioned diffusion: replacing class labels with CLIP text encodings
- Generating flowers from free-form text descriptions:
  "A round white daisy", "A bright yellow sunflower", "A dark red rose"

---

*Credits: Workshop materials adapted from NVIDIA Deep Learning Institute (DLI) course*
*"Generative AI with Diffusion Models". Original materials © NVIDIA Corporation.*
