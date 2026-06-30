# Generative AI with Diffusion Models

A hands-on workshop based on NVIDIA Deep Learning Institute (DLI) materials,
adapted for Google Colab (T4 GPU).

> **Credits:** Original course materials by NVIDIA Corporation / NVIDIA Deep Learning Institute.
> Workshop adapted and delivered by:
> - **Theory:** Domen Verber
> - **Practical:** Jani Dugonik
>
> Faculty of Electrical Engineering and Computer Science (FERI),
> University of Maribor, Slovenia

---

## Prerequisite

This workshop assumes the knowledge covered in
"Rapid Application Development using Large Language Models"
(or equivalent knowledge of deep learning and neural network architectures).

---

## Workshop Structure

3 parts, 90 minutes each (30 min theory + 60 min practical).

| Part | Topic | Key concepts |
|---|---|---|
| 1 | U-Nets and Forward Diffusion | U-Net, skip connections, noise schedule, DDPM |
| 2 | Reverse Diffusion and Optimizations | Reverse sampling, GroupNorm, GELU, sinusoidal time embeddings, residual blocks |
| 3 | Conditional Generation and CLIP | Classifier-free guidance, CLIP, text-to-image |

See [`workshop_program.md`](workshop_program.md) for the full program with theory and practical topics per part.

---

## Repository Structure

```
├── notebooks/
│   ├── Part1_UNets_ForwardDiffusion.ipynb
│   ├── Part2_ReverseDiffusion_Optimizations.ipynb
│   └── Part3_ConditionalGeneration_CLIP.ipynb
├── workshop_program.md
├── LICENSE.md
└── README.md
```

All notebooks are self-contained: no external utility files are required.
Every building block (U-Net layers, noise schedule, embedding blocks) is defined
directly inside the notebook.

---

## Requirements

- Google Colab with **T4 GPU** (Runtime > Change runtime type > T4 GPU)
- Google Drive access for Part 2 and Part 3 (loading pre-trained model checkpoints)
- No API keys needed

---

## Pre-Workshop Preparation (presenter only)

Parts 2 and 3 rely on model checkpoints trained before the workshop,
since full training would take too long during a live 60-minute session.

### Before Part 2

Train the optimized U-Net for 50 epochs on FashionMNIST and save to Google Drive:

```python
torch.save(model_opt50.state_dict(), 'unet_50epochs.pth')
```

Upload to: `My Drive/BestCourse2026/workshop-genai/unet_50epochs.pth`

Training time on T4: approximately 17 minutes.

### Before Part 3

1. Train the CFG-conditioned flowers model with CLIP context for 100 epochs
   (run the cell marked `PRE-WORKSHOP` in `Part3_ConditionalGeneration_CLIP.ipynb`)
2. Save to: `My Drive/BestCourse2026/workshop-genai/flowers_clip.pth`
3. Make sure the flowers dataset is available at:
   `My Drive/BestCourse2026/workshop-genai/data/cropped_flowers/`
   with subfolders `daisy/`, `sunflowers/`, `roses/`

Training time on T4: approximately 30 minutes for 100 epochs.

---

## How to Use

### Option A: Open directly in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File > Open notebook > GitHub**
3. Paste this repository URL and select the notebook

### Option B: Download and upload manually

1. Download the `.ipynb` file from this repository
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. Click **File > Upload notebook**

---

## License

Workshop materials adapted from NVIDIA DLI course
"Generative AI with Diffusion Models".
Original materials Copyright NVIDIA Corporation.

Adaptations for Google Colab by Domen Verber and Jani Dugonik,
FERI, University of Maribor.
