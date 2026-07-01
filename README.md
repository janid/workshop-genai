# Generative AI with Diffusion Models

A hands-on workshop based on NVIDIA Deep Learning Institute (DLI) materials,
adapted for Google Colab (T4 GPU) and for local environments.

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

3 parts, 90 minutes each (30 min theory + 30 min demo + 30 min practice time).

| Part | Topic | Key concepts |
|---|---|---|
| 1 | U-Nets and Forward Diffusion | U-Net, skip connections, noise schedule, DDPM |
| 2 | Reverse Diffusion and Optimizations | Reverse sampling, GroupNorm, GELU, sinusoidal time embeddings, residual blocks |
| 3 | Conditional Generation and CLIP | Classifier-free guidance, CLIP, text-to-image |

Each part follows the same pattern: the instructor presents a working demo
end to end, then students get 30 minutes of Practice Time to rebuild key
pieces themselves, with hidden solutions in the notebook for self-checking.

See [`workshop_program.md`](workshop_program.md) for the full program with theory, demo, and practice topics per part.

---

## Repository Structure

```
├── notebooks/
│   ├── Part1_UNets_ForwardDiffusion.ipynb
│   ├── Part2_ReverseDiffusion_Optimizations.ipynb
│   └── Part3_ConditionalGeneration_CLIP.ipynb
├── homework/
│   ├── Homework_OptionA_ToyDiffusion.ipynb
│   ├── Homework_OptionA_ToyDiffusion.md
│   ├── Homework_OptionB_ConditionalConstellation.ipynb
│   ├── Homework_OptionB_ConditionalConstellation.md
│   ├── Homework_OptionC_FlowerGallery.ipynb
│   └── Homework_OptionC_FlowerGallery.md
├── workshop_program.md
├── LICENSE.md
└── README.md
```

All notebooks are self-contained: no external utility files are required.
Every building block (U-Net layers, noise schedule, embedding blocks) is defined
directly inside the notebook.

---

## Requirements

### Option A: Google Colab (recommended)

- Colab with **T4 GPU** (Runtime > Change runtime type > T4 GPU)
- Google Drive access for Part 2 and Part 3 (loading pre-trained model checkpoints)
- No API keys needed
- `torch`, `torchvision`, and `matplotlib` are pre-installed. Part 2 installs
  `einops` and Part 3 installs CLIP automatically in their first cells.

### Option B: Local environment

Each notebook's first code cell checks whether it is running on Colab and,
if not, installs what is missing:

```python
# If running locally (not on Colab), install the required packages.
import sys
if 'google.colab' not in sys.modules:
    %pip install torch torchvision matplotlib numpy
```

Part 2 additionally needs `einops` (already installed unconditionally by
its own cell, on Colab or locally). Part 3 additionally needs `seaborn`
(ships with Colab by default, but not with a plain local Python install,
so it is included in Part 3's local-install cell) and CLIP (already
installed unconditionally by its own cell).

A GPU is recommended for Parts 2 and 3 but not required, everything also
runs on CPU, just slower. Part 1's image resolution (16x16) and Part 3's
2D-point homeworks are deliberately small so a CPU is enough to follow
along.

Local runs still need Google Drive access for the pre-trained checkpoints
in Parts 2 and 3 (see below), either by syncing the relevant folder
locally or downloading the `.pth` files manually.

---

## Pre-Workshop Preparation (presenter only)

Parts 2 and 3 rely on model checkpoints trained before the workshop,
since full training would take too long during a live 30-minute demo.

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

Both checkpoints are required before the session starts, there is no live
fallback if either is missing.

---

## Homework (Optional)

Three self-contained options for the end of the workshop or as take-home
work, each up to 60 minutes, CPU-only, no Drive or API key required except
where noted:

| Option | Title | Focus |
|---|---|---|
| A | Toy 2D Diffusion | Rebuild the full pipeline (forward + reverse + model) on 2D points |
| B | Conditional Constellation Generator | Combines all 3 parts: forward, reverse, and classifier-free guidance on 2D shapes |
| C | Flower Prompt Gallery and Curation | Reuses the pretrained Part 3 flowers model, focused on prompt design and analysis (needs the Part 3 Drive checkpoint) |

Each option ships as a notebook (working example + `# TODO` sections +
collapsed solutions) and a companion `.md` with full instructions. See
`homework/` for all six files.

---

## Speaker Notes (Presenter Only)

`speaker_notes/` contains per-part timing, talking points, expected
output values, and common student sticking points for Practice Time. These
are for internal use and are excluded from the public repository via
`.gitignore`, same as the pretrained `.pth` checkpoints.

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

### Option C: Run locally

1. Download the `.ipynb` file
2. Open it in Jupyter Lab, Jupyter Notebook, or VS Code
3. Run the first cell, it installs anything missing automatically (see
   Requirements above)

---

## License

Workshop materials adapted from NVIDIA DLI course
"Generative AI with Diffusion Models".
Original materials Copyright NVIDIA Corporation.

Adaptations for Google Colab by Domen Verber and Jani Dugonik,
FERI, University of Maribor.
