# Homework Option B: Conditional Constellation Generator

**Time:** up to 60 minutes
**Requires:** nothing extra. Runs on CPU in under a minute. No GPU, no Drive, no API key.

## Overview

This homework combines all three parts of the workshop in one place: the
forward process from Part 1, the reverse process from Part 2, and
classifier-free guidance from Part 3, all on tiny 2D point clouds instead of
images. A small MLP conditioned on a class label plays the role of the
flowers UNetCFG.

The dataset, the forward process, the reverse step, the conditional model,
and the training loop are all already working. Your job is Step 5: the
classifier-free guidance combination at sampling time.

## Steps

1. **A Multi-Shape Dataset.** Three shape classes: circle, spiral, and cross.
   Already working, just run it.

2. **Forward Diffusion.** Same `q` function as always. Already working.

3. **Reverse Diffusion.** Same `reverse_step` formula as Part 2 and as
   Homework Option A. Already working.

4. **Conditional Model and Training.** The model also takes a one-hot class
   vector, the same way Part 3's UNetCFG does. Training uses context dropout
   (10 percent of the time the class is hidden), the same idea as Part 3.
   10000 steps, about 30-40 seconds on CPU. Already working.

5. **Conditional Sampling with CFG (your turn).** Implement the guidance
   combination inside `sample_conditional`, the same formula as Part 3:
   `e_t = (1 + w) * e_cond - w * e_uncond`. The function currently ignores
   `w` completely. A collapsed solution is provided if you get stuck.

6. **Generate and Visualize.** Generates each shape class separately and
   shows them side by side. Already working, once Step 5 is fixed.

7. **Try It Yourself.** Design a 4th shape class of your own (a starting
   idea for a star shape is included), add it to the dataset, retrain, and
   regenerate.

**Bonus (optional):** Try negative guidance, the same idea as Part 3's
bonus task: push away from a different shape class instead of the
unconditional prediction, and compare.

## What to Submit

The completed notebook, with the Step 5 fix applied, the Step 7 experiment
run (or the Bonus, if you tried that instead), and the Your Report cell at
the end filled in.
