# Homework Option A: Build a Toy 2D Diffusion Model

**Time:** up to 60 minutes
**Requires:** nothing extra. Runs on CPU in under a minute. No GPU, no Drive, no API key.

## Overview

This homework rebuilds the full diffusion pipeline from Parts 1 and 2, but on
2D points instead of images. A tiny MLP plays the role of the U-Net. Everything
runs in seconds on CPU, so you can experiment freely.

The dataset, the forward process, the model, and the training loop are all
already working. Your job is Step 5: the reverse diffusion step. After that,
Step 7 asks you to try a different shape of your own choosing.

## Steps

1. **The Dataset.** Two crescent-shaped clusters of 2D points (`make_moons`).
   Already working, just run it.

2. **Forward Diffusion.** Same noise schedule and `q` function as Part 1,
   adapted to 2D points. Already working. The visualization shows the moons
   dissolving into noise as `t` increases.

3. **The Model.** A tiny MLP (a few thousand parameters) takes a noisy point
   and a timestep, and predicts the noise. Already working.

4. **Training.** 8000 steps, about 15-20 seconds on CPU. Already working.

5. **Reverse Diffusion (your turn).** Implement `reverse_step`, the same
   formula as Part 2's `reverse_q`, just for 2D points. The function is
   currently a placeholder that does nothing useful. A collapsed solution is
   provided if you get stuck.

6. **Generate and Visualize.** Samples from pure noise using your
   `reverse_step`, and shows the points forming back into the moon shapes.
   Already working, once Step 5 is fixed.

7. **Try It Yourself.** Swap the dataset for a different 2D shape
   (`make_circles`, `make_blobs`, or your own point generator) and retrain.

**Bonus (optional):** `make_moons` already gives you 2 class labels. Try
adding simple class conditioning, the same idea as Part 3's classifier-free
guidance, and generate each moon separately.

## What to Submit

The completed notebook, with the Step 5 fix applied, the Step 7 experiment
run, and the Your Report cell at the end filled in.
