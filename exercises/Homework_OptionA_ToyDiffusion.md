# Homework Option A: Build a Toy 2D Diffusion Model

**Time:** up to 60 minutes
**Requires:** nothing extra. Runs on CPU in under a minute. No GPU, no Drive, no API key.

---

## Goal

In the workshop you built a diffusion model for 16x16 FashionMNIST images.
In this assignment you will build the exact same kind of model, but for
simple 2D points instead of images. This removes all the complexity of
convolutions and U-Nets, so you can focus purely on the diffusion math.

You will train a tiny neural network to generate points that form the
shape of two crescent moons, starting from pure random noise.

---

## Step 1: Run the setup cells (5 min)

Open `Homework_OptionA_ToyDiffusion.ipynb` and run the first few cells.
They generate the training data (a two-moons point cloud) and set up
the same noise schedule you used in Part 1 of the workshop.

---

## Step 2: Complete the FILL_IN sections (20 min)

There are two `FILL_IN` sections in this notebook, both formulas you
already used in the workshop:

1. The closed-form forward diffusion formula (same as Part 1, Exercise 2)
2. The reverse diffusion formula (same as Part 2, Exercise 1)

If you get stuck, click the hidden solution below each one.

---

## Step 3: Train and generate (10 min)

Run the training cell. It should take less than a minute on CPU.
Then run the sampling cell to generate new points from noise.
You should see a scatter plot comparing the real two-moons shape
to the points your model generated.

---

## Step 4: Try a different shape (15 min)

Change the dataset to a different shape. Two easy options:

- Replace `make_moons` with `make_circles` (also from `sklearn.datasets`)
- Write your own point generator, for example points along a heart shape
  or a spiral (a short example is provided in the notebook as a comment)

Retrain the model on your new shape and generate samples again.

---

## Step 5: Write a short report (10 min)

At the end of your notebook, answer:

1. Did your model learn the two-moons shape well? What did the generated points look like?
2. What shape did you try in Step 4? Did the model learn it as well as the moons?
3. What happens if you reduce the number of training steps a lot (for example to 200)?
   Try it and describe what you see.

---

## Self-Check

- [ ] Both FILL_IN sections are completed and the training cell runs without error
- [ ] You generated samples and compared them visually to the real data
- [ ] You tried a second shape
- [ ] You wrote a short report
