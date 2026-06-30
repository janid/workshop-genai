# Homework Option B: Build a Diffusion Generator for a Topic of Your Choice

**Time:** up to 60 minutes
**Requires:** nothing extra. Runs on CPU in under a minute. No GPU, no Drive, no API key.

---

## Goal

In Part 1 and Part 2 you built a diffusion model for FashionMNIST images.
In this assignment you will build the same kind of model again, but for a
topic of your own choice. The only requirement: your data must be small
numeric vectors (between 2 and 8 numbers per example).

---

## Step 1: Choose your topic

Pick one of these, or propose your own:

- **Colors**: RGB values, visualized as color swatches
- **Simple shapes**: 2D points forming a shape, visualized as a scatter plot
- **Melodies**: short sequences of note pitches (for example 4 numbers between 60 and 72,
  like MIDI note numbers), visualized as a bar chart
- **Daily patterns**: a short numeric sequence representing something like
  hourly temperature or hourly activity level, visualized as a line plot
- **Your own idea**: any small numeric vector with a natural visualization

Your data should have at least 200 examples for the model to learn the shape
of the distribution reasonably well.

---

## Step 2: Build your dataset (15 min)

Open `Homework_OptionB_ChooseYourDiffusion.ipynb`.
Write code that creates a tensor `X` of shape `[n_samples, dim]`,
where `dim` is the number of numbers per example (2 for points, 3 for colors,
4 for a 4-note melody, etc.). A few example snippets are given as comments
to help you get started.

---

## Step 3: Complete the FILL_IN sections (15 min)

There are two `FILL_IN` sections, the same two formulas you used in the
workshop (Part 1, Exercise 2, and Part 2, Exercise 1). They work the same
way regardless of how many numbers are in each data point.

---

## Step 4: Train, generate, and visualize (20 min)

Run the training cell. Then write a small visualization function appropriate
for your topic: a scatter plot for points, color swatches for RGB values,
a bar or line plot for sequences. A few starter snippets are provided.

---

## Step 5: Write a short report (10 min)

At the end of your notebook, answer:

1. What topic did you choose, and why?
2. Did the generated samples look similar in style to your training data?
   Give one specific example of something that worked well and one that did not.
3. If you had more time, what would you change to improve the results
   (more training steps, more training data, a bigger model, something else)?

---

## Self-Check

- [ ] Your dataset `X` has at least 200 examples and a sensible shape
- [ ] Both FILL_IN sections are completed and the training cell runs without error
- [ ] You wrote a visualization function appropriate for your topic
- [ ] You wrote a short report
