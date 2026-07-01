# Homework Option C: Flower Prompt Gallery and Curation

**Time:** up to 60 minutes
**Requires:** Google Drive with the pretrained flowers model from Part 3
(`flowers_clip.pth`), and the `UNetCFG` class definition from your Part 3
notebook. No new training, no new architecture.

## Overview

This homework is lighter on code than Options A and B, and heavier on
creative thinking and analysis. You reuse the pretrained CLIP-conditioned
flowers model from Part 3 as-is. The real work is designing prompts that
push the model somewhere interesting, and writing up what you find.

## Steps

1. **Setup.** Loads CLIP and the pretrained flowers model, exactly like the
   Part 3 demo. You need to paste in the `UNetCFG` class definition from
   your Part 3 notebook before this cell will run (a reminder comment marks
   where). Already working otherwise.

2. **Design Your Prompts (your turn).** Write 5 to 6 of your own creative
   text prompts. Be more adventurous than the workshop demo: unusual
   colors, moods, or combinations. A collapsed list of example prompts is
   provided if you are stuck, but your own ideas are just as valid, even if
   some of them fail.

3. **Generate the Gallery.** Generates one image per prompt at a fixed
   guidance weight. Already working, runs once your prompts are filled in.

4. **Compare Guidance Weights (your turn).** Pick 2 of your prompts, the
   ones you are most curious about, and compare them at a weak (`w=0.5`)
   and a strong (`w=2.0`) guidance weight.

## What to Submit

The completed notebook, with your 5-6 prompts from Step 2, the gallery from
Step 3, the comparison from Step 4, and the Your Report (Curator's Notes)
cell at the end filled in.

The report asks you to identify your best and worst result, explain how
guidance weight changed your two comparison prompts, and suggest one
concrete wording change you would try next. This is the main deliverable
for this option, treat it as a short curator's writeup, not just a few
words.
