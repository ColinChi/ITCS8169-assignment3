# ITCS 6169/8169 Assignment 3 — One-Shot Scene Recognition with Frozen Foundation Models

16 scene classes, **one labeled training image per class** (16 images in total). The classifier is a frozen
SigLIP 2 vision-language model plus an 18,432-parameter linear head fitted to the 16 images in 0.1 s.

## Final result

One-shot draw 0 (fixed in advance), SigLIP 2 SO400M/16 @384, reproduced from the raw images by `evaluate.py`:

| Setting | What is used | **Test accuracy (400 images)** |
|---|---|---:|
| Zero-shot | class names only, no training image | 88.75% (355/400) |
| **One-shot, inductive** | 16 training images; each test image classified on its own | **92.00% (368/400)** |
| **One-shot, transductive** | 16 training images + the 400 unlabeled test images, assigned jointly | **92.25% (369/400)** |

Over 10 random one-shot draws, the same pipelines reach 91.0 ± 1.0% (inductive) and 92.5 ± 1.4% (transductive)
on the cached features (`results/summary_main.csv`). For comparison, our Assignment 2 system trained from scratch on
all 2,400 labeled images reached 90.75%.

```bash
python -m pip install -r requirements.txt
python evaluate.py checkpoints/siglip2_so400m_seed0.pt --per-class   # downloads the backbone from Hugging Face
```

`checkpoints/siglip2_so400m_seed0.pt` (2.5 MiB) holds the trained head, the text classifier, the shot features and
the one-shot split. The frozen backbone (`ViT-SO400M-16-SigLIP2-384`, `webli`) is downloaded by open_clip.

## The one-shot rule

- `splits/oneshot_seed<k>.json` lists the single image drawn per class for draw k (uniformly at random,
  `random.Random(k)`), with its SHA-256. Draw 0 is the reported model; draws 1–9 measure how much the result depends
  on which image happens to be drawn. Images were never chosen by looking at them or at results.
- `src/data.py` is the only code that touches `data/train`. It lists file names to draw the split and otherwise
  opens training images only through a split file (`load_shots` asserts one image per class and checks the hash).
  The other 2,384 training images are never decoded, encoded or used in any way, not even without labels.
- Every result of one draw uses exactly that draw's 16 images. Nothing is pooled across draws.
- There is no validation set: any held-out labeled image would be a second sample per class. Hyperparameters are the
  published defaults of each method (listed below), and **every number in this repository is a test accuracy**.
  Methods were compared on the test set, so the tables are a study of the methods, not a held-out model
  selection. Main backbone, draw 0, and the inductive/transductive split were fixed before any result.
- Test images are used without labels in two ways, reported separately: per-image test-time augmentation
  (inductive), and transductive inference over the 400 test embeddings. The transductive "balanced assignment"
  assumes roughly equal class frequencies.

## Method

1. **Frozen encoders.** Six contrastive vision-language models through open_clip (table below). All weights stay
   frozen; images go through the model's own preprocessing. Fifteen classes are grayscale, *Flower* is RGB.
2. **Text classifier (zero-shot).** For each class: 10 templates (including "a black and white photo of a {}")
   × the folder name and common synonyms (`src/prompts.py`), embeddings averaged (CLIP prompt ensembling).
3. **One-shot head: CLAP** (Silva-Rodríguez et al., CVPR 2024). A linear head W initialised with the text
   classifier and trained with cross-entropy on 66 augmented views of the 16 shots (original, flip, 64 random
   resized crops covering 50–100% of the area), with the class-adaptive penalty Σ_c λ_c‖W_c − t_c‖². λ_c is the
   zero-shot probability of the shot's true class, so the head may move away from the text only for classes the
   text describes poorly. The objective is convex; it is solved to convergence with L-BFGS, so there is no learning
   rate or epoch count to choose.
4. **Test time.** Inductive: the original view (view averaging over 34 views and "Zero" voting over 64 views were
   evaluated and did not help). Transductive: CLAP probabilities of the view-averaged test embeddings, then a
   Sinkhorn assignment of the 400 test images with equal class mass (as in PT-MAP).
5. **Also evaluated** (defaults in brackets): Tip-Adapter (α = 1, β = 5.5), nearest image prototype, TransCLIP
   (λ = 1, 3-NN, 10 iterations), label propagation (α = 0.99, γ = 3, k = 5/10/20), balanced curriculum
   self-training (20% steps), and synthetic support sets from SDXL: 32 text-to-image images per class
   (SuS-X style, class names only) and 16 IP-Adapter variations of each shot (30 steps, guidance 5, scale 0.6).
6. **Encoder fine-tuning: CLIP-LoRA** (Zanella & Ben Ayed, CVPRW 2024) with its defaults: rank-2 LoRA on q, k, v
   of all 27 vision blocks (α = 1, dropout 0.25, 373k parameters), AdamW (lr 2e-4, weight decay 0.01, cosine),
   500 iterations of batch 32, random resized crops (8–100% area) + flips (`finetune_lora.py`). Unlike CLIP-LoRA,
   only the vision encoder is adapted, and the class weights stay the frozen prompt-ensemble text classifier.

## Results

Test accuracy (%) of the main backbone. Seed 0 = the reported draw; mean ± std and range over 10 draws.

| Method | Seed 0 | 10 draws: mean ± std | min – max |
|---|---:|---:|---:|
| *Zero-shot (0 images)* | | | |
| single prompt, folder names | 89.75 | – | – |
| prompt ensemble, folder names | 89.25 | – | – |
| prompt ensemble, synonyms | 88.75 | – | – |
| &nbsp;&nbsp;+ colour removed from all test images | 88.75 | – | – |
| &nbsp;&nbsp;+ view averaging (34 views) | 89.50 | – | – |
| &nbsp;&nbsp;+ Zero TTA (64 views, top-10% vote) | 89.25 | – | – |
| SDXL name-only support set + CLAP | 86.50 | – | – |
| *One-shot, inductive* | | | |
| nearest image prototype | 78.00 | 76.15 ± 3.59 | 70.50 – 81.25 |
| Tip-Adapter (α 1, β 5.5) | 89.75 | 89.62 ± 0.40 | 89.00 – 90.25 |
| **CLAP head (final inductive)** | **92.00** | 90.97 ± 1.02 | 89.25 – 92.00 |
| &nbsp;&nbsp;+ view averaging | 89.50 | 90.75 ± 1.06 | 89.00 – 92.50 |
| &nbsp;&nbsp;+ Zero TTA | 87.75 | 89.75 ± 0.91 | 87.75 – 90.75 |
| CLAP + SDXL text-to-image support | 90.00 | 89.18 ± 1.78 | 85.00 – 90.75 |
| CLAP + SDXL shot variations | 90.25 | 91.10 ± 0.76 | 89.75 – 92.00 |
| CLAP + both synthetic sets | 90.00 | 89.90 ± 1.62 | 85.75 – 91.25 |
| CLIP-LoRA fine-tuned encoder + text classifier | 89.00 | 90.03 ± 1.62 | 88.25 – 93.00 |
| CLIP-LoRA fine-tuned encoder + CLAP head | 91.25 | 90.20 ± 1.57 | 88.25 – 92.75 |
| *Transductive (400 unlabeled test images)* | | | |
| zero-shot + balanced assignment | 91.00 | – | – |
| zero-shot + TransCLIP | 87.75 | – | – |
| **CLAP + balanced assignment (final transductive)** | **92.00** | 92.53 ± 1.42 | 90.00 – 95.25 |
| CLIP-LoRA encoder + balanced assignment | 91.25 | 93.30 ± 1.12 | 91.25 – 95.25 |
| CLIP-LoRA encoder + CLAP + balanced assignment | 91.00 | 93.30 ± 1.26 | 91.00 – 95.75 |
| CLAP + TransCLIP | 88.00 | 92.07 ± 2.44 | 88.00 – 95.50 |
| CLAP + label propagation (k = 10) | 89.00 | 86.88 ± 2.26 | 83.75 – 90.00 |
| CLAP + balanced self-training | 89.00 | 93.57 ± 3.20 | 87.50 – 96.75 |
| CLAP + shot variations + self-training | 86.75 | 93.72 ± 3.43 | 86.75 – 97.00 |

The end-to-end run (`evaluate.py`, from raw images) gives 92.25% for the final transductive row instead of 92.00%:
one borderline test image changes its Sinkhorn assignment because the bf16 features differ in the last bits when
the views are batched differently. Accuracies of `evaluate.py` are identical across repeated runs and GPUs.

Backbones (seed 0 / mean ± std over 10 draws; `results/summary_backbones.csv`). Oracle: 5-fold logistic regression
on the test embeddings with all test labels (analysis only, `scripts/oracle_probe.py`).

| Backbone | Zero-shot | CLAP (inductive) | CLAP + balanced (transductive) | Oracle (all labels) |
|---|---:|---:|---:|---:|
| SigLIP 2 SO400M/16 @384 (main) | 88.75 | 92.00 / 91.0 ± 1.0 | 92.00 / 92.5 ± 1.4 | 97.50 |
| SigLIP 2 g-opt/16 @384 | 88.00 | 90.00 / 90.1 ± 1.1 | 90.25 / 92.4 ± 1.4 | 98.50 |
| SigLIP SO400M/14 @384 | 86.75 | 88.25 / 89.8 ± 1.0 | 87.75 / 90.2 ± 1.5 | 98.00 |
| OpenAI CLIP L/14 @336 | 88.00 | 89.50 / 91.8 ± 1.6 | 90.25 / 93.0 ± 2.2 | 97.00 |
| DFN-5B CLIP H/14 @378 | 89.25 | 87.25 / 89.7 ± 1.8 | 89.00 / 91.0 ± 1.5 | 98.25 |
| PE-Core bigG/14 @448 | 86.25 | 87.25 / 89.1 ± 1.8 | 89.50 / 91.8 ± 1.6 | 97.75 |

Prompt sets behind the text classifier (`run_ablations.py`, seed 0 / mean ± std over 10 draws). All three were
defined before any test result; the first one was fixed in advance and is used by the submitted model.
**Choosing among them by these numbers is model selection on the test set**, so the other two columns are
test-informed upper estimates. The CLIP-LoRA encoder was fine-tuned with the first prompt set in all columns.

| Method | names + synonyms, 10 templates (submitted) | names, 10 templates | names, “a photo of a {}.” |
|---|---:|---:|---:|
| zero-shot | 88.75 | 89.25 | 89.75 |
| zero-shot + balanced assignment | 91.00 | 91.00 | 92.25 |
| Tip-Adapter | 89.75 / 89.6 ± 0.4 | 89.50 / 89.6 ± 0.2 | 89.75 / 89.7 ± 0.2 |
| CLAP head | 92.00 / 91.0 ± 1.0 | 92.00 / 90.9 ± 0.8 | 92.00 / 91.1 ± 1.1 |
| CLAP + balanced assignment | 92.00 / 92.5 ± 1.4 | 93.00 / 93.3 ± 1.6 | 93.00 / 93.5 ± 1.4 |
| CLAP + TransCLIP | 88.00 / 92.1 ± 2.4 | 86.00 / 91.3 ± 3.6 | 88.25 / 91.3 ± 3.5 |
| CLAP + balanced self-training | 89.00 / 93.6 ± 3.2 | 89.75 / 94.7 ± 2.7 | 95.50 / 94.3 ± 3.1 |
| CLIP-LoRA encoder + CLAP head | 91.25 / 90.2 ± 1.6 | 90.50 / 90.0 ± 1.2 | 90.00 / 90.2 ± 1.5 |
| CLIP-LoRA encoder + CLAP + balanced assignment | 91.00 / 93.3 ± 1.3 | 92.00 / 93.4 ± 1.2 | 92.75 / 93.9 ± 1.3 |

The only labelled evidence about the prompts that the one-shot rule allows is the 16 shots themselves: the
cross-entropy of the zero-shot classifier on the shot views (`results/prompt_fit_siglip2_so400m.csv`) is lowest for
the submitted prompt set in 8 of 10 draws. The submitted model therefore keeps it.

Anchor of the CLAP head (submitted prompts, seed 0 / mean ± std over 10 draws):

| Weight of the anchor to the text | inductive | + balanced assignment |
|---|---:|---:|
| class-adaptive λ (CLAP, submitted) | 92.00 / 91.0 ± 1.0 | 92.00 / 92.5 ± 1.4 |
| constant λ = 1 | 91.50 / 90.8 ± 1.3 | 92.25 / 92.5 ± 1.1 |
| no anchor (λ = 0) | 90.75 / 89.8 ± 1.6 | 90.75 / 91.7 ± 1.9 |

### Observations

- **The text prior and the class balance do most of the work.** Zero-shot SigLIP 2 reaches 88.75%, and 91.00% with
  the balanced assignment, without any training image. A single image as a prototype gets only 76%, while text
  plus the image (CLAP) gets 91.0 ± 1.0% (better than zero-shot in all ten draws) and 92.5 ± 1.4% with the balanced
  assignment. The head moves only 0.02 (L2) away from the text weights.
- **The anchor matters, its class-adaptive weight does not.** Without the anchor the head loses 1.2 points; a
  constant weight gives the same accuracy as CLAP's λ_c.
- **Plain class names are the better prior on this test set.** With “a photo of a {}.” the balanced zero-shot
  classifier already reaches 92.25%, CLAP + balanced assignment 93.00% (93.5 ± 1.4%), and with the LoRA encoder
  93.9 ± 1.3% (worst draw 92.00%). This was found on the test set, and the shots themselves favour the submitted
  prompts, so it is reported as a test-informed variant only.
- **The gap is label information, not representation.** All six encoders separate the classes at 97–98.5% when
  all labels are available, and larger encoders (SigLIP 2 giant, PE bigG) are not better one-shot learners.
- **Errors sit in classes whose definition is dataset-specific:** InsideCity vs. Street/TallBuilding and
  OpenCountry vs. Mountain/Forest. Zero-shot gets 4/25 InsideCity images right, CLAP 12/25.
- **Which image is drawn matters.** In draw 0 the InsideCity shot (a façade with a car on the road) looks like a
  street, and the Street shot looks like a dense downtown scene; CLAP's λ for InsideCity is 0.25 versus ≥ 0.79 for
  every other class. Transductive methods amplify the shot: self-training ranges from 87.5% to 96.75% across draws.
  The balanced assignment is the only transductive step that helps on average without large failures.
- **Fine-tuning the encoder does not pay off inductively.** CLIP-LoRA fits the 16 shots perfectly (training loss
  0.002–0.011) and lowers inductive accuracy (90.2 ± 1.6% with a CLAP head vs. 91.0 ± 1.0% frozen). Its features do
  suit the balanced assignment better (93.3 ± 1.1% vs. 92.5 ± 1.4%, worst draw 91.25% vs. 90.0%), but not on
  draw 0 (91.25% vs. 92.00%), and a draw costs 4 H200-minutes instead of 0.1 s. The final model keeps the
  frozen encoder fixed in advance.
- **Synthetic images add no dataset-specific information.** SDXL images generated from class names are a weaker
  prior than the text embedding itself (86.5% vs. 88.75%); shot variations slightly reduce the variance of CLAP but
  do not raise its mean.
- **Generic test-time augmentation barely matters for scene images.** View averaging and Zero voting add at most
  0.75 points to zero-shot and lower CLAP's mean by 0.2 and 1.2 points; crops drop the context that defines a scene.
- **No colour shortcut.** Removing colour from all test images leaves zero-shot accuracy unchanged.

## Reproducing everything

`scripts/run_all.sh` runs the full study on 4 GPUs (about 45 minutes on H200s): one-shot splits → frozen features for
six backbones → SDXL support sets → all methods on 10 draws → CLIP-LoRA fine-tuning per draw → prompt and anchor
ablations → tables → final model → end-to-end evaluation.
Place the course data as `data/train/<class>/*.jpg` (2,400 images) and `data/test/<class>/*.jpg` (400 images).

| File | Purpose |
|---|---|
| `src/data.py` | One-shot draws, the only access path to training images, test list |
| `src/models.py` | open_clip model registry, view transforms |
| `src/prompts.py` | Class names, synonyms and templates |
| `src/methods.py` | Zero-shot, prototype, Tip-Adapter, CLAP, Zero TTA, Sinkhorn, TransCLIP, label propagation, self-training |
| `extract_features.py` | Frozen features of test views, shot views and prompts → `cache/<model>/` |
| `generate.py`, `extract_synth.py` | SDXL text-to-image and IP-Adapter support sets and their features |
| `run_methods.py` | Every method × 10 draws on cached features → `results/acc_<model>.csv`, `results/probs/` |
| `run_ablations.py` | Prompt sets and CLAP anchor on the main backbone → `results/acc_<model>_ablation.csv`, `results/prompt_fit_<model>.csv` |
| `ensemble.py` | Probability averaging across backbones (+0.5–1 point; not used in the final model) |
| `src/lora.py`, `finetune_lora.py`, `eval_lora.py` | CLIP-LoRA fine-tuning of the vision encoder per draw, features, evaluation |
| `train.py` | Final head from the 16 raw images → `checkpoints/<model>_seed<k>.pt` |
| `evaluate.py` | Test accuracy of a checkpoint from the raw test images; writes per-image predictions |
| `scripts/summarize.py`, `scripts/oracle_probe.py`, `scripts/build_report.py` | Tables, oracle analysis, report (PDF and Word) |

Software: Python 3.10, PyTorch 2.14.1 (CUDA 13.0), open_clip 4.2.2, diffusers 0.41.0 (`requirements.txt`).
The DFN-5B checkpoint in open_clip stores the untrained temperature (1/0.07); we use CLIP's 100 for it
(`LOGIT_SCALE_OVERRIDE` in `src/models.py`).
