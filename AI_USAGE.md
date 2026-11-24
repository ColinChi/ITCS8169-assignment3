# Use of AI tools

**Tool.** Claude Code (Anthropic) was used as a pair programmer throughout this assignment.

**What the AI did.**
- Read the assignment and proposed the plan (`PLAN.md`): frozen vision-language encoders, zero-shot → one-shot head →
  test-time adaptation → ensembles, with a one-shot protocol of ten random draws.
- Wrote the code in this repository: the one-shot split and data-access layer, feature caching for six open_clip
  backbones, all classifiers in `src/methods.py`, the SDXL support-set generator, the final training and evaluation
  scripts, the result tables, and the report generator.
- Ran the experiments on a 4×H200 node and summarized the results.
- Drafted `README.md` and the two-page report; every number is read from `results/` by the scripts.

**Decisions made by the student.**
- Use SigLIP as the main foundation model, and the existing `cv` conda environment.
- The one-shot rule: the other 2,384 training images are not used at all, not even without labels.
- The 16 images are drawn at random with fixed seeds, never chosen by hand or by results. Draw 0 is the reported
  model, and draws 1–9 measure the spread.
- No validation set: use the published default hyperparameters of each method.
- Report inductive and transductive results separately.
- After the first round of results, the student approved the SDXL support-set study, then packaging and the report,
  and then the encoder fine-tuning study (CLIP-LoRA).
- The report follows the format of the student's Assignment 2 report, and the variants that improve accuracy on
  the test set are reported in it.

**Final model.** The submitted model is the frozen-encoder pipeline that was fixed in advance (SigLIP 2 SO400M, CLAP
head, draw 0). Neither the SDXL variant nor the LoRA variant replaced it. LoRA has a higher 10-draw transductive mean,
but neither variant is better on draw 0, and switching on the basis of test results would amount to model selection
on the test set.

**Corrections and checks.**
- **Test-informed additions, disclosed.** The balanced (Sinkhorn) assignment, label propagation and balanced
  self-training were added after TransCLIP had been seen to fail on the test set. They use published defaults
  and were not tuned, but the decision to try them, and the choice of the final rows, is test-informed. The README
  and the report state this.
- **Prompt sets and anchor, test-informed.** The check of the report showed that (i) the zero-shot classifier
  with the balanced assignment reaches 91.00% without any training image, a row the shortened report had dropped,
  so an earlier draft credited the single image with the whole gain; (ii) CLAP's class-adaptive weight performs
  like a constant one, only the anchor itself matters; (iii) the plain prompt “a photo of a {}.” is a better prior
  on the test set than the prompts with synonyms that were fixed in advance (93.00% instead of 92.00% for CLAP +
  balanced assignment). All three prompt sets were defined before any test result, but preferring one of them now
  would be model selection on the test set, and the 16 shots themselves favour the submitted prompts in 8 of 10
  draws. The submitted model is unchanged; the variants are reported as test-informed upper estimates.
- **DFN temperature.** The open_clip DFN-5B checkpoint reports a logit scale of 14.3 (the untrained 1/0.07). This was
  noticed in the extraction log and replaced by CLIP's 100 (`LOGIT_SCALE_OVERRIDE`). Zero-shot accuracy is unaffected
  (argmax); probability-based methods would otherwise have been miscalibrated for that backbone.
- **Solver check.** The CLAP L-BFGS solution was compared with a float64 solve run to a gradient norm of 1e-7.
  The two differ by 2e-5, so "solved to convergence" holds.
- **End-to-end vs. cached features.** `evaluate.py` recomputes features from the raw images. It matches the cached
  zero-shot and inductive accuracies exactly. The transductive accuracy differs by one image (92.25% vs. 92.00%),
  because bf16 features change in the last bits when the views are batched differently. Repeated `evaluate.py` runs
  on different GPUs give identical accuracies. The report quotes the end-to-end numbers and explains the difference.
- **Checkpoint loading.** The first checkpoint stored a `TorchVersion` object, which PyTorch's safe `torch.load`
  refuses. It is now stored as a plain string.
- **CLIP-LoRA deviation.** CLIP-LoRA adapts both encoders. Here only the vision encoder is adapted, and the
  class weights stay the frozen prompt-ensemble text classifier. All other settings are the paper's defaults.
  The runs used GPUs 1–2, because GPUs 0 and 3 were taken by other jobs on the shared node.
- **Oracle analysis.** `scripts/oracle_probe.py` uses test labels and is reported as analysis only. It was never
  used to choose a method.
