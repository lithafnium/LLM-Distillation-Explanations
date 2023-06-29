# Towards Distilled Language Model Interpretability

Code for distilation for LLMs and building post-hoc explanations. 

We aimed to compare several post-hoc explanations methods as they are applied to BERT and distilled
versions of BERT across a variety of tasks (GLUE benchmark), to see if post-hoc explanations
are preserved during distillation. To be able to use BERT and its distilled variants on the GLUE tasks,
we first finetuned BERT and its distilled variants on different GLUE metrics. We were then able to
run a number of post-hoc explanations and then compare the token-level or word-level attributions
between a distilled BERT model and BERT. We also perform sentence perturbations to evaluate robustness, comparing evaluation metrics to the original
base model for each GLUE task. After that, we design our own distillation method, attention weight
alignment, that optimizes for preserving post-hoc explanations as a method of aligning a student
model with the teacher mode. 

- `build_explanation.py`: builds LIME, SHAP, and integrated gradients explanations for BERT-based models
- `model_robustness.py`: insert perturbations into GLUE datasets to evaluate model robustness
- `distillation/`: distilation module forked from Huggingface, where we align the attention weights of the student to match that of the teacher 
