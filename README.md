# Transfer-Learning-with-Joint-Fine-Tuning-for-Multimodal-Sentiment-Analysis

This is the code for the Paper "Guilherme L. Toledo, Ricardo M. Marcacini: Transfer Learning with Joint Fine-Tuning for Multimodal Sentiment Analysis (LXAI Research Workshop at ICML 2022)".

## Data preparation
### MVSA-1.2k
We used a subset from [MVSA](https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/) dataset.
Our pipeline to pre-process the dataset followed these steps:
- We assign to each sample the most common label among the annotators.
- Randomly `(seed=42)` balanced the dataset, ending up with 1211 samples for each label.
### HatefulMemes
We used the [HatefulMemes](https://ai.facebook.com/tools/hatefulmemes/) dataset. As for preprocessing, we merged `train.jsonl` and `dev_seen.jsonl` for the experiments.

## Logger
As a visualizer for each fold result, we used wandb as our logger.
If you ry to reproduce our code, make sure you are logged in your wandb account and don't forget to use the project as a parameter :)

## Testing
There are 2 .ckpt files with our best models from each 10-Fold experiment (MVSA and HatefulMemes).

Feel free to load the model with the fine-tuned parameters for inference :D
