# Llama 7B Finetune

## Content

This script finetunes a Llama7B model using either a local dataset or a dataset from the Hugging Face library. The model is quantized to 4 bits and is trained with specific configurations, including LoRA and training parameters.

## Command-Line Arguments

- `--HF`: A boolean value (`True`/`False`). Use `True` to specify that the dataset should be loaded from Hugging Face and `False` for a local dataset. Default is `True`.
  
- `--dataset-dir`: A string that indicates either the name of the Hugging Face dataset to be used or the directory of the local dataset. No default value.

- `--save-steps`: An integer that specifies the interval of steps to save model checkpoints. Default is `25`.

- `--save-total-limit`: An integer that specifies the maximum number of checkpoints to keep. Default is `2`.

- `--learning-rate`: A float that specifies the learning rate for training. Default is `2e-4`.

- `--num-train-epochs`: An integer that specifies the number of training epochs. Default is `1`.

## Usage

To run the script using a Hugging Face dataset with default parameters:

```bash
python finetune_llama.py --HF True --dataset-dir "mlabonne/guanaco-llama2-1k"
