# MVPFormer: A foundation model with multi-variate parallel attention to generate neuronal activity

MVPFormer is a foundation model trained and tested on almost 10,000 hours of iEEG recordings. It can do next-state prediction and, with the addition of classification heads, can also detect seizures.

If your GPU has compute capability > 8.0, i.e., it is Ampere or later, MVPFormer will automatically use the optimised Flash-MVPA; otherwise, it will run with the slower and more memory hungry PyTorch MVPA implementation. Doing inference without Flash-MVPA is supported, while training without Flash-MVPA is not recommended.

## Prepare the environment

To prepare the environment for running MVPFormer you need a mixture of pip and compilation from source. 

### Pip

The `requirements.txt` file is provided in the repository. Simply install all requirements with `pip install -r requirements.txt`.

### DeepSpeed

You have to compile [`DeepSpeed`](https://www.deepspeed.ai/tutorials/advanced-install/) manually to activate some necessary extensions. The procedure can vary based on your software and hardware stack, here we report our reference installation steps.

```bash
DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LAMB=1 pip install --no-cache-dir deepspeed --global-option="build_ext" --global-option="-j8"
```

## Inference with MVPFormer

We use `PyTorch Lightning` to distribute reproducible configuration files for our experiments. The example testing configuration file can be found in the `configs` folder. You can start testing with:
```bash
python main.py test --config configs/mvpformer_classification.yaml --model.init_args.base_model '<base_checkpoint_path>' --model.init_args.head_model '<head_checkpoint_path>' --data.init_args.folder '<dataset_path>' --data.init_args.test_patients ['<dataset_subject>']
```

## Testing MVPFormer

We use `PyTorch Lightning` to distribute reproducible configuration files for our experiments. The example testing configuration file can be found in the `configs` folder. You can start testing with:
```bash
python main.py fit --config configs/mvpformer_classification.yaml --model.init_args.base_model '<base_checkpoint_path>' --model.init_args.head_model '<head_checkpoint_path>' --data.init_args.folder '<dataset_path>' --data.init_args.train_patients ['<dataset_subject>']
```

The example parameters are equivalent to what we have used to train MVPFormer, except in the hardware setup such as the number of GPUs and the number of CPU workers.

## Checkpoints

The checkpoints can be downloaded from [this](https://ibm.box.com/v/mvpformer-checkpoints) location. The checkpoint with `base` are the base models with only generative pre-training. The `swec` models are the classification heads.

## Disclaimer

This software may only be used for research. For other applications any liability is denied. In particular, the software must not be used for diagnostic purposes.

## License

If you would like to see the detailed LICENSE click [here](LICENSE).

```text
#
# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: Apache-2.0
#
```