# Transformer Groove Infilling

**Note:** This repo depends on libraries that are not yet public. A standalone zip version of this code containing all necessary libraries can be downloaded [here]

## Abstract

Infilling drums refers to complementing a drum pattern with additional drum events that are stylistically consistent with the loop. This task can be applied to computer-assisted composition; for instance, the composer can sketch some instrument parts of a drum beat and obtain the system's suggestions for new parts. In this thesis, we present the Transformer Groove Infilling, a Transformer Neural Network approach to the infilling task. Until now, the infilling of drum beats has been implemented using Recurrent Neural Network (RNN) architectures, in particular, sequence-to-sequence models that employ LSTM cells. However, in such architectures, as a consequence of sequential computation, proximity is emphasized when dealing with dependencies in the input sequence. Furthermore, those models receive the audio loops as symbolic input sequences. In contrast, the Transformer Groove Infilling model is based on the Transformer architecture, which relies entirely on self-attention mechanisms to represent the input sequences, which allows for faster training since parallelization is possible. In addition, we present a novel direct audio representation that enables the Transformer Groove Infilling to receive the input drum loops in the audio domain, avoiding their transcription and tokenization.

We train several instances of the Transformer Groove Infilling model to perform the following infilling subtasks: the infilling of closed hi-hats, the infilling of both kicks and snares, and the infilling of a drum loop without an instrument constraint. For comparison purposes, we also trained an equivalent model with a symbolic input representation in order to highlight the efficiency of the audio representation proposed in this thesis.

## clone this repo

This repo has submodules. You can clone it including them using:

```bash
git clone --recurse-submodules -j8 git@github.com:---/---.git
```

## environment

To run this project you need some external packages. We provide an `envrionment.yaml` you can use to build the environment with conda:

```bash
conda env create -f environment.yml
```

and activate it using

```bash
conda activate groove
```

However conda sometimes fails when building environments. You can create the environment from scratch:

```bash
conda create --name groove python=3.6
conda activate groove
```

and install all packages:

```bash
conda install -c conda-forge fluidsynth
# if you have a cuda gpu you can install a suitable pytorch distribution here https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio -c pytorch
pip install magenta==1.1.7
pip install visual_midi
pip install tables
pip install note_seq
pip install pyFluidSynth
pip install wandb
pip install bokeh
pip install pandas
pip install PySoundFile
pip install colorcet
pip install holoviews
pip install tqdm
pip install note_seq
pip install matplotlib
```

## Quickstart

The model hyperparameters can be passed to the training script through a yaml file or through the CLI. Yaml configuration file examples can be found in the `model/configs/` directory ([here](model/configs/)).

You can try an example by running:

```
$ cd model
$ python3 train.py --config=configs/InfillingClosedHH_training.yaml --wandb=False
```

If you want to keep track of your experiments with W\&B, you can set up an account and configure your API key following [this documentation](https://docs.wandb.ai/quickstart).

Currently, the following options are available for the CLI:

```
$ python model/train.py -h

  --paths PATHS         paths file
  --testing TESTING     testing mode
  --wandb WANDB         log to wandb
  --eval_train EVAL_TRAIN
                        evaluator train set
  --eval_test EVAL_TEST
                        evaluator test set
  --eval_validation EVAL_VALIDATION
                        evaluator validation set
  --only_final_eval ONLY_FINAL_EVAL
                        only final total evaluation
  --dump_eval DUMP_EVAL
                        dump evaluator file
  --load_model LOAD_MODEL
                        load model parameters
  --notes NOTES         wandb run notes
  --tags TAGS           wandb run tags
  --config CONFIG       yaml config file. if given, the rest of the arguments
                        are not taken into account
  --experiment EXPERIMENT
                        experiment id
  --encoder_only ENCODER_ONLY
                        transformer encoder only
  --optimizer_algorithm OPTIMIZER_ALGORITHM
                        optimizer_algorithm
  --d_model D_MODEL     model dimension
  --n_heads N_HEADS     number of heads for multihead attention
  --dropout DROPOUT     dropout factor
  --num_encoder_decoder_layers NUM_ENCODER_DECODER_LAYERS
                        number of encoder/decoder layers
  --hit_loss_penalty HIT_LOSS_PENALTY
                        non_hit loss multiplier (between 0 and 1)
  --batch_size BATCH_SIZE
                        batch size
  --dim_feedforward DIM_FEEDFORWARD
                        feed forward layer dimension
  --learning_rate LEARNING_RATE
                        learning rate
  --epochs EPOCHS       number of training epochs
```
