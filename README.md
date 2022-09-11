# Knowledge-Graph-Reasoning-with-Self-supervised-Reinforcement-Learning
Official code for the following paper:

PLACEHOLDER

![summary image of system architecture](docs/summary_image.png)

## Setup
### Dependencies
#### Use Docker
Build the docker image
```
docker build -< Dockerfile -t multi_hop_kg:v1.0
```

Spin up a docker container and run experiments inside it.
```
nvidia-docker run -v `pwd`:/workspace/MultiHopKG -it multi_hop_kg:v1.0
```
*The rest of the readme assumes that one works interactively inside a container. If you prefer to run experiments outside a container, please change the commands accordingly.*

#### Manually set up 
Alternatively, you can install Pytorch (>=1.12.0+cu116) manually and use the Makefile to set up the rest of the dependencies. 
```
make setup
```

### Prepare and run experiments
#### Set up an experiment
Run the following command to set up an experiment
```
./experiment_setup.sh configs/<rl Base Model>/<dataset>.sh <gpu-ID>
```
The following rl base models are implemented: `MINERVA`, and `MultiHopKG-ConvE`.
The following datasets are available: `FB15K-237`, `FB60K-NYT10` (only available for `MINERVA` base model), `NELL-995`, and `WN18RR`.
`<gpu-ID>` is a non-negative integer number representing the GPU index.

* Note: Setup will take a while for any experiment using `MultiHopKG-ConvE` as the RL base model as a standalone ConvE model must be trained to be used for reward shaping.

#### Run an experiment
Run the following command to train a model
```
./experiment_run.sh configs/<rl Base Model>/<dataset>.sh <gpu-ID> <experiment_name>
```
experiment_name will be used to name the experiment's output folder, which will be located in the out/ directory. The structure of the output directory is as follows
```
<experiment_name>_<current_time>
    └── <rl Base Model>
            └── <dataset>
                    └── <experiment_name>_<current_time>
                            ├── config.txt
                            ├── log.txt
                            ├── model
                            │       └── <saved model files>
                            ├── checkpoint_sl_<ckpt_#>
                            │       ├── model_weights
                            |       |       └── <saved model files>
                            │       └── scores.txt
                            └── checkpoint_sl_<ckpt_#+1>
                            etc.
```

#### Process Data
To generate heatmaps and 