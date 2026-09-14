# ABC: Video Frame Interpolation with Bezier Control Points

ABC is a video frame interpolation (VFI) model that models the motion
between two frames with a **quadratic Bezier curve**. Instead of assuming
linear motion, a lightweight control estimator predicts two Bezier control
points (one per input frame), which are shared by a coarse-to-fine
pyramid-based motion estimator and frame synthesis network. Given two input
frames and a time step, ABC estimates bi-directional flow and synthesizes
the intermediate frame.

This repository contains the **interpolation** part of the project, i.e.,
the target frame always lies between the two input frames (t in (0, 1)).


## Python and Cuda environment
This code has been tested with PyTorch 1.13 and Cuda 11.7. It should also be
compatible with higher versions of PyTorch and Cuda. Run the following command
to initialize the environment:
```
conda create --name abc python=3.8
conda activate abc
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip3 install cupy_cuda11x
pip3 install -r requirements.txt
```

In particular, CuPy package is required for running the forward warping
operation (refer to
[softmax-splatting](https://github.com/sniklaus/softmax-splatting) for details).


## Play with demo
We place trained model weights in `checkpoints`, and provide a script to test
our frame interpolation model. Given two consecutive input frames, and the
desired time step, run the following command, then you will obtain estimated
bi-directional flow and interpolated frame in the `./demo/output` directory.
```
python3 -m demo.interp_imgs \
--frame0 demo/images/beanbags0.png \
--frame1 demo/images/beanbags1.png \
--time_period 0.5
```
Here the `time_period` (float number in 0~1) indicates the time step of the
intermediate frame you want to interpolate.


## Training on Vimeo90K

ABC is trained in **two stages** on Vimeo90K. Please download the
[Vimeo90K](http://toflow.csail.mit.edu/) dataset (both the septuplet and the
triplet splits are used).

Estimating the motion of the intermediate frame directly from two input
frames is ill-posed: the same inputs admit many motion patterns, which makes
training ambiguous and produces blurry results. ABC resolves this in the
first stage by **deriving the Bezier control points from the ground-truth
frame with RAFT**, so that the network only has to estimate the flow between
the two inputs. The second stage then attaches a Bezier Control point
estimation Module (BCM) and learns to predict the control points itself.

### Stage 1: train with GT/RAFT control points on Vimeo90K-septuplet

The first stage trains the feature extractor, flow estimation module, and
frame generation module on the **septuplet** split. Each sample randomly
picks three frames `idx0 < idx_gt < idx1` (the outer two are the inputs, the
middle one is the target, t in (0, 1)); the Bezier control points are
computed on the fly from the ground-truth frame with
[RAFT](https://github.com/princeton-vl/RAFT) (see
`Pipeline.compute_gt_control_points`), following Eqn. (7) of the paper.

Download the RAFT checkpoint first:
```
cd getflow/RAFT && ./download_models.sh && cd ../..
```
Then start the first-stage training (`--training_mode raft_gt_control`):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=10000 -m tools.train \
        --world_size=4 \
        --data_root /path/to/vimeo_septuplet \
        --train_log_root /path/to/train_log \
        --exp_name abc-base-stage1 \
        --batch_size 8 \
        --nr_data_worker 2 \
        --dataset_type septuplet \
        --training_mode raft_gt_control \
        --raft_checkpoint getflow/RAFT/models/raft-things.pth
```

### Stage 2: joint training with the control point estimator (BCM) on Vimeo90K-triplet

The second stage initializes the feature extractor, flow estimation module,
and frame generation module with the **stage-1 weights**, adds the Bezier
Control point estimation Module (BCM), and retrains everything jointly on
the **triplet** split (`--training_mode learned_control`). The BCM predicts
the control points as `B = 0.5 * F + delta_B`, where `F` is the estimated
bi-directional flow and `delta_B` is a residual estimated by a small
convolutional network from the encoder features and the flow.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=10000 -m tools.train \
        --world_size=4 \
        --data_root /path/to/vimeo_triplet \
        --train_log_root /path/to/train_log \
        --exp_name abc-base-stage2 \
        --batch_size 8 \
        --nr_data_worker 2 \
        --dataset_type triplet \
        --training_mode learned_control \
        --load_pretrain \
        --model_file /path/to/train_log/abc-base-stage1/trained-models/model.pkl
```

### Training settings

- Each stage is trained for 800K iterations with the AdamW optimizer on
  4 GPUs (total batch size 32, i.e. `batch_size` 8 per GPU); the learning
  rate follows a cosine schedule from 2e-4 to 2e-5.
- The loss is the sum of the Charbonnier loss and the census loss between
  the ground truth and the interpolation estimated at the bottom pyramid
  level (following UPR-Net).
- The data augmentation follows UPR-Net: random cropping to 256x256 patches,
  channel reversal, vertical/horizontal flips, rotations, and temporal
  reversal.

### Alternative control-point modes

The `--training_mode` argument also supports other configurations of the
control points (used for ablations in the paper):

- `control_from_data`: do not train the BCM; use control points pre-computed
  on the dataset (see `getflow/`)
- `linear_motion`: do not train the BCM; assume uniform linear motion
  (`B = 0.5 * F`)

#### Generate reference Bezier control points (for `control_from_data`)
The `control_from_data` mode needs the RAFT checkpoint. Download
`raft-things.pth` into `getflow/RAFT/models/` (see
`getflow/RAFT/download_models.sh`), then pre-compute the control points for
the Vimeo90K triplets:
```
cd getflow
python get_flow.py --sample_list_path tri_trainlist.txt --root_path /path/to/vimeo_triplet/ --sample_length 3
python get_flow.py --sample_list_path tri_testlist.txt --root_path /path/to/vimeo_triplet/ --sample_length 3
```
This stores a `bezier_1_3.npy` / `bezier_3_1.npy` next to every triplet,
which `VimeoDataset_point` reads during training.

Please assign `data_root` to the path of the Vimeo90K split used for training
(septuplet for stage 1, triplet for stage 2), and optionally assign
`train_log_root` as the path to save logs (trained weights and tensorboard
logs). We do not recommend saving logs under the codebase directory. If
`train_log_root` is not explicitly assigned, all logs will be saved in
`./train-log` by default.


### Some tips for training
- If you want to train large or LARGE versions of our model, please assign
  the argument `model_size` as `large` or `LARGE`.

- If you have suspended the training and want to restart from a previous
  checkpoint, please assign the argument `resume` as `True` in the training
  command.

- By default, we set total batch_size as 32, and use 4 GPUs for distributed
  training, with each GPU processing 8 samples in a batch (`batch_size` is
  set as 8 in our training command). Therefore, if you use 2 GPUs for
  training, please set `batch_size` as 16 in the training command.

- You can view the training curve, interpolation, and optical flow using
  TensorBoard, by running a command like `tensorboard
  --logdir=./train-log/abc-base/tensorboard`.


## Benchmarking

#### Trained model weights
We have placed our trained model weights in `./checkpoints`. The weights of
base/large/LARGE versions of our model are named as `abc_base.pkl`,
`abc_large.pkl`, `abc_llarge.pkl`, respectively.

#### Benchmark datasets
We evaluate our model on Vimeo90K, UCF101, SNU-FILM, and 4K1000FPS. If you
want to benchmark our model, please download
[Vimeo90K](http://toflow.csail.mit.edu/),
[UCF101](https://liuziwei7.github.io/projects/VoxelFlow),
[SNU-FILM](https://myungsub.github.io/CAIN/),
[4K1000FPS](https://github.com/JihyongOh/XVFI#X4K1000FPS).

#### Benchmarking scripts
We provide scripts to test frame interpolation accuracy on Vimeo90K, UCF101,
SNU-FILM, and 4K1000FPS. You should configure the path to benchmark datasets
when running these scripts.
```
python3 -m tools.benchmark_vimeo90k --data_root /path/to/vimeo_triplet/
python3 -m tools.benchmark_ucf101 --data_root /path/to/ucf101/
python3 -m tools.benchmark_snufilm --data_root /path/to/SNU-FILM/
python3 -m tools.benchmark_8x_4k1000fps --test_data_path /path/to/4k1000fps/test
```
By default, we test the base version of our model. To test the large/LARGE
versions, please change corresponding arguments (`model_size` and `model_file`)
in benchmarking scripts.

Additionally, run the following command can test our runtime.
```
python -m tools.runtime
```


## Acknowledgement
We borrow some codes from
[RIFE](https://github.com/megvii-research/ECCV2022-RIFE),
[softmax-splatting](https://github.com/sniklaus/softmax-splatting),
[EBME](https://github.com/srcn-ivl/EBME) and
[UPR-Net](https://github.com/srcn-ivl/UPR-Net).
[RAFT](https://github.com/princeton-vl/RAFT) is used in `getflow/` for
reference control-point generation. We thank the authors for their excellent
work. When using our code, please also pay attention to the licenses of
RIFE, softmax-splatting, EBME, UPR-Net and RAFT.
