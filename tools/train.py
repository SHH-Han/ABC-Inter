import os
import sys
import shutil
import cv2
import math
import time
import numpy as np
import random
import argparse
import warnings
from importlib import import_module
from distutils.util import strtobool
from loguru import logger
from typing import Dict, Tuple

import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from core.pipeline import Pipeline
from core.dataset import (
    VimeoDataset,
    VimeoDataset_point,
    VimeoSeptupletDataset,
    VimeoSeptupletEvalDataset,
)

try:
    import lpips
except ImportError:
    lpips = None

try:
    import pyiqa
except ImportError:
    pyiqa = None

CONTROL_FROM_DATA = "control_from_data"
LINEAR_MOTION = "linear_motion"
LEARNED_CONTROL = "learned_control"
RAFT_GT_CONTROL = "raft_gt_control"

warnings.filterwarnings("ignore")


def get_learning_rate(total_step, cur_step, init_lr, min_lr=1e-6):
    if cur_step < 2000:
        mul = cur_step / 2000.
        return init_lr * mul
    else:
        mul = np.cos((cur_step - 2000) / (total_step - 2000.) * math.pi)\
                * 0.5 + 0.5
        return  (init_lr - min_lr) * mul + min_lr


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= (0.5 * (normalized_flow_map[:, :, 0]\
            + normalized_flow_map[:, :, 1]))
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def bgr_to_rgb_tensor(tensor):
    return tensor[:, [2, 1, 0], :, :]


def load_eval_metrics(device):
    if lpips is None:
        raise ImportError(
                "lpips is required for validation metrics. "
                "Install it with `pip install lpips`.")
    if pyiqa is None:
        raise ImportError(
                "pyiqa is required for NIQA/NIQE validation metrics. "
                "Install it with `pip install pyiqa`.")
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    niqa_model = pyiqa.create_metric('niqe', device=device)
    return lpips_model, niqa_model


EVAL_LPIPS_MODEL = None
EVAL_NIQA_MODEL = None


def train(ppl, dataset_cfg_dict, optimizer_cfg_dict):
    model_log_dir = os.path.join(THIS_EXP_LOG_DIR, "trained-models")
    tf_log_dir = os.path.join(THIS_EXP_LOG_DIR, "tensorboard")

    # dataset config
    batch_size = dataset_cfg_dict.get("batch_size", 32)
    eval_batch_size = dataset_cfg_dict.get("eval_batch_size", batch_size)
    nr_data_worker = dataset_cfg_dict.get("nr_data_worker", 4)
    crop_h = dataset_cfg_dict.get("crop_h", 256)
    crop_w = dataset_cfg_dict.get("crop_w", 256)
    training_mode = dataset_cfg_dict["training_mode"]
    use_dataset_control_points = training_mode == CONTROL_FROM_DATA
    data_root = dataset_cfg_dict["data_root"]
    dataset_type = dataset_cfg_dict["dataset_type"]
    # The control_from_data mode reads pre-computed Bezier control points that
    # sit next to each Vimeo90K triplet (see getflow/); every other mode only
    # needs the RGB triplet.
    if dataset_type == "septuplet":
        dataset_train = VimeoSeptupletDataset(
                dataset_name='train', data_root=data_root,
                crop_h=crop_h, crop_w=crop_w)
        dataset_val = VimeoSeptupletEvalDataset(
                dataset_name='validation', data_root=data_root)
    elif use_dataset_control_points:
        dataset_train = VimeoDataset_point(dataset_name='train', data_root=data_root)
        dataset_val = VimeoDataset_point(dataset_name='validation', data_root=data_root)
    else:
        dataset_train = VimeoDataset(dataset_name='train', data_root=data_root)
        dataset_val = VimeoDataset(dataset_name='validation', data_root=data_root)

    sampler = DistributedSampler(dataset_train)
    val_sampler = DistributedSampler(dataset_val, shuffle=False)
    train_data = DataLoader(dataset_train, batch_size=batch_size,
            num_workers=nr_data_worker,
            pin_memory=True, drop_last=True, sampler=sampler)
    val_data = DataLoader(dataset_val, batch_size=eval_batch_size,
            num_workers=nr_data_worker, pin_memory=True, sampler=val_sampler)

    # optimizer config
    total_step = int(optimizer_cfg_dict["steps"])
    save_interval = int(optimizer_cfg_dict["save_interval"])
    init_lr = optimizer_cfg_dict.get("init_lr", 2e-5)
    min_lr = optimizer_cfg_dict.get("min_lr", 2e-6)
    loss_type = optimizer_cfg_dict.get("loss_type", "l2+census")
    metric_batch_size = optimizer_cfg_dict.get("metric_batch_size", 16)

    step = 1
    if RESUME:
        optimizer_ckpt_file = optimizer_cfg_dict["ckpt_file"]
        info_dict = torch.load(optimizer_ckpt_file)
        step = info_dict["step"] + 1
        if step > total_step:
            raise ValueError(
                    "Previous trained steps have exceeded the total step of"\
                    "this experiment!  Please check the value of current step"\
                    "and total step.")

    writer = None
    writer_val = None
    is_write = False if RESUME else True
    if LOCAL_RANK == 0:
        writer = SummaryWriter(tf_log_dir + '/train')
        writer_val = SummaryWriter(tf_log_dir + '/validate')
    if not RESUME:
        if not dataset_cfg_dict.get("skip_initial_eval", False):
            evaluate(ppl, step, val_data, writer_val, training_mode, is_write,
                    metric_batch_size=metric_batch_size)
        ppl.save_model(model_log_dir, LOCAL_RANK)

    time_stamp = time.time()
    step_per_epoch = len(train_data)
    epoch_counter = 0
    last_epoch = False

    while step <  total_step+1:
        if step + step_per_epoch >  total_step:
            last_epoch = True

        epoch_counter += 1
        sampler.set_epoch(epoch_counter)

        for data_batch in train_data:
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            # Septuplet batches carry a per-sample time period (the target is
            # a random middle frame, t in (0, 1)); triplet batches default to
            # t = 0.5 (the middle frame).
            if isinstance(data_batch, (tuple, list)):
                data, time_period = data_batch
            else:
                data, time_period = data_batch, None
            data_gpu = data.to(
                    DEVICE, dtype=torch.float, non_blocking=True)
            data_gpu[:,:9] = data_gpu[:,:9] / 255.
            if time_period is None:
                time_period = torch.full(
                        (data_gpu.shape[0], 1, 1, 1), 0.5,
                        device=DEVICE, dtype=data_gpu.dtype)
            else:
                time_period = time_period.to(
                        DEVICE, dtype=data_gpu.dtype, non_blocking=True)
                time_period = time_period.reshape(-1, 1, 1, 1)

            img0 = data_gpu[:, :3]
            img1 = data_gpu[:, 3:6]
            gt = data_gpu[:, 6:9]
            B = None
            if training_mode == CONTROL_FROM_DATA:
                B = data_gpu[:,9:]
            learning_rate = get_learning_rate(total_step, step, init_lr, min_lr)
            pred, extra_dict = ppl.train_one_iter(
                    img0, img1, gt, B_point=B,
                    learning_rate=learning_rate,
                    time_period=time_period,
                    loss_type=loss_type)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if step % 100 == 1 and LOCAL_RANK == 0:
                writer.add_scalar(
                        '1-loss_interp_l2', extra_dict["loss_interp_l2"] , step)
                writer.add_scalar('2-learning_rate', learning_rate, step)
                if "loss_warp" in extra_dict:
                    writer.add_scalar(
                            '3-loss_warp', extra_dict["loss_warp"], step)
            if step % 1000 == 1 and LOCAL_RANK == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)\
                        .astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255)\
                        .astype('uint8')
                overlay = 0.5 * img0 + 0.5 * img1
                overlay = (overlay.permute(0, 2, 3, 1).detach().cpu().numpy()\
                        * 255).astype('uint8')
                bi_flow = extra_dict["bi_flow"]
                bi_flow = bi_flow.permute(0, 2, 3, 1).detach().cpu().numpy()
                nr_show = min(6, batch_size)
                for i in range(nr_show):
                    imgs = np.concatenate((overlay[i], gt[i], pred[i]), 1)\
                            [:, :, ::-1]
                    writer.add_image(
                            str(i) + '/0-overlay-gt-pred',
                            imgs, step, dataformats='HWC')
                    writer.add_image(
                            str(i) + '/1-flow_01_pred',
                            flow2rgb(bi_flow[i][:, :, :2]),
                            step, dataformats='HWC')
                writer.flush()
            if LOCAL_RANK == 0:
                print("{} => train step: {}/{}; time: {:.2f}+{:.2f}; "\
                        "loss_interp_l2: {:.4e}".format(
                            EXP_NAME, step, total_step,
                            data_time_interval, train_time_interval,
                            extra_dict["loss_interp_l2"]))

            if step % save_interval == 0:
                psnr = evaluate(ppl, step, val_data, writer_val, training_mode,
                        metric_batch_size=metric_batch_size)
                if LOCAL_RANK == 0:
                    ppl.save_model(model_log_dir, LOCAL_RANK)
                    ppl.save_optimizer_state(THIS_EXP_LOG_DIR, LOCAL_RANK, step)
                    logger.info("{} => val step: {}; "\
                            "psnr: {:.4f}".format(EXP_NAME, step, psnr))
                    if step % (save_interval * 50) == 0:
                        ppl.save_model(model_log_dir, LOCAL_RANK, save_step=step)
                        ppl.save_optimizer_state(THIS_EXP_LOG_DIR, LOCAL_RANK, step)

            step += 1
            if last_epoch and step == total_step + 1:
                break

        dist.barrier()


def evaluate(ppl, step, val_data, writer_val, training_mode, is_write=True,
        metric_batch_size=16):
    global EVAL_LPIPS_MODEL, EVAL_NIQA_MODEL
    if EVAL_LPIPS_MODEL is None or EVAL_NIQA_MODEL is None:
        EVAL_LPIPS_MODEL, EVAL_NIQA_MODEL = load_eval_metrics(DEVICE)
    psnr_sum = torch.zeros(1, device=DEVICE)
    lpips_sum = torch.zeros(1, device=DEVICE)
    niqa_sum = torch.zeros(1, device=DEVICE)
    metric_count = torch.zeros(1, device=DEVICE)
    start_time_stamp = time.time()
    time_stamp = start_time_stamp
    nr_val = val_data.__len__()
    for i, data_batch in enumerate(val_data):
        data_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        with torch.no_grad():
            if isinstance(data_batch, (tuple, list)) and len(data_batch) == 4:
                # Septuplet eval: one input pair with n middle-frame targets.
                img0, img1, gt, time_period = data_batch
                b, n, c, h, w = gt.shape
                img0 = img0.to(DEVICE, dtype=torch.float, non_blocking=True) / 255.
                img1 = img1.to(DEVICE, dtype=torch.float, non_blocking=True) / 255.
                gt = gt.to(DEVICE, dtype=torch.float, non_blocking=True) / 255.
                time_period = time_period.to(
                        DEVICE, dtype=torch.float, non_blocking=True)
                img0 = img0[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, c, h, w)
                img1 = img1[:, None].expand(-1, n, -1, -1, -1).reshape(b * n, c, h, w)
                gt = gt.reshape(b * n, c, h, w)
                time_period = time_period.reshape(b * n, 1, 1, 1)
            else:
                # Triplet batches: the middle frame (t = 0.5) is the target.
                if isinstance(data_batch, (tuple, list)):
                    data, time_period = data_batch
                else:
                    data, time_period = data_batch, None
                data_gpu = data.to(
                        DEVICE, dtype=torch.float, non_blocking=True)
                data_gpu[:,:9] = data_gpu[:,:9] / 255.
                if time_period is None:
                    time_period = torch.full(
                            (data_gpu.shape[0], 1, 1, 1), 0.5,
                            device=DEVICE, dtype=data_gpu.dtype)
                else:
                    time_period = time_period.to(
                            DEVICE, dtype=data_gpu.dtype, non_blocking=True)
                    time_period = time_period.reshape(-1, 1, 1, 1)
                img0 = data_gpu[:, :3]
                img1 = data_gpu[:, 3:6]
                gt = data_gpu[:, 6:9]
            # Only the learned_control model produces usable control points at
            # inference time (via its control_estimator); every other training
            # mode has no control points to feed at eval, so fall back to
            # linear motion.
            eval_control_mode = LEARNED_CONTROL \
                    if training_mode == LEARNED_CONTROL else LINEAR_MOTION
            # The pyramid model requires spatial dims divisible by 32; full
            # resolution eval frames (e.g. 540x960) are not, so pad to the
            # next multiple of 32 and crop the prediction back afterwards.
            orig_h, orig_w = img0.shape[-2:]
            pad_h = (32 - orig_h % 32) % 32
            pad_w = (32 - orig_w % 32) % 32
            if pad_h or pad_w:
                img0 = F.pad(img0, (0, pad_w, 0, pad_h), mode='replicate')
                img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode='replicate')
            pred, bi_flow, extra_dict = ppl.inference(
                    img0, img1, B_point=None, time_period=time_period,
                    control_mode=eval_control_mode)
            if pad_h or pad_w:
                pred = pred[..., :orig_h, :orig_w]
        pred = pred.clamp(0.0, 1.0)
        mse = torch.mean((gt - pred) * (gt - pred), dim=(1, 2, 3))
        psnr = -10 * torch.log10(mse + 1e-12)
        pred_rgb = bgr_to_rgb_tensor(pred)
        gt_rgb = bgr_to_rgb_tensor(gt)
        with torch.no_grad():
            batch_lpips_sum = pred.new_tensor(0.0)
            batch_niqa_sum = pred.new_tensor(0.0)
            for start in range(0, pred_rgb.shape[0], metric_batch_size):
                end = start + metric_batch_size
                pred_chunk = pred_rgb[start:end]
                gt_chunk = gt_rgb[start:end]
                lpips_val = EVAL_LPIPS_MODEL(
                        pred_chunk * 2 - 1, gt_chunk * 2 - 1)
                niqa_val = EVAL_NIQA_MODEL(pred_chunk)
                batch_lpips_sum += lpips_val.reshape(-1).sum()
                batch_niqa_sum += niqa_val.reshape(-1).sum()
        psnr_sum += psnr.sum()
        lpips_sum += batch_lpips_sum
        niqa_sum += batch_niqa_sum
        metric_count += psnr.numel()
        eval_time_interval = time.time() - time_stamp
        time_stamp = time.time()
        if LOCAL_RANK == 0:
            print('{} => val step: {}: {}/{}; time: {:.2f}+{:.2f}'.format(
                    EXP_NAME, step, i, nr_val,
                    data_time_interval, eval_time_interval))

    dist.all_reduce(psnr_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(lpips_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(niqa_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(metric_count, op=dist.ReduceOp.SUM)
    psnr = (psnr_sum / metric_count).item()
    lpips_score = (lpips_sum / metric_count).item()
    niqa_score = (niqa_sum / metric_count).item()
    if LOCAL_RANK == 0:
        print('eval time: {}'.format(time.time() - start_time_stamp))
        if is_write:
            writer_val.add_scalar('0-psnr', psnr, step)
            writer_val.add_scalar('1-lpips', lpips_score, step)
            writer_val.add_scalar('2-niqa', niqa_score, step)
        logger.info('{} => val step: {}; psnr: {:.4f}; lpips: {:.4f}; '
                'niqa: {:.4f}'.format(
                    EXP_NAME, step, psnr, lpips_score, niqa_score))

    return psnr


def init_exp_env():
    def prompt(query):
        sys.stdout.write("%s [y/n]:" % query)
        val = input()

        try:
            ret = strtobool(val)
        except ValueError:
            sys.stdout("please answer with y/n")
            return prompt(query)
        return ret

    # process the path
    if (LOCAL_RANK == 0) and (not RESUME):
        # init train log dir, model and tf dir
        if os.path.exists(THIS_EXP_LOG_DIR):
            while True:
                if prompt("Would you like to re-write"\
                        "the existing experimental saving dir?") == True:
                    shutil.rmtree(THIS_EXP_LOG_DIR)
                    break
                else:
                    print("Exit the program."\
                            "Please assign another expriment name!")
                    exit()

        train_log_dir_link = os.path.join(THIS_CODEBASE_DIR, "train-log")
        this_exp_model_dir = os.path.join(THIS_EXP_LOG_DIR, "trained-models")
        this_exp_tf_dir = os.path.join(THIS_EXP_LOG_DIR, "tensorboard")
        if not os.path.exists(TRAIN_LOG_ROOT):
            os.makedirs(TRAIN_LOG_ROOT)
        if not os.path.exists(train_log_dir_link):
            cmd = "ln -s %s %s" % (TRAIN_LOG_ROOT, train_log_dir_link)
            os.system(cmd)
        os.makedirs(THIS_EXP_LOG_DIR)
        os.makedirs(this_exp_model_dir)
        os.makedirs(this_exp_tf_dir)

    # set logger file
    if LOCAL_RANK == 0:
        logger.add(os.path.join(THIS_EXP_LOG_DIR, "runtime.log"))

    # init cuda env
    torch.distributed.init_process_group(backend="nccl", world_size=WORLD_SIZE)
    torch.cuda.set_device(LOCAL_RANK)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='train upr-net for video frame interpolation')

    # => args for basic information
    parser.add_argument('--exp_name', default="abc-base", type=str,
            help='experiment name, will be used to save all generated files')
    parser.add_argument('--train_log_root', default="train-log", type=str,
            help='root dir to save all training logs')
    parser.add_argument('--resume', default=False, type=bool,
            help='resume from previously saved experiment logs')

    #**********************************************************#
    # => args for distributed training
    # torch >= 2.0 passes "--local-rank" (hyphen) while older torch used
    # "--local_rank" (underscore); accept both so the same code runs under
    # either launcher.
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank',
            default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')

    #**********************************************************#
    # => args for data loader and rand crop size
    parser.add_argument('--data_root', type=str, default='',
            help='root dir of the Vimeo90K dataset (triplet or septuplet)')
    parser.add_argument('--dataset_type', type=str, default='septuplet',
            choices=['triplet', 'septuplet'],
            help='dataset layout to use for training; "septuplet" samples '
                 'arbitrary middle frames (first training stage), "triplet" '
                 'uses the fixed middle frame (second training stage)')
    parser.add_argument('--batch_size', type=int, default=8,
            help='batch size for data loader')
    parser.add_argument('--eval_batch_size', type=int, default=None,
            help='batch size for validation; defaults to batch_size')
    parser.add_argument('--skip_initial_eval', action='store_true',
            help='skip the validation pass before the first training step')
    parser.add_argument('--nr_data_worker', type=int, default=2,
            help='number of the worker for data loader')
    parser.add_argument('--crop_h', type=int, default=256,
            help='height of cropped patch')
    parser.add_argument('--crop_w', type=int, default=256,
            help='width of cropped patch')

    #**********************************************************#
    # => args for model
    parser.add_argument('--model_size', type=str, default="base",
            help='model size, one of (base, large, LARGE)')
    parser.add_argument('--pyr_level', type=int, default=3,
            help='the number of pyramid levels of ABC during training')
    # parser.add_argument('--nr_lvl_skipped', type=int, default=0,
    #         help='the number of skipped high-resolution levels for 4K input')
    parser.add_argument('--load_pretrain', action='store_true')
    parser.add_argument('--training_mode', type=str, default=LEARNED_CONTROL,
            choices=[CONTROL_FROM_DATA, LINEAR_MOTION, LEARNED_CONTROL,
                    RAFT_GT_CONTROL],
            help='how to handle control points during training and inference')
    parser.add_argument('--finetune', action='store_true',
            help='deprecated alias for --training_mode learned_control')
    parser.add_argument('--use_bcp', action='store_true',
            help='deprecated alias for --training_mode control_from_data')
    parser.add_argument('--model_file', type=str, default="",
            help='weight of ABC-Inter')

    #**********************************************************#
    # => args for optimizer
    parser.add_argument('--init_lr', type=float, default=2e-4,
            help='init learning rate')
    parser.add_argument('--min_lr', type=float, default=2e-5,
            help='min learning rate, till the end of training')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
            help='wegith decay')
    parser.add_argument('--lpips_weight', type=float, default=0.01,
            help='weight of the LPIPS perceptual loss term')
    parser.add_argument('--warp_loss_weight', type=float, default=0.0,
            help='weight of the masked warped-frame-vs-GT supervision '
                 '(0 disables it)')
    parser.add_argument('--metric_batch_size', type=int, default=16,
            help='chunk size for LPIPS and NIQA validation metrics')
    parser.add_argument('--raft_checkpoint', type=str,
            default='getflow/RAFT/models/raft-things.pth',
            help='RAFT checkpoint for --training_mode raft_gt_control')
    parser.add_argument('--steps', type=float, default=0.8e6,
            help='total steps (iteration) for training')
    parser.add_argument('--save_interval', type=float, default=0.2e4,
            help='iteration interval to save model')
    parser.add_argument('--loss_type', type=str, default="l2+census",
            help='training loss')

    #**********************************************************#
    # => organize args in groups
    args = parser.parse_args()

    model_cfg_dict = dict(
            model_size = args.model_size,
            pyr_level = args.pyr_level,
            load_pretrain = args.load_pretrain,
            model_file = args.model_file
            )

    optimizer_cfg_dict = dict(
            init_lr=args.init_lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            lpips_weight=args.lpips_weight,
            warp_loss_weight=args.warp_loss_weight,
            metric_batch_size=args.metric_batch_size,
            raft_checkpoint=args.raft_checkpoint,
            steps=args.steps,
            save_interval=args.save_interval,
            loss_type=args.loss_type
            )

    if args.use_bcp:
        args.training_mode = CONTROL_FROM_DATA
    if args.finetune:
        args.training_mode = LEARNED_CONTROL

    dataset_cfg_dict = dict(
            nr_data_worker=args.nr_data_worker,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size or args.batch_size,
            skip_initial_eval=args.skip_initial_eval,
            crop_h=args.crop_h,
            crop_w=args.crop_w,
            data_root=args.data_root,
            dataset_type=args.dataset_type,
            training_mode=args.training_mode
            )


    #**********************************************************#
    # => parse args and init the training environment
    # global variable
    EXP_NAME = args.exp_name
    TRAIN_LOG_ROOT = args.train_log_root
    LOCAL_RANK = args.local_rank
    WORLD_SIZE = args.world_size
    DEVICE = torch.device("cuda", LOCAL_RANK)
    THIS_CODEBASE_DIR = os.path.split(os.path.split(__file__)[0])[0]
    THIS_EXP_LOG_DIR =os.path.join(TRAIN_LOG_ROOT, EXP_NAME)

    optimizer_cfg_dict["ckpt_file"] = os.path.join(THIS_EXP_LOG_DIR, "optimizer-ckpt.pth")
    if not os.path.exists(optimizer_cfg_dict["ckpt_file"]):
        args.resume = False
    if args.resume:
        model_cfg_dict["load_pretrain"] = True
        model_cfg_dict["model_file"] = os.path.join(
                THIS_EXP_LOG_DIR, "trained-models", "model.pkl")
    RESUME = args.resume

    # init the exp environment
    init_exp_env()

    #**********************************************************#
    # => init the pipeline and train the pipeline
    ppl = Pipeline(
            model_cfg_dict, optimizer_cfg_dict,
            LOCAL_RANK, training_mode=args.training_mode, resume=RESUME)
    logger.info("start the training task: %s (%s)" % (EXP_NAME, args.training_mode))
    train(ppl, dataset_cfg_dict, optimizer_cfg_dict)
