import os
import sys
import torch
import numpy as np
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from .loss import EPE, Ternary, LapLoss, LPIPSLoss

from core.models.abc_base import Model as base_model
from core.models.abc_large import Model as large_model
from core.models.abc_llarge import Model as LARGE_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONTROL_FROM_DATA = "control_from_data"
LINEAR_MOTION = "linear_motion"
LEARNED_CONTROL = "learned_control"
RAFT_GT_CONTROL = "raft_gt_control"



class Pipeline:
    def __init__(self,
            model_cfg_dict,
            optimizer_cfg_dict=None,
            local_rank=-1,
            training_mode="learned_control",
            resume=False
            ):
        self.model_cfg_dict = model_cfg_dict
        self.optimizer_cfg_dict = optimizer_cfg_dict or {}
        self.epe = EPE()
        self.ter = Ternary()
        self.laploss = LapLoss()
        self.lpips_weight = self.optimizer_cfg_dict.get("lpips_weight", 0.01)
        self.lpips_loss = LPIPSLoss() if self.lpips_weight > 0 else None
        # Extra supervision that pushes each forward-warped input frame towards
        # the GT (masked to valid, non-hole regions).  0.0 keeps legacy behavior.
        self.warp_loss_weight = self.optimizer_cfg_dict.get(
                "warp_loss_weight", 0.0)
        self.raft_control_estimator = None
        # The raft estimator is only needed for the ablation mode that derives
        # ground-truth Bezier control points from online RAFT flow.
        if training_mode == RAFT_GT_CONTROL:
            self.init_raft_control_estimator()
        # self.flowloss = FlowLoss()

        self.init_model()
        self.device()
        self.training_mode = training_mode

        # AdamW uses one learning-rate schedule for every parameter.
        # The control estimator is frozen in the modes that do not learn it.
        if training_mode in {
                CONTROL_FROM_DATA, LINEAR_MOTION, RAFT_GT_CONTROL}:
            for p in self.model.control_estimator.parameters():
                p.requires_grad = False

        self.optimG = None
        if optimizer_cfg_dict is not None:
            self.optimG = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.optimizer_cfg_dict["init_lr"],
                weight_decay=self.optimizer_cfg_dict["weight_decay"])

        # `local_rank == -1` is used for testing, which does not need DDP
        if local_rank != -1:
            self.model = DDP(self.model, device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=False)

        # Restart the experiment from last saved model, by loading the state of
        # the optimizer
        if resume:
            assert training_mode, "To restart the training, please init the"\
                    "pipeline with a training mode!"
            print("Load optimizer state to restart the experiment")
            ckpt_dict = torch.load(self.optimizer_cfg_dict["ckpt_file"])
            self.optimG.load_state_dict(ckpt_dict["optimizer"])


    def train(self):
        self.model.train()


    def eval(self):
        self.model.eval()


    def device(self):
        self.model.to(DEVICE)


    def init_raft_control_estimator(self):
        project_root = os.path.dirname(os.path.dirname(__file__))
        getflow_root = os.path.join(project_root, "getflow")
        raft_root = os.path.join(getflow_root, "RAFT")
        raft_core = os.path.join(raft_root, "core")
        for path in (getflow_root, raft_root, raft_core):
            if path not in sys.path:
                sys.path.insert(0, path)
        abc_core = sys.modules.pop("core", None)
        for name in list(sys.modules):
            if name.startswith("core."):
                sys.modules.pop(name, None)
        try:
            from core.raft import RAFT
            from core.utils.utils import InputPadder
        finally:
            for name in list(sys.modules):
                if name == "core" or name.startswith("core."):
                    sys.modules.pop(name, None)
            if abc_core is not None:
                sys.modules["core"] = abc_core
        from easydict import EasyDict as edict

        checkpoint = self.optimizer_cfg_dict.get(
                "raft_checkpoint",
                os.path.join(raft_root, "models", "raft-things.pth"))
        if not os.path.isabs(checkpoint):
            checkpoint = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)), checkpoint)
        args = edict({
                'mixed_precision': False,
                'small': False,
                'alternate_corr': False})
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(checkpoint))
        model = model.module.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.raft_control_estimator = model
        self.raft_input_padder_cls = InputPadder


    @torch.no_grad()
    def estimate_raft_flow(self, img0, img1):
        padder = self.raft_input_padder_cls(img0.shape)
        img0_pad, img1_pad = padder.pad(img0 * 255., img1 * 255.)
        _, flow = self.raft_control_estimator(
                img0_pad, img1_pad, iters=20, test_mode=True)
        return padder.unpad(flow)


    @torch.no_grad()
    def compute_gt_control_points(self, img0, img1, gt, time_period=0.5):
        flow_0t = self.estimate_raft_flow(img0, gt)
        flow_01 = self.estimate_raft_flow(img0, img1)
        flow_1t = self.estimate_raft_flow(img1, gt)
        flow_10 = self.estimate_raft_flow(img1, img0)
        t = time_period
        denom = 2 * (1 - t) * t + 1e-6
        point_0 = (flow_0t - flow_01 * t.square()) / denom
        point_1 = (flow_1t - flow_10 * (1 - t).square()) / denom
        return torch.cat([point_0, point_1], dim=1)


    @staticmethod
    def convert_state_dict(rand_state_dict, pretrained_state_dict):
        param =  {
            k.replace("module.", "", 1): v
            for k, v in pretrained_state_dict.items()
            }
        param = {k: v
                for k, v in param.items()
                if ((k in rand_state_dict) and (rand_state_dict[k].shape \
                        == param[k].shape))
                }
        rand_state_dict.update(param)
        return rand_state_dict


    def init_model(self):

        def load_pretrained_state_dict(model, model_file):
            if (model_file == "") or (not os.path.exists(model_file)):
                raise ValueError(
                        "Please set the correct path for pretrained model!")

            print("Load pretrained model from %s."  % model_file)
            rand_state_dict = model.state_dict()
            pretrained_state_dict = torch.load(model_file)

            return Pipeline.convert_state_dict(
                    rand_state_dict, pretrained_state_dict)

        # check args
        model_cfg_dict = self.model_cfg_dict
        model_size = model_cfg_dict["model_size"] \
                if "model_size" in model_cfg_dict else "base"
        pyr_level = model_cfg_dict["pyr_level"] \
                if "pyr_level" in model_cfg_dict else 3
        nr_lvl_skipped = model_cfg_dict["nr_lvl_skipped"] \
                if "nr_lvl_skipped" in model_cfg_dict else 0
        load_pretrain = model_cfg_dict["load_pretrain"] \
                if "load_pretrain" in model_cfg_dict else False
        model_file = model_cfg_dict["model_file"] \
                if "model_file" in model_cfg_dict else ""

        # instantiate model
        if model_size == "LARGE":
            self.model = LARGE_model(pyr_level, nr_lvl_skipped)
        elif model_size == "large":
            self.model = large_model(pyr_level, nr_lvl_skipped)
        else:
            self.model = base_model(pyr_level, nr_lvl_skipped)

        # load pretrained model weight
        if load_pretrain:
            state_dict = load_pretrained_state_dict(
                    self.model, model_file)
            self.model.load_state_dict(state_dict)
        else:
            print("Train from random initialization.")


    def save_optimizer_state(self, path, rank, step):
        if rank == 0:
            optimizer_ckpt = {
                     "optimizer": self.optimG.state_dict(),
                     "step": step
                     }
            torch.save(optimizer_ckpt, "{}/optimizer-ckpt.pth".format(path))


    def save_model(self, path, rank, save_step=None):
        if (rank == 0) and (save_step is None):
            torch.save(self.model.state_dict(), '{}/model.pkl'.format(path))
        if (rank == 0) and (save_step is not None):
            torch.save(self.model.state_dict(), '{}/model-{}.pkl'\
                    .format(path, save_step))

    def inference(self, img0, img1,
            time_period=0.5, B_point=None,
            pyr_level=3,
            nr_lvl_skipped=0,
            control_mode=None):
        if control_mode is None:
            control_mode = self.training_mode
        # The raft_gt_control mode derives control points from the
        # ground-truth frame, which is unavailable at inference time.  If the
        # caller does not supply control points, fall back to a uniform-motion
        # assumption (B = 0.5 * F), as done in the paper.
        if control_mode == RAFT_GT_CONTROL and B_point is None:
            control_mode = LINEAR_MOTION
        interp_img, bi_flow, extra_dict = self.model(img0, img1,
                time_period=time_period,
                B_point=B_point,
                pyr_level=pyr_level,
                nr_lvl_skipped=nr_lvl_skipped,
                control_mode=control_mode)
        return interp_img, bi_flow, extra_dict


    def train_one_iter(self, img0, img1, gt, B_point=None, learning_rate=0, time_period=0.5, loss_type="l2+census"):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        self.train()

        if self.training_mode == RAFT_GT_CONTROL:
            B_point = self.compute_gt_control_points(
                    img0, img1, gt, time_period=time_period)
        interp_img, bi_flow, model_extra_dict = self.model(
            img0, img1, time_period, B_point=B_point,
            control_mode=(CONTROL_FROM_DATA
                    if self.training_mode == RAFT_GT_CONTROL
                    else self.training_mode))

        with torch.no_grad():
            loss_interp_l2_nograd = (((interp_img - gt) ** 2 + 1e-6) ** 0.5)\
                    .mean()

        loss_G = 0
        if self.lpips_loss is not None:
            lpips_loss = self.lpips_loss(interp_img, gt)
        else:
            lpips_loss = interp_img.new_tensor(0.0)
        if loss_type == "l1":
            loss_G = (interp_img - gt).abs().mean()
        elif loss_type == "l2":
            loss_interp_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_G = loss_interp_l2
        elif loss_type == "l2+census":
            loss_interp_l2 = (((interp_img - gt) ** 2 + 1e-6) ** 0.5).mean()
            loss_ter = self.ter(interp_img, gt).mean()
            # bi_flow[:, :2] = 2*time_period*(1-time_period)*bi_flow[:, :2] + time_period * time_period * bi_flow[:, 2:4]
            # bi_flow[:, 4:6] = 2*time_period*(1-time_period)*bi_flow[:, 4:6] + (1-time_period) * (1-time_period) * bi_flow[:, 6:]
            # # bi_flow[0:2] = (bi_flow[2:4] - 0.5 * bi_flow[0:2]) * time_period + 0.5 * bi_flow[0:2] * time_period * time_period
            # # bi_flow[4:6] = (bi_flow[6:8] - 0.5 * bi_flow[4:6]) * (1-time_period) + 0.5 * bi_flow[4:6] * (1-time_period) * (1-time_period)
            # loss_flow = (((bi_flow - flow) ** 2 + 1e-6) ** 0.5).mean()
            # loss_flow = self.flowloss(bi_flow,torch.cat([img0,img1])).mean()
            loss_G = loss_interp_l2 + loss_ter
        else:
            ValueError("unsupported loss type!")

        loss_G = loss_G + self.lpips_weight * lpips_loss

        # Supervise each forward-warped input frame against the GT so that the
        # flow / control estimator aligns the whole frame (large-motion
        # background), not only the regions that happen to overlap after merge.
        # Holes left by softsplat are excluded via the coverage mask.
        warp_loss = interp_img.new_tensor(0.0)
        if self.warp_loss_weight > 0:
            warped_img0 = model_extra_dict["warped_img0"]
            warped_img1 = model_extra_dict["warped_img1"]
            warp_mask0 = model_extra_dict["warp_mask0"]
            warp_mask1 = model_extra_dict["warp_mask1"]
            eps = 1e-6
            warp_loss0 = ((((warped_img0 - gt) ** 2 + eps) ** 0.5)
                    * warp_mask0).sum() / (warp_mask0.sum() + eps)
            warp_loss1 = ((((warped_img1 - gt) ** 2 + eps) ** 0.5)
                    * warp_mask1).sum() / (warp_mask1.sum() + eps)
            warp_loss = warp_loss0 + warp_loss1
            loss_G = loss_G + self.warp_loss_weight * warp_loss


        self.optimG.zero_grad()
        loss_G.backward()
        # for name, param in self.model.named_parameters():
        #     if param.grad is None:
        #         print(name)
        self.optimG.step()

        extra_dict = {}
        extra_dict["loss_interp_l2"] = loss_interp_l2_nograd
        extra_dict["loss_warp"] = warp_loss.detach()
        extra_dict["bi_flow"] = bi_flow

        return interp_img, extra_dict



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pass
