import cv2
import os
import torch
import numpy as np
import random
from glob import glob
from PIL import ImageEnhance, Image
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(0)


class SnuFilm(Dataset):
    def __init__(self, data_root, data_type="extreme"):
        self.data_root = data_root
        self.data_type = data_type
        assert data_type in ["easy", "medium", "hard", "extreme"]
        self.load_data()


    def __len__(self):
        return len(self.meta_data)


    def load_data(self):
        if self.data_type == "easy":
            easy_file = os.path.join(self.data_root, "eval_modes/test-easy.txt")
            with open(easy_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "medium":
            medium_file = os.path.join(self.data_root, "eval_modes/test-medium.txt")
            with open(medium_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "hard":
            hard_file = os.path.join(self.data_root, "eval_modes/test-hard.txt")
            with open(hard_file, 'r') as f:
                self.meta_data = f.read().splitlines()
        if self.data_type == "extreme":
            extreme_file = os.path.join(self.data_root, "eval_modes/test-extreme.txt")
            with open(extreme_file, 'r') as f:
                self.meta_data = f.read().splitlines()


    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = imgpath.split()

        # Load images
        img0 = cv2.imread(os.path.join(self.data_root, imgpaths[0]))
        gt = cv2.imread(os.path.join(self.data_root, imgpaths[1]))
        img1 = cv2.imread(os.path.join(self.data_root, imgpaths[2]))

        return img0, gt, img1


    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)


class UCF101(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.load_data()


    def __len__(self):
        return len(self.meta_data)


    def load_data(self):
        triplet_dirs = glob(os.path.join(self.data_root, "*"))
        self.meta_data = triplet_dirs


    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = [os.path.join(imgpath, 'frame_00.png'),
                os.path.join(imgpath, 'frame_01_gt.png'),
                os.path.join(imgpath, 'frame_02.png')]

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1


    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)


class VimeoDataset(Dataset):
    def __init__(self, dataset_name, data_root, has_crop_aug=True):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.has_crop_aug = has_crop_aug
        self.crop_h = 256
        self.crop_w = 256
        self.image_root = os.path.join(self.data_root, 'sequences')

        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.load_data()


    def __len__(self):
        return len(self.meta_data)


    def load_data(self):
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist


    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1


    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = [os.path.join(self.image_root, imgpath, 'im1.png'),
                os.path.join(self.image_root, imgpath, 'im2.png'),
                os.path.join(self.image_root, imgpath, 'im3.png')]

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1


    def __getitem__(self, index):
        img0, gt, img1 = self.getimg(index)
        if self.dataset_name == 'train':
            # random resize
            # if random.uniform(0, 1) < 0.1:
            #     img0 = cv2.resize(img0, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR)
            #     img1 = cv2.resize(img1, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR)
            #     gt = cv2.resize(gt, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR)
            #     flow_gt = cv2.resize(flow_gt, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR) * 2.0

            # random crop
            if self.has_crop_aug:
                img0, gt, img1 = self.aug(img0, gt, img1, self.crop_h, self.crop_w)
            # random channel reverse
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            # random vertical flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            # random horizontal flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            # random rotation
            if random.uniform(0, 1) < 0.5:
                rot_option = np.random.randint(1, 4)
                img0 = np.rot90(img0, rot_option)
                img1 = np.rot90(img1, rot_option)
                gt = np.rot90(gt, rot_option)
            # random temporal reverse
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        return torch.cat((img0, img1, gt), 0)

class VimeoDataset_point(Dataset):
    def __init__(self, dataset_name, data_root, has_crop_aug=True):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.has_crop_aug = has_crop_aug
        self.crop_h = 256
        self.crop_w = 256
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.load_data()


    def __len__(self):
        return len(self.meta_data)


    def load_data(self):
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist


    def aug(self, img0, gt, img1, flow, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow = flow[x:x+h, y:y+w, :]
        return img0, gt, img1, flow


    def getimg(self, index):
        imgpath = self.meta_data[index]
        imgpaths = [os.path.join(self.image_root, imgpath, 'im1.png'),
                os.path.join(self.image_root, imgpath, 'im2.png'),
                os.path.join(self.image_root, imgpath, 'im3.png')]

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])

        B_point0 = np.load(os.path.join(self.image_root, imgpath, 'bezier_1_3.npy'))
        B_point1 = np.load(os.path.join(self.image_root, imgpath, 'bezier_3_1.npy'))
        B_point = np.concatenate([B_point0,B_point1], axis=2)
        # return img0, gt, img1
        return img0, gt, img1, B_point


    def __getitem__(self, index):
        # img0, gt, img1 = self.getimg(index)
        img0, gt, img1, B_point = self.getimg(index)

        if self.dataset_name == 'train':
            # random resize
            # if random.uniform(0, 1) < 0.1:
            #     img0 = cv2.resize(img0, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR)
            #     img1 = cv2.resize(img1, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR)
            #     gt = cv2.resize(gt, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR)
            #     flow_gt = cv2.resize(flow_gt, dsize=None, fx = 2.0, fy = 2.0,
            #             interpolation=cv2.INTER_LINEAR) * 2.0

            # random crop
            if self.has_crop_aug:
                img0, gt, img1, B_point = self.aug(img0, gt, img1, B_point, self.crop_h, self.crop_w)
            # random channel reverse
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            # random vertical flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                B_point = B_point[::-1]
                B_point = np.concatenate((B_point[:, :, 0:1], -B_point[:, :, 1:2],
                    B_point[:, :, 2:3], -B_point[:, :, 3:4]), 2)

            # random horizontal flip
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                B_point = B_point[:, ::-1]
                B_point = np.concatenate((-B_point[:, :, 0:1], B_point[:, :, 1:2],
                    -B_point[:, :, 2:3], B_point[:, :, 3:4]), 2)
            # random rotation
            if random.uniform(0, 1) < 0.5:
                rot_option = np.random.randint(1, 4)
                img0 = np.rot90(img0, rot_option)
                img1 = np.rot90(img1, rot_option)
                gt = np.rot90(gt, rot_option)
                B_point = np.rot90(B_point, rot_option)
                if rot_option == 1:
                    B_point = np.concatenate((B_point[:, :, 1:2], -B_point[:, :, 0:1],
                        B_point[:, :, 3:4], -B_point[:, :, 2:3]), 2)
                if rot_option == 2:
                    B_point = -B_point
                if rot_option == 3:
                    B_point = np.concatenate((-B_point[:, :, 1:2], B_point[:, :, 0:1],
                        -B_point[:, :, 3:4], B_point[:, :, 2:3]), 2)
            # random temporal reverse
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                B_point = np.concatenate((B_point[:, :, 2:4], B_point[:, :, 0:2]), 2)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        B_point = torch.from_numpy(B_point.copy()).permute(2,0,1)

        # flow1t = torch.from_numpy(flow1t.copy()).permute(2, 0, 1)
        # flowt0 = torch.from_numpy(flowt0.copy()).permute(2, 0, 1)
        # return torch.cat((img0, img1, gt), 0), flowt0, flow1t
        return torch.cat((img0, img1, gt, B_point), 0)


class VimeoSeptupletDataset(Dataset):
    """Vimeo90K-septuplet dataset for arbitrary-time interpolation.

    Each sample holds 7 frames.  A training item randomly picks three frames
    idx0 < idx_gt < idx1; the outer two are the inputs and the middle one is
    the target, so t = (idx_gt - idx0) / (idx1 - idx0) lies in (0, 1).
    This dataset is used in the first training stage, where the Bezier
    control points are derived from the ground-truth frame with RAFT (see
    Pipeline.compute_gt_control_points).
    """

    def __init__(self, dataset_name, data_root, crop_h=256, crop_w=256,
            has_aug=True):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.has_aug = has_aug
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.image_root = os.path.join(self.data_root, 'sequences')

        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        self.load_data()


    def __len__(self):
        return len(self.meta_data)


    def load_data(self):
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist


    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1


    def getimg(self, index):
        file_names = ['im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png',
                'im6.png', 'im7.png']
        imgpath = self.meta_data[index]
        img_dir = os.path.join(self.image_root, imgpath)
        # Interpolation: pick three random frames in temporal order; the
        # middle one is the target (t in (0, 1)).
        rand_idxs = np.random.choice(7, 3, replace=False)
        rand_idxs.sort()
        idx0, idx_gt, idx1 = rand_idxs[0], rand_idxs[1], rand_idxs[2]
        imgpaths = [
                os.path.join(img_dir, file_names[idx0]),
                os.path.join(img_dir, file_names[idx_gt]),
                os.path.join(img_dir, file_names[idx1])
                ]
        t = (idx_gt - idx0) / (idx1 - idx0)
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        return img0, gt, img1, t


    def __getitem__(self, index):
        img0, gt, img1, t = self.getimg(index)
        if self.dataset_name == 'train':
            img0, gt, img1 = self.aug(img0, gt, img1, self.crop_h, self.crop_w)
            if self.has_aug:
                # random channel reverse
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, :, ::-1]
                    img1 = img1[:, :, ::-1]
                    gt = gt[:, :, ::-1]
                # random vertical flip
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[::-1]
                    img1 = img1[::-1]
                    gt = gt[::-1]
                # random horizontal flip
                if random.uniform(0, 1) < 0.5:
                    img0 = img0[:, ::-1]
                    img1 = img1[:, ::-1]
                    gt = gt[:, ::-1]
                # random rotation
                if random.uniform(0, 1) < 0.5:
                    rot_option = np.random.randint(1, 4)
                    img0 = np.rot90(img0, rot_option)
                    img1 = np.rot90(img1, rot_option)
                    gt = np.rot90(gt, rot_option)
                # random temporal reverse
                if random.uniform(0, 1) < 0.5:
                    tmp = img1
                    img1 = img0
                    img0 = tmp
                    t = 1 - t

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        t = torch.tensor(np.array(t, dtype=np.float32)).reshape(1, 1, 1)
        return torch.cat((img0, img1, gt), 0), t


class VimeoSeptupletEvalDataset(Dataset):
    """Validation split of Vimeo90K-septuplet for arbitrary-time interpolation.

    The two outer frames (im1, im7) are the inputs; the five inner frames
    im2..im6 are the targets, giving t = 1/6 .. 5/6.
    """

    def __init__(self, dataset_name, data_root):
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'sep_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist
        else:
            self.meta_data = self.testlist

    def __getitem__(self, index):
        imgpath = self.meta_data[index]
        img_dir = os.path.join(self.image_root, imgpath)
        # Inputs im1 (t=0) and im7 (t=1); targets im2..im6 lie between them,
        # giving t = 1/6 .. 5/6 (interpolation).
        img0 = cv2.imread(os.path.join(img_dir, 'im1.png'))
        img1 = cv2.imread(os.path.join(img_dir, 'im7.png'))
        gt_indices = [2, 3, 4, 5, 6]
        time_values = [(i - 1) / 6.0 for i in gt_indices]
        gts = [cv2.imread(os.path.join(img_dir, 'im{}.png'.format(i)))
                for i in gt_indices]
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gts = torch.stack([
                torch.from_numpy(gt.copy()).permute(2, 0, 1)
                for gt in gts], dim=0)
        times = torch.tensor(np.array(time_values, dtype=np.float32))
        times = times.reshape(len(gt_indices), 1, 1, 1)
        return img0, img1, gts, times


class X_Test(Dataset):
    def make_2D_dataset_X_Test(test_data_path, multiple, t_step_size):
        """ make [I0,I1,It,t,scene_folder] """
        """ 1D (accumulated) """
        testPath = []
        t = np.linspace(
                (1 / multiple), (1 - (1 / multiple)), (multiple - 1)
                )
        for type_folder in sorted(glob(os.path.join(test_data_path, '*', ''))):  # [type1,type2,type3,...]
            for scene_folder in sorted(glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
                frame_folder = sorted(glob(scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
                for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                    if idx == len(frame_folder) - 1:
                        break
                    for mul in range(multiple - 1):
                        I0I1It_paths = []
                        I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                        I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                        I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])  # It
                        I0I1It_paths.append(t[mul])
                        I0I1It_paths.append(scene_folder.split(os.path.join(test_data_path, ''))[-1])  # type1/scene1
                        testPath.append(I0I1It_paths)
        return testPath


    def frames_loader_test(I0I1It_Path):
        frames = []
        for path in I0I1It_Path:
            frame = cv2.imread(path)
            frames.append(frame)
        (ih, iw, c) = frame.shape
        frames = np.stack(frames, axis=0)  # (T, H, W, 3)

        """ np2Tensor [-1,1] normalized """
        frames = X_Test.RGBframes_np2Tensor(frames)

        return frames


    def RGBframes_np2Tensor(imgIn, channel=3):
        ## input : T, H, W, C
        if channel == 1:
            # rgb --> Y (gray)
            imgIn = np.sum(
                    imgIn * np.reshape(
                        [65.481, 128.553, 24.966], [1, 1, 1, 3]
                        ) / 255.0,
                    axis=3,
                    keepdims=True) + 16.0

        # to Tensor
        ts = (3, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
        imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

        return imgIn

    def __init__(self, test_data_path, multiple):
        self.test_data_path = test_data_path
        self.multiple = multiple
        self.testPath = X_Test.make_2D_dataset_X_Test(
                self.test_data_path, multiple, t_step_size=32)

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " \
                    + self.test_data_path + "\n"))


    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]
        frames = X_Test.frames_loader_test(I0I1It_Path)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0),\
                scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


if  __name__ == "__main__":
    pass
