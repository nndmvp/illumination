import pywt
import torch
import torchvision.transforms
import glob
from torch.utils.data.dataset import Dataset
from data.util import *
from torchvision.transforms import functional as F
from pytorch_wavelets import DWTForward, DWTInverse


def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

def to_quaternion(L):
    """
    将 LL 子带（PyTorch 张量）转换为四元数形式（实部为 0，虚部分别为 R、G、B）
    输入: LL, 形状为 [batch_size, 3, H, W]
    输出: quaternion_LL, 形状为 [batch_size, 4, H, W]
    """
    _, h, w = L.shape
    # 创建实部为 0 的四元数张量
    quaternion_L = torch.zeros(4, h, w, device=L.device)
    # 虚部分别为 R、G、B
    quaternion_L[1:, :, :] = L
    return quaternion_L

def rgb_to_wavelet(rgb_img, j, random_index, device='cpu'):
    # 将 numpy 数组转换为 PyTorch 张量，并调整维度以适应 DWTForward 的输入要求
    rgb_img = np.array(rgb_img)
    i = torch.tensor(rgb_img, dtype=torch.float32).to(device).permute(2, 0, 1) / 255.0  # 调整维度到 [B, C, H, W]
    i = to_quaternion(i)

    return i


class FusionDataset(Dataset):
    def __init__(self,
                 split,
                 crop_size=128,  # resolution in training
                 min_max=(-1, 1),
                 ir_path='./PathToIr/',
                 vi_path='./PathToVis/',
                 sourse_path='./data/data/sourse',
                 is_crop=True):
        super(FusionDataset, self).__init__()
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        self.is_crop = is_crop
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor=1)
        self.hr_transform = train_hr_transform(crop_size)
        self.vis_ir_transform = train_vis_ir_transform()  # transform from rgb to grayscale
        self.min_max = min_max
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.1)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.1)

        self.new_size_test = (480, 480)

        if split == 'train':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            data_dir_so = sourse_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_so, self.filenames_so = prepare_data_path(data_dir_so)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'val':
            data_dir_vis = vi_path
            data_dir_ir = ir_path
            data_dir_so = sourse_path
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_so, self.filenames_so = prepare_data_path(data_dir_so)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split == 'train':
            visible_image = Image.open(self.filepath_vis[index])
            infrared_image = Image.open(self.filepath_ir[index])
            so_image = Image.open(self.filepath_so[index])
            if self.is_crop:
                crop_size = self.hr_transform(visible_image)
                visible_image, infrared_image, so_image = F.crop(visible_image, crop_size[0], crop_size[1],
                                                                 crop_size[2],
                                                                 crop_size[3]), \
                    F.crop(infrared_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3]), \
                    F.crop(so_image, crop_size[0], crop_size[1], crop_size[2], crop_size[3])
            # Random horizontal flipping
            if random.random() > 0.5:
                visible_image = self.hflip(visible_image)
                infrared_image = self.hflip(infrared_image)
                so_image = self.hflip(so_image)
            # Random vertical flipping
            if random.random() > 0.5:
                visible_image = self.vflip(visible_image)
                infrared_image = self.vflip(infrared_image)
                so_image = self.vflip(so_image)
            j = 1
            random_index = random.randint(0, 0)

            visible_image = rgb_to_wavelet(visible_image, j, random_index)
            infrared_image = rgb_to_wavelet(infrared_image, j, random_index)
            so_image = rgb_to_wavelet(so_image, j, random_index)

            cat_img = torch.cat([visible_image, infrared_image, so_image], axis=0)
            x = torch.max(cat_img)
            y = torch.min(cat_img)

            return {'img': cat_img, 'vis': visible_image, 'ir': infrared_image, 'so': so_image}, self.filenames_vis[
                index]

        elif self.split == 'val':
            visible_image = Image.open(self.filepath_vis[index])
            infrared_image = Image.open(self.filepath_ir[index])
            so_image = Image.open(self.filepath_so[index])
            #
            visible_image = ToTensor()(visible_image)
            visible_image = visible_image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]

            infrared_image = ToTensor()(infrared_image)
            infrared_image = infrared_image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]
            so_image = ToTensor()(so_image)
            so_image = so_image * (self.min_max[1] - self.min_max[0]) + self.min_max[0]

            cat_img = torch.cat([visible_image, infrared_image, so_image], axis=0)

            return {'img': cat_img, 'vis': visible_image, 'ir': infrared_image, 'so': so_image}, self.filenames_vis[
                index]

    def __len__(self):
        return self.length
