#%%

from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.data.data_util import get_label4image


@DATASET_REGISTRY.register()
class PairedTextImageDataset(data.Dataset):
    """Read only lq images in the test phase, with text label
    Note: For Scene Text Super Resolution task. Label stores in meta_info_file. Needs a unicode mapping file to convert label to tensor.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc).

    There is one mode:
    1. 'meta_info_file': Use meta information file to generate paths. Seperator: ' '

    Args:
         opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            unicode_mapping_dict (str): Path to pickle file, stores unicode mapping file.
            labels_dict (dict): Dictionary for {img_name: label}.
            meta_info_file (str): Path for meta information file
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(PairedTextImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.scale = opt['scale'] if 'scale' in opt else 1

        self.ucode_dict_path = opt['unicode_mapping_dict']
        self.labels_dict_path = opt['labels_dict_path']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        # Get labels
        self.label_dict = get_label4image((self.lq_folder, self.gt_folder), self.labels_dict_path, self.ucode_dict_path)

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)


        # load lq image
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # load gt image
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # load label
        label = self.label_dict[osp.basename(gt_path)]

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # Removed because text image dont need to be crop
            # # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])


        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'label': label}


#%%
if __name__ == '__main__':
    opt = dict(
        dataroot_gt='datasets/Nom-Test/GT_mod4',
        dataroot_lq='datasets/Nom-Test/LQ_bicx2',
        unicode_mapping_dict='datasets/Nom-Test/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl',
        labels_dict_path = 'datasets/Nom-Test/label.txt',
        io_backend=dict(type='disk'),
        phase = 'train',
        gt_size = 256,
        scale = 2,
        use_hflip = True,
        use_rot = False,
    )

    dataset = PairedTextImageDataset(opt=opt)

    sample = dataset[0]
    print(sample.keys())
    img_lq = sample['lq'].permute(1, 2, 0).numpy()
    img_gt = sample['gt'].permute(1, 2, 0).numpy()
    label = sample['label']

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(img_lq)
    plt.subplot(1, 2, 2)
    plt.imshow(img_gt)
    plt.show()

    print(label)
    print(chr(int(label, 16)))