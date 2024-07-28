#%%
from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.data.data_util import get_label4image

@DATASET_REGISTRY.register()
class SingleTextImageDataset(data.Dataset):
    """Read only lq images and labels in the test phase, with text label.
    Used for testing recognition model accuracy on SR images.

    There is one mode:
    1. 'meta_info_file': Use meta information file to generate paths. Seperator: ' '

    Args:
         opt (dict): Config for train datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq.
            unicode_mapping_dict (str): Path to pickle file, stores unicode mapping file.
            labels_dict (dict): Dictionary for {img_name: label}.
            meta_info_file (str): Path for meta information file
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(SingleTextImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.lq_folder = opt['dataroot_lq']

        self.ucode_dict_path = opt['unicode_mapping_dict']
        self.labels_dict_path = opt['labels_dict']
        # Get labels
        self.label_dict = get_label4image((self.lq_folder, self.lq_folder), self.labels_dict_path, self.ucode_dict_path)

        self.paths = sorted(list(scandir(self.lq_folder, full_path=True)))


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load lq image
        lq_path = self.paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # load lq label
        label = self.label_dict[osp.basename(lq_path)]

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)



        return {'lq': img_lq, 'lq_path': lq_path, 'label': label}


#%%
if __name__ == '__main__':
    opt = dict(
        dataroot_lq='datasets/Nom-Test/LQ_bicx2',
        unicode_mapping_dict='datasets/Nom-Test/HWDB1.1-bitmap64-ucode-hannom-v2-tst_seen-label-set-ucode.pkl',
        labels_dict = 'datasets/Nom-Test/label.txt',
        io_backend=dict(type='disk'),
        scale = 2,
        use_hflip = True,
        use_rot = False,
    )
    dataset = SingleTextImageDataset(opt=opt)
    sample = dataset[5]

    img_lq = sample['lq']
    label = sample['label']

    from matplotlib import pyplot as plt
    plt.imshow(img_lq.permute(1, 2, 0).numpy())
    plt.show()
    print(label)
    print(chr(int(label, 16)))