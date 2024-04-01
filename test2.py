#%%
from basicsr.utils.options import *
import sys

sys.argv = ['test2.py', '-opt', 'BasicSR/options/test/ESRGAN/test_ESRGAN_x4_woGT.yml']
root_path = 'C:\\Users\\Soppo\\Documents\\GitHub\\Thesis\\BasicSR'
opt, _ = parse_options(root_path, is_train=False)
opt['scale'] = 2
opt['network_g']['scale'] = 2

#%%
from basicsr.models import build_model
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import esrgan_model

# test_loaders = []
# for _, dataset_opt in sorted(opt['datasets'].items()):
#     test_set = build_dataset(dataset_opt)
#     test_loader = build_dataloader(
#         test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
#     # logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
#     test_loaders.append(test_loader)
# model = build_model(opt)

model = esrgan_model.ESRGANModel(opt)
print(model)
print(model.net_g)
#%%
import torch
import cv2
# Random tensor (3, 64, 64)
temp_image = cv2.imread('test.png')
temp = torch.from_numpy(temp_image).permute(2, 0, 1).unsqueeze(0).float()
print('Temp shape: ', temp.shape)
data = {'lq': temp}
#%%
# Model forward
model.feed_data(data)
model.test()
print(model.output.shape)