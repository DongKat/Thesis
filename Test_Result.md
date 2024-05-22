# A Markdown file for recording experiment results

Currently recording:

- NomResnet Thu's Checkpoint accuracy on SR_raw_crops

## SR_Resnet Ckpt Experiments for NomDatasetCrop

| Model                      | ResNet_train | ToK1871_Acc | ToK1902_Acc | LVT_Acc | ToK1902_PSNR | ToK1902_SSIM | LVT_PSNR | LVT_SSIM |
| -------------------------- | ------------ | ----------- | ----------- | ------- | ------------ | ------------ | -------- | -------- |
| None-Baseline              | HR           | 0.9442      | 0.7528      | 0.7270  | N/A          | N/A          | N/A      | N/A      |
| ESRGAN                     | HR           |             | 0.7251      | 0.4463  | 13.330       | 0.343        | 14.926   | 0.175    |
| ESRGAN_Nom                 | HR           |             |             |         | 17.615       | 0.617        | 20.447   | 0.496    |
| Real-ESRGAN                | HR           |             | 0.3778      | 0.4401  |              |              |          |          |
| Real-ESRGAN_x2_RealCE      | HR           |             | 0.5438      | 0.6928  |              |              |          |          |
| Real-ESRGAN_x2_ToK1871     | HR           |             | 0.5515      | 0.6432  |              |              |          |          |
| Real-ESRGAN_x2_ToK1871_USM | HR           |             | 0.4753      | 0.5472  |              |              |          |          |

## SR_Resnet Ckpt Experiments for NomDatasetCrop, with SR crops

| SR_Model                               | ResNet_train | ToK1871-Acc | ToK1902-Acc | LVT_Acc | CASIA_HWDB_Acc | ToK1902_PSNR | ToK1902_SSIM | LVT_PSNR | LVT_SSIM |
| -------------------------------------- | ------------ | ----------- | ----------- | ------- | -------------- | ------------ | ------------ | -------- | -------- |
| None-Baseline                          | HR           | 0.9442      | 0.7628      | 0.7270  | 0.9350         | N/A          | N/A          | N/A      |          |
| Real-ESRGAN_x2                         | SR           |             |             |         |                |              |              |          |          |
| Real-ESRGAN_x2_wMixedBackground        | SR           | 0.9956      | 0.7719      | 0.7342  | 0.8327         |              |              |          |          |
| Real-ESRGAN_x2_RealCE                  | SR           | 0.9907      | 0.7808      | 0.7121  |                |              |              |          |          |
| Real-ESRGAN_x2_RealCE_wMixedBackground | SR           | 0.9960      | 0.7980      | 0.7390  |                |              |              |          |          |

## SR_Resnet Ckpt Experiments for NomDatasetYolo

| Model           | ResNet | ToK1871_Acc | ToK1902_Acc | LVT_Acc | ToK1902_PSNR | ToK1902_SSIM | LVT_PSNR | LVT_SSIM |
| --------------- | ------ | ----------- | ----------- | ------- | ------------ | ------------ | -------- | -------- |
| None            | HR     | 98%         | 75%         | 72%     | N/A          | N/A          | N/A      | N/A      |
| ESRGAN          | HR     |             |             |         |              |              |          |          |
| ESRGAN_Nom      | HR     |             |             |         |              |              |          |          |
| Real-ESRGAN     | HR     |             |             |         |              |              |          |          |
| Real-ESRGAN_Nom | HR     |             |             |         |              |              |          |          |

## Resnet Exp / ToK1871

| Yolo_crops_224_norm | Raw_crops_224_norm | Yolo_crops_256_norm | Raw_crops_256_norm | Yolo_crops_64_norm | Raw_crops_64_norm |
| ------------------- | ------------------ | ------------------- | ------------------ | ------------------ | ----------------- |
| 0.7575              | 0.7601             | 0.7287              | 0.7359             | 0.0003             | 0.0003            |

## Resnet Exp / ToK1902

- Apprently NomDatasetCrop gives wrong label, that's why the acc is messed up

### With ImageNet mean, std Normalization

| Yolo_crops_224_norm | Raw_crops_224_norm | Yolo_crops_256_norm | Raw_crops_256_norm | Yolo_crops_64_norm | Raw_crops_64_norm |
| ------------------- | ------------------ | ------------------- | ------------------ | ------------------ | ----------------- |
| 0.7575              | 0.7601             | 0.7287              | 0.7359             | 0.0003             | 0.0003            |

### Without ImageNet mean, std Normalization

| Yolo_crops_224 | Raw_crops_224 | Yolo_crops_256 | Raw_crops_256 | Yolo_crops_64 | Raw_crops_64 |
| -------------- | ------------- | -------------- | ------------- | ------------- | ------------ |
| 0.7209         | 0.7174        |                |               |               |              |

# Planning

- Finetune Ckpt with SR
- Pretrained Resnet101 with dataset SR, raw

- HWDB 1.1, Dai Viet Su Ky
