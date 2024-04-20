# A Markdown file for recording experiment results

Currently recording:

- NomResnet Thu's Checkpoint accuracy on SR_raw_crops

## SR_ResnetCkpt Experiments

| Model           | ResNet_train | ToK1871_Acc_Train | ToK1902_Acc | LVT_Acc | ToK1902_PSNR | ToK1902_SSIM | LVT_PSNR | LVT_SSIM |
| --------------- | ------------ | ----------------- | ----------- | ------- | ------------ | ------------ | -------- | -------- |
| None            | HR           | 98%               | 75%         | 72%     | N/A          | N/A          | N/A      | N/A      |
| ESRGAN          | SR           |                   |             |         | 13.330       | 0.343        | 14.926   | 0.175    |
| ESRGAN_Nom      | SR           |                   |             |         | 17.615       | 0.617        | 20.447   | 0.496    |
| Real-ESRGAN     | SR           |                   |             |         |              |              |          |          |
| Real-ESRGAN_Nom | SR           |                   |             |         |              |              |          |          |
| HAT             | SR           |                   |             |         |              |              |          |          |
| HAT_Nom         | SR           |                   |             |         |              |              |          |          |

## Resnet Exp / ToK1902

- Apprently NomDatasetCrop gives wrong label, that's why the acc is messed up

### With ImageNet mean, std Normalization

| Yolo_crops_224_norm | Raw_crops_224_norm | Yolo_crops_256_norm | Raw_crops_256_norm | Yolo_crops_64_norm | Raw_crops_64_norm |
| ------------------- | ------------------ | ------------------- | ------------------ | ------------------ | ----------------- |
| 0.7575              | 0.7601             | 0.7287              | 0.7359             | 0.0003             | 0.0003            |

### Without ImageNet mean, std Normalization

| Yolo_crops_224 | Raw_crops_224 | Yolo_crops_256 | Raw_crops_256 | Yolo_crops_64 | Raw_crops_64 |
| -------------- | ------------- | -------------- | ------------- | ------------- | ------------ |
| 0.758          | 0.760         |                |               | 0.0003        | 0.0003       |
