# A Markdown file for recording experiment results

Currently recording:

- NomResnet Thu's Checkpoint accuracy on SR_raw_crops

## Resnet Ckpt Experiments on SR crops

### Scale: 2

| Model         | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ------------- | ------------ | ----- | ----------- | ----------- | ------- |
| None-Baseline | Raw          |       | 0.9414      | 0.7649      | 0.7286  |
| Real-ESRGAN   | Raw          | 2     | 0.9114      | 0.7159      | 0.5429  |
| SwinIR        | Raw          | 2     |             |             |         |
| HAT           | Raw          | 2     |             |             |         |
| TSRN          | Raw          | 2     | 0.8695      | 0.6257      | 0.3178  |
| TBSRN         | Raw          | 2     | 0.9945      | 0.7665      | 0.7935  |
| TATT          |

### Scale: 4

| Model         | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ------------- | ------------ | ----- | ----------- | ----------- | ------- |
| None-Baseline | Raw          |       | 0.9414      | 0.7649      | 0.7286  |
| SRCNN         | Raw          | 4     | 0.9062      | 0.7156      | 0.5442  |
| EDSR          | Raw          | 4     | 0.8440      | 0.4609      | 0.6924  |
| SRResnet      | Raw          | 4     | 0.7826      | 0.4180      | 0.6871  |
| ESRGAN        | Raw          | 4     | 0.8981      | 0.5504      | 0.5429  |
| Real-ESRGAN   | Raw          | 4     | 0.9214      | 0.7151      | 0.6752  |
| SwinIR        | Raw          | 4     |             |             |         |
| HAT           | Raw          | 4     |             |             |         |

## SR_Resnet Ckpt Experiments on SR crops with mixed Backgrounds

### Scale: 2

| Model         | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ------------- | ------------ | ----- | ----------- | ----------- | ------- |
| None-Baseline | Raw          |       | 0.9442      | 0.7528      | 0.7270  |
| None          | SR           |       |             |             |         |
| Real-ESRGAN   | SR           | 2     |             |             |         |
| SwinIR        | SR           | 2     |             |             |         |
| HAT           | SR           | 2     |             |             |         |
| TSRN          | Raw          | 2     |             |             |         |
| TBSRN         | Raw          | 2     |             |             |         |

### Scale: 4

| Model       | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ----------- | ------------ | ----- | ----------- | ----------- | ------- |
| ESRGAN      | SR           | 4     |             |             |         |
| Real-ESRGAN | SR           | 4     |             |             |         |
| SwinIR      | SR           | 4     |             |             |         |
| HAT         | SR           | 4     |             |             |         |

# TO-DO thang 8

- MixedDomain
- Test SR on LR-HR dataset. HR - raw images, LR - bicubic of raw images
- Text focus loss

# NomNaOCR Experiments on retrained

### Scale: 2

| Model         | Type    | Scale | CNNxCTC_acc   | Transformer_SC_acc |
| ------------- | ------- | ----- | ------------- | ------------------ |
| None-Baseline | Retrain |       |               |
| Real-ESRGAN   | Raw     | 2     | 0.1325 0.7274 | 0.0926 0.7146
| SwinIR        | Raw     | 2     |
| HAT           | Raw     | 2     | 0.2284 0.8012 | 0.1931 0.8001      |
| ESRGAN        | Raw     | 4     | 0.1712 0.7270 |
| Real-ESRGAN   | Raw     | 4     |
| SwinIR        | Raw     | 4     |
| HAT           | Raw     | 4     | 0.1874 0.7495 | 0.1514 0.7556      |
| TSRN          | Raw     | 2     | 0.0825 0.3468 | 0.0708 0.3757         |
| TBSRN         | Raw     | 2     | 0.0723 0.5651 | 0.5775 0.5775         |
| SRCNN         | Raw     | 4     | 0.0370 0.2481 | 0.0277 0.3027      |
| EDSR          | Raw     | 4     | 0.1662 0.7251 |
| SRResnet      | Raw     | 4     | 0.1673 0.7236 |
