# A Markdown file for recording experiment results

Currently recording:

- NomResnet Thu's Checkpoint accuracy on SR_raw_crops

## Resnet Ckpt Experiments on SR crops

### Scale: 2

| Model         | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ------------- | ------------ | ----- | ----------- | ----------- | ------- |
| None-Baseline | Raw          |       | 0.9414      | 0.7649      | 0.7286  |
| Real-ESRGAN   | Raw          | 2     | 0.9114      | 0.7159      |         |
| SwinIR        | Raw          | 2     |             |             |         |
| HAT           | Raw          | 2     |             |             |         |
| TSRN          | Raw          | 2     | 0.8695      | 0.6257      | 0.3178  |
| TBSRN         | Raw          | 2     | 0.9945      | 0.7665      | 0.7935  |
| TATT          |

### Scale: 4

| Model       | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ----------- | ------------ | ----- | ----------- | ----------- | ------- |
| EDSR_M      | Raw          | 4     | 0.8440      | 0.4609      | 0.6924  |
| EDSR_L      | Raw          | 4     | N/A         | N/A         | N/A     |
| SRCNN       | Raw          | 4     | 0.9216      |             |         |
| SRResnet    | Raw          | 4     | 0.0003      | 0.4180      | 0.6871        |
| ESRGAN      | Raw          | 4     | 0.8981      | 0.5504      |         |
| Real-ESRGAN | Raw          | 4     | 0.9214      | 0.7151      | 0.6752  |
| SwinIR      | Raw          | 4     |             |             |         |
| HAT         | Raw          | 4     |             |             |         |

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
