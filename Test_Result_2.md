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
| TSRN          | Raw          | 2     |             |             | 0.3178  |

### Scale: 4
| Model       | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ----------- | ------------ | ----- | ----------- | ----------- | ------- |
| ESRGAN      | Raw          | 4     | 0.8981      | 0.5504      |         |
| Real-ESRGAN | Raw          | 4     | 0.9214      | 0.7151      | 0.6752  |
| SwinIR      | Raw          | 4     |             |             |         |
| HAT         | Raw          | 4     |             |             |         |

## SR_Resnet Ckpt Experiments on SR crops
### Scale: 2
| Model         | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ------------- | ------------ | ----- | ----------- | ----------- | ------- |
| None-Baseline | Raw          |       | 0.9442      | 0.7528      | 0.7270  |
| Real-ESRGAN   | SR           | 2     |             |             |         |
| SwinIR        | SR           | 2     |             |             |         |
| HAT           | SR           | 2     |             |             |         |

### Scale: 4
| Model       | ResNet_train | Scale | ToK1871_Acc | ToK1902_Acc | LVT_Acc |
| ----------- | ------------ | ----- | ----------- | ----------- | ------- |
| ESRGAN      | SR           | 4     |             |             |         |
| Real-ESRGAN | SR           | 4     |             |             |         |
| SwinIR      | SR           | 4     |             |             |         |
| HAT         | SR           | 4     |             |             |         |
