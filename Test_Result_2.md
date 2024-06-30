# A Markdown file for recording experiment results

Currently recording:

- NomResnet Thu's Checkpoint accuracy on SR_raw_crops

## Resnet Ckpt Experiments on SR crops

### Scale: 2

| Model         | ResNet_train | Scale | ToK1871_Acc  | ToK1902_Acc  | LVT_Acc      |
| ------------- | ------------ | ----- | ------------ | ------------ | ------------ |
| None-Baseline | Raw          |       | 0.9414       | 0.7649       | 0.7286       |
| Real-ESRGAN   | Raw          | 2     | 0.9114       | 0.7159       | 0.5429       |
| SwinIR        | Raw          | 2     | 0.9152       | 0.6750       | 0.6324       |
| HAT           | Raw          | 2     | **0.9339**   | **0.7601**   | **0.7247**   |
| TSRN          | Raw          | 2     | 0.8695       | 0.6257       | 0.3178       |
| TBSRN         | Raw          | 2     | **_0.9945_** | **_0.7665_** | **_0.7935_** |
| SRCNN         | Raw          | 4     | 0.9062       | 0.7156       | 0.5442       |
| EDSR          | Raw          | 4     | 0.8440       | 0.4609       | 0.6924       |
| SRResnet      | Raw          | 4     | 0.7826       | 0.4180       | 0.6871       |
| ESRGAN        | Raw          | 4     | 0.8981       | 0.5504       | 0.5429       |
| Real-ESRGAN   | Raw          | 4     | 0.9214       | 0.7151       | 0.6752       |
| SwinIR        | Raw          | 4     | 0.8029       | 0.3402       | 0.5031       |
| HAT           | Raw          | 4     | **0.9178**   | **0.7558**   | **0.7229**   |

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

| Model         | Type | Scale | CNNxCTC_acc       | Transformer_SC_acc |
| ------------- | ---- | ----- | ----------------- | ------------------ |
| None-Baseline |      |       | **0.2942 0.8473** | **0.2714 0.8490**  |
| Real-ESRGAN   | Raw  | 2     | 0.1325 0.7274     | 0.0926 0.7146      |
| SwinIR        | Raw  | 2     | 0.1930 0.7714     | 0.1594 0.7776      |
| HAT           | Raw  | 2     | **0.2284 0.8012** | **0.1931 0.8001**  |
| TSRN          | Raw  | 2     | 0.0825 0.3468     | 0.0708 0.3757      |
| TBSRN         | Raw  | 2     | 0.0723 0.5651     | 0.5775 0.5775      |
| ESRGAN        | Raw  | 4     | 0.1712 0.7270     | 0.1374 0.7317      |
| Real-ESRGAN   | Raw  | 4     | 0.0892 0.6172     | 0.0557 0.6398      |
| SwinIR        | Raw  | 4     | 0.1235 0.6906     | 0.0779 0.6876      |
| HAT           | Raw  | 4     | **0.1874 0.7495** | **0.1514 0.7556**  |
| SRCNN         | Raw  | 4     | 0.0370 0.2481     | 0.0277 0.3027      |
| EDSR          | Raw  | 4     | 0.1662 0.7251     | 0.1355 0.7308      |
| SRResnet      | Raw  | 4     | 0.1673 0.7236     | 0.1366 0.7292      |
