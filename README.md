# MIA
A membership inference attack on Resnet18

## RMIA Offline

Initially experiments are done to select best `a` value for the offline attack.
For the `a` experiments we fix gamma=2, k(num_reference_models)=8, |Z| = 500

| a Value | TPR@0.05FPR |
|---------|-------------|
| 0.1     | 0.0507      |
| 0.2     | 0.052       |
| 0.3     |             |
| 0.4     |             |
| 0.5     | 0.054       |
| 1.0     | 0.044       |