# MIA
A membership inference attack on Resnet18

## Logit classifier per class

44 SVM's are trained(on logits) for each class in the pub.pt. During inference, first a classifier is picked according to the label of the sample. Then the selected model gives a logit for membership.

TPR@0.05FPR = 0.059

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

These values have high variance, due to the randomness in training shadow models.

## About the dataset

Eventhough the underlying distribution is unknown, we can extract some information from pub.pt

Below are 20 random member samples

![image](https://github.com/OshanJayawardana/MIA/blob/main/figures/members.png)

Below are 20 random non-member samples

![image](https://github.com/OshanJayawardana/MIA/blob/main/figures/non_members.png)

There are no visible differences between members and non-members.
But one important thing to notice is that there are different types of classes varying from blood cell images to animal images.

Below are 5 images per class

![image](https://github.com/OshanJayawardana/MIA/blob/main/figures/class_images.png)

Below is a pca plot of the pub.pt

![image](https://github.com/OshanJayawardana/MIA/blob/main/figures/pca.png)

You can observe a symmetry in the PCA
