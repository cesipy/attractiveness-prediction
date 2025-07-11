# Results




## evaluation on my dataset 

pretrained resnet18, finetuned on scut for 30 epochs with the following params, also without extracting images first: 
```python
FC_DIM_SIZE = 1024
DROPOUT_PROB = 0.5
OUTFEATURES = 1
LR = 4e-5
EPOCHS = 30
BATCH_SIZE = 32

```

| #  | Image                         | Predicted Score |
|----|------------------------------|-----------------|
| 1  | res/test/photo_1.jpg         | 6.065 / 10      |
| 2  | res/test/photo_2.jpg         | 5.284 / 10      |
| 3  | res/test/photo_3.jpg         | 5.834 / 10      |
| 4  | res/test/photo_4.jpg         | 6.138 / 10      |
| 5  | res/test/photo_5.jpg         | 5.682 / 10      |
| 6  | res/test/photo_6.jpg         | 6.539 / 10      |
| 7  | res/test/photo_7.jpg         | 5.658 / 10      |
| 8  | res/test/photo_8.jpg         | 5.070 / 10      |
| 9  | res/test/photo_9.jpg         | 6.612 / 10      |
| 10 | res/test/photo_10.jpg        | 7.345 / 10      |
| 11 | res/test/photo_11.jpg        | 5.842 / 10      |
| 12 | res/test/photo_12.jpg        | 5.643 / 10      |
| 13 | res/test/photo_13.jpg        | 7.636 / 10      |
| 14 | res/test/photo_14.jpg        | 5.811 / 10      |
| 15 | res/test/photo_15.jpg        | 6.297 / 10      |
| 16 | res/test/photo_16.jpg        | 6.023 / 10      |
| 17 | res/test/photo_17.jpg        | 6.588 / 10      |
| 18 | res/test/photo_18.jpg        | 6.202 / 10      |
| 19 | res/test/photo_19.jpg        | 6.348 / 10      |
| 20 | res/test/photo_20.jpg        | 7.473 / 10      |
| 21 | res/test/photo_21.jpg        | 6.379 / 10      |
| 22 | res/test/photo_22.jpg        | 5.610 / 10      |
| 23 | res/test/photo_23.jpg        | 6.282 / 10      |
| 24 | res/test/photo_24.jpg        | 6.029 / 10      |
| 25 | res/test/photo_25.jpg        | 7.083 / 10      |
| 26 | res/test/photo_26.jpg        | 6.977 / 10      |
| 27 | res/test/photo_27.jpg        | 7.307 / 10      |
| 28 | res/test/photo_28.jpg        | 6.766 / 10      |
| 29 | res/test/photo_29.jpg        | 5.354 / 10      |
| 30 | res/test/photo_30.jpg        | 5.483 / 10      |
| 31 | res/test/photo_31.jpg        | 5.221 / 10      |
| 32 | res/test/photo_32.jpg        | 4.990 / 10      |
| 33 | res/test/photo_33.jpg        | 6.312 / 10      |
| 34 | res/test/photo_34.jpg        | 6.643 / 10      |


Here's the cropped face detection results formatted in the same table style:

## Face-Cropped Results

| #  | Image                         | Predicted Score | Notes |
|----|------------------------------|-----------------|-------|
| 1  | res/test/photo_1.jpg         | 5.690 / 10      |       |
| 2  | res/test/photo_2.jpg         | 5.728 / 10      |       |
| 3  | res/test/photo_3.jpg         | 4.935 / 10      |       |
| 4  | res/test/photo_4.jpg         | 6.209 / 10      |       |
| 5  | res/test/photo_5.jpg         | 5.316 / 10      |       |
| 6  | res/test/photo_6.jpg         | No face detected | Skipped |
| 7  | res/test/photo_7.jpg         | 5.339 / 10      |       |
| 8  | res/test/photo_8.jpg         | 5.539 / 10      |       |
| 9  | res/test/photo_9.jpg         | 6.941 / 10      |       |
| 10 | res/test/photo_10.jpg        | 7.588 / 10      |       |
| 11 | res/test/photo_11.jpg        | 7.011 / 10      |       |
| 12 | res/test/photo_12.jpg        | 6.818 / 10      |       |
| 13 | res/test/photo_13.jpg        | 7.213 / 10      |       |
| 14 | res/test/photo_14.jpg        | 5.494 / 10      |       |
| 15 | res/test/photo_15.jpg        | 5.863 / 10      |       |
| 16 | res/test/photo_16.jpg        | 5.933 / 10      |       |
| 17 | res/test/photo_17.jpg        | 5.178 / 10      |       |
| 18 | res/test/photo_18.jpg        | 5.692 / 10      |       |
| 19 | res/test/photo_19.jpg        | 6.800 / 10      |       |
| 20 | res/test/photo_20.jpg        | 7.647 / 10      |       |
| 21 | res/test/photo_21.jpg        | 7.076 / 10      |       |
| 22 | res/test/photo_22.jpg        | 4.681 / 10      |       |
| 23 | res/test/photo_23.jpg        | 4.836 / 10      |       |
| 24 | res/test/photo_24.jpg        | 6.280 / 10      |       |
| 25 | res/test/photo_25.jpg        | 6.941 / 10      |       |
| 26 | res/test/photo_26.jpg        | 6.921 / 10      |       |
| 27 | res/test/photo_27.jpg        | 6.348 / 10      |       |
| 28 | res/test/photo_28.jpg        | 7.110 / 10      |       |
| 29 | res/test/photo_29.jpg        | No face detected | Skipped |
| 30 | res/test/photo_30.jpg        | 5.735 / 10      |       |
| 31 | res/test/photo_31.jpg        | 5.964 / 10      |       |
| 32 | res/test/photo_32.jpg        | 4.010 / 10      |       |
| 33 | res/test/photo_33.jpg        | 6.992 / 10      |       |
| 34 | res/test/photo_34.jpg        | 5.362 / 10      |       |

| Image | Original Score | Face-Cropped Score | Difference |
|-------|---------------|-------------------|-------------|
| res/test/photo_1.jpg | 6.065 | 5.690 | -0.375 |
| res/test/photo_2.jpg | 5.284 | 5.728 | +0.444 |
| res/test/photo_3.jpg | 5.834 | 4.935 | -0.899 |
| res/test/photo_4.jpg | 6.138 | 6.209 | +0.071 |
| res/test/photo_5.jpg | 5.682 | 5.316 | -0.366 |
| res/test/photo_6.jpg | 6.539 | No face | N/A |
| res/test/photo_7.jpg | 5.658 | 5.339 | -0.319 |
| res/test/photo_8.jpg | 5.070 | 5.539 | +0.469 |
| res/test/photo_9.jpg | 6.612 | 6.941 | +0.329 |
| res/test/photo_10.jpg | 7.345 | 7.588 | +0.243 |
| res/test/photo_11.jpg | 5.842 | 7.011 | +1.169 |
| res/test/photo_12.jpg | 5.643 | 6.818 | +1.175 |
| res/test/photo_13.jpg | 7.636 | 7.213 | -0.423 |
| res/test/photo_14.jpg | 5.811 | 5.494 | -0.317 |
| res/test/photo_15.jpg | 6.297 | 5.863 | -0.434 |
| res/test/photo_16.jpg | 6.023 | 5.933 | -0.090 |
| res/test/photo_17.jpg | 6.588 | 5.178 | -1.410 |
| res/test/photo_18.jpg | 6.202 | 5.692 | -0.510 |
| res/test/photo_19.jpg | 6.348 | 6.800 | +0.452 |
| res/test/photo_20.jpg | 7.473 | 7.647 | +0.174 |
| res/test/photo_21.jpg | 6.379 | 7.076 | +0.697 |
| res/test/photo_22.jpg | 5.610 | 4.681 | -0.929 |
| res/test/photo_23.jpg | 6.282 | 4.836 | -1.446 |
| res/test/photo_24.jpg | 6.029 | 6.280 | +0.251 |
| res/test/photo_25.jpg | 7.083 | 6.941 | -0.142 |
| res/test/photo_26.jpg | 6.977 | 6.921 | -0.056 |
| res/test/photo_27.jpg | 7.307 | 6.348 | -0.959 |
| res/test/photo_28.jpg | 6.766 | 7.110 | +0.344 |
| res/test/photo_29.jpg | 5.354 | No face | N/A |
| res/test/photo_30.jpg | 5.483 | 5.735 | +0.252 |
| res/test/photo_31.jpg | 5.221 | 5.964 | +0.743 |
| res/test/photo_32.jpg | 4.990 | 4.010 | -0.980 |
| res/test/photo_33.jpg | 6.312 | 6.992 | +0.680 |
| res/test/photo_34.jpg | 6.643 | 5.362 | -1.281 |