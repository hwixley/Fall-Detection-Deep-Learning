# Fall Detection Processing & Modelling

This was developed for my Undergraduate Thesis in AI & Computer Science at the University of Edinburgh. Please find the link to my report attached to this repository.

## Preprocessing
1. Compile json data recording chunks into their relevant recording objects
2. Parse these recording objects into sliding windows
3. Standardise the dataset
4. Randomly shuffle the dataset samples
5. Split the dataset into train/validation/test sets
6. Label the data by applying the sigmoid function to the average label in a sliding window (where 1 represents fall and 0 no fall)

## Modelling
I trained LSTM and ResNet deep learning models on my dataset using variable window sizes and tuning parameters. Overall ResNet152 proved to be the best performing model on a standardised, and shuffled dataset with a 2s window size which achieved 92.8% AUC, 87.28% sensitivity, and 98.33% specificity. 

![fall-detection-resnet-performance-graph](https://user-images.githubusercontent.com/57837950/233863694-4d9e1fd2-4c03-46a6-b7a0-1b9367f603e5.png)

![fall-detection-resnet-performance](https://user-images.githubusercontent.com/57837950/233863700-e62c4571-7845-45a8-928a-85bbb369f401.png)
![fall-detection-lstm-performance](https://user-images.githubusercontent.com/57837950/233863701-280cf48a-0691-458a-bdd0-d2071453dc0c.png)
![fall-detection-baseline-performance](https://user-images.githubusercontent.com/57837950/233863702-a6a15987-917e-499b-8ed5-203d0e36bafe.png)

## Exporting:
Exported my PyTorch model to `.tflite` using the following conversions: PyTorch -> ONNX -> TensorFlow -> TFLite

## Live ECG Data During a Fall
https://user-images.githubusercontent.com/57837950/234980065-a28f9c21-29b1-40d9-8c9e-2d8abbce4304.mp4
(The section between the green and red bars represents a fall)
