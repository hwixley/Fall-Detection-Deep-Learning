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

## Exporting:
Exported my PyTorch model to `.tflite` using the following conversions: PyTorch -> ONNX -> TensorFlow -> TFLite
