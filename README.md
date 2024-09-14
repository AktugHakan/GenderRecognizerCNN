# Gender Recognizer  
Machine learning - Image processing project that recognizes the gender of the person from *face* image. Implented in TensorFlow on Python

## Dataset
To train and test my model I used the [Gender Classification Dataset by Ashutosh Chauhan](https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset)  

![Example images from dataset](example_images/1.jpg)
![Example images from dataset](example_images/2.jpg)
![Example images from dataset](example_images/3.jpg)
![Example images from dataset](example_images/4.jpg)
---
![Example images from dataset](example_images/5.jpg)
![Example images from dataset](example_images/6.jpg)
![Example images from dataset](example_images/7.jpg)
![Example images from dataset](example_images/8.jpg)

To use the model with another dataset, images should be seperated in two folders (male - female)  

## Caching
When model imports the data, it saves the data as `<hash>.npy` file in the `.cache` folder and it will be imported to skip the image processing part and speed up the program.  

After the model has been trained, the state of the model will be saved in `model.keras` file to avoid unnecessary training.