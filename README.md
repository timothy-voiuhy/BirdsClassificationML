## BirdsClassificationML 
In this project i implement the code for training a computer vision ai model both in pytorch and tensorflow for classifing 525 species of birds.

## Architecture
I have used the MobileNetv2 as the backbone of the model since it is really designed for machines with low system requirements including emebedded devices, mobile phones and not leaving my pc :)
I then added 2 dense layers to adjust the ouput to suit my classes(525)

## Perfomance
I have sofar obtained a 88% train accuracy 93% validation accuracy and 90% test accuracy in the 10 first epochs. I have however really used a very large batch_size of 2048 for the train dataset since i have been using google colabs 346GB TPU 

I hope to get something like 97% validation accuracy in another 10 epochs of training.

## Dataset
The dataset is from kaggle.com titled: birds_525_species_image_classification

## Deploying.
As an additional i have designed a simple server application that directly connects to my cpp Qt gui for using the models i make.
But it completely adjustable for any use.