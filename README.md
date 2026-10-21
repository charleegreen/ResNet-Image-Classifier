# ResNet-34 Image Classifier

This is a project for my CS421 Neural Networks class at Western Colorado University. I built a ResNet-34 style model from scratch in TensorFlow/Keras to classify pictures of celebrities.

## What it does

The model takes a 224x224 image and predicts which celebrity is in it. I built all of the layers myself instead of using a pretrained model.

## How I built it

- I started with a 7x7 convolution layer and a max pooling layer.
- After that, there are 4 groups of residual blocks. Each block has two convolution layers and a skip connection that adds the input back to the output.
- Each new group makes the image smaller and uses more filters (64, then 128, 256 and 512).
- When the image size changes, the skip connection doesn't fit anymore, so I used a 1x1 convolution to make the shapes match.
- At the end, there's a global average pooling layer and a softmax layer that makes the prediction.

## Problems I ran into

The dataset was pretty small, so the model started overfitting. To help with that, I used:

- **Data augmentation:** randomly flipping, rotating and zooming the training images
- **Early stopping:** stopping training when accuracy stopped getting better
- **Learning rate scheduling:** lowering the learning rate when the model got stuck

Getting the shapes to line up for the skip connections was also tricky.

## Files

- `resnet34_starter.py` builds and trains the model, then saves it.
- `resnet34_test.py` loads the saved model and tests it on the test images.

## How to run it

You need Python and TensorFlow:

```
pip install tensorflow numpy
```

Change the dataset paths in both files to your own folder, then run:

```
python resnet34_starter.py
python resnet34_test.py
```

## What I learned

- How skip connections help deep networks train
- How to keep track of the image size through each layer
- Some ways to deal with overfitting when you don't have a lot of data
