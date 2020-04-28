# Distracted-Driver-Detection

## A very very brief introduction
This is a project related to the state farm distracted driver detection dataset. Used 6 different VGG settings
and fully optimized RAM usage.
The dataset reading, transformation and model training only takes around 9.4 GB at most in total
with image dimension (224,224,3) and batch size 64.
Shrink image dimension or batch size or use fully convolutional layer setting will cost even less memory.
