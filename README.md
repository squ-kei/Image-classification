

## A very very brief introduction
Used 6 different VGG settings and fully optimized RAM usage.
The dataset reading, transformation and model training only takes around 9.4 GB at most in total
with image dimension (224,224,3) and batch size 64.
Shrink image dimension or batch size or use fully convolutional layer setting will cost even less memory.
A few things learned about optimize memory
1) pre locate memory space to store data, don't use list.append
2) use short hand operator (+=,-=) instead of using assignment like a = a + b
3) when shuffling data, first get the final index beforehand and store it. Reduce the times trying to slice the orginal data
4) be careful about dtype. for exmaple, use float16 instead of float32 if it's possible. that will dramatically reduce memory usage
5) use tensorflow's dataset api will help read data progressively and save a lot of memory, but it has to read data again in every iteration which will reduce the training speed. So if it is possible use the cache() method.
