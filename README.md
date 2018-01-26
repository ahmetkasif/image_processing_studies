# image-processing-cpp
Low level digital image processing studies using c++ and opencv bindings 

Week 1

- Load image to memory.
- Find minimum values at all channels.
- Find maximum values at all channels.
- Find average values at all channels.
- Show original image.

Week 2

- Convert RGB image to Grayscale, render grayscale image.
- Slice grayscale image's planes and render each of them.
- Reduce resolution of grayscale image 3 times by factor of 4 and render each of them.

Week 3

- Equalize histogram of a grayscale image, if rgb, then convert to grayscale first.
- Apply gamma conversions to image, ranging from 0.5 to 2.0. 
Then render images along with their histograms.
- Slice image in to its bit planes then render each of them.

Week 4

- Apply histogram matching to an image. Render both original and outputted image.
- Apply gamma correction with values of 0.5, 1 and 1.5 to a grayscale image. 
Then apply histogram equalization. Also render images along with their histograms.
- Apply local histogram equalization to an image with mask size of 3x3.
- Apply mean fiter to an image by a mask size of 3x3, 5x5 and 7x7.

Week 5

- Local histogram equalization in the size of 4x4, 8x8 ve 16x16
- Addition of Salt & Pepper noise to image, than applying mean
and median filters to see which one works better
- Addition of Gaussian noise to image, than applying mean
and median filters to see which one works better
- Standard Deviation filtering on input image

Week 6

- Getting rid of gaussian noise present in an image with sampling of
1, 5, 10, 20, 50, 100 for standard deviation values of both 10 and 20.
- Applying laplacian sharpening filter on an image along with its mean filter applied copies.
For mean filter mask size 3x3, 5x5 and 7x7.

Week 7

- Gradient filtering, along with roberts and sobel variations.
- Converting an 8bit image to 1bit image using mean value thresholding.
- Transfering an image via fourier transform to frequency space and using
inverse fourier transform to convert image back to pixel space.

Week 8

- Low pass Ideal filter for r = size/6 & size/2
- Low pass Butterworth filter for r = size/6 & size/2 & d0 = 1,2,3
- Low pass Gaussian filter for d0 = 10, 20, 40
- Mean mask filtering in frequency domain

Week 9

Will be updated..

Week 10

Will be updated..

Week 11

Will be updated..

Final Project

Will be updated..
