#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <random>
#include <chrono>

using std::cout;
using std::cin;
using std::endl;

using namespace std;
using namespace cv;

Mat gradientFilter(Mat img);
Mat gradientAngle(Mat img);
Mat robertsGF(Mat img);
Mat sobelGF(Mat img);
Mat mergeImages(Mat imgList[], int rows, int cols);
Mat convertImage(Mat img);
Mat cbPattern(int size);
Mat grayToBinary(Mat img);
Mat fastFT(Mat img);
Mat iFFT(Mat img);
Mat shiftTransform(Mat img);
Mat showComplexI(Mat img);

// FFT Fonksiyonu
Mat fastFT(Mat img){
  Mat oimg = Mat::zeros(img.size(), CV_64F);

  /*
  Take real part from img, add complex part
  as zero matrice and merge them in complexI
  */

  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
  Mat complexI;
  merge(planes, 2, complexI);

  dft(complexI, complexI);

  return complexI;
}

/*
  Function to split complex and real parts of image
  and return real part's logarihtmic scale
*/
Mat showComplexI(Mat img){
  Mat planes[2];
  split(img, planes);
  Mat oimg = planes[0];

  return oimg;
}

// Function to shift and normalize dft image for better visualization
Mat shiftTransform(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, img.type());

  // crop the spectrum for odd sizes
  oimg = img(Rect(0, 0, img.cols & -2, img.rows & -2));

  // origin as center
  int cx = oimg.cols/2;
  int cy = oimg.rows/2;

  Mat tl(oimg, Rect(0, 0, cx, cy)); // Top-Left
  Mat tr(oimg, Rect(cx, 0, cx, cy)); // Top-Right
  Mat bl(oimg, Rect(0, cy, cx, cy)); // Bottom-Left
  Mat br(oimg, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
  tl.copyTo(tmp);
  br.copyTo(tl);
  tmp.copyTo(br);

  tr.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
  bl.copyTo(tr);
  tmp.copyTo(bl);
  log(oimg, oimg);
  normalize(oimg, oimg, 0, 1, CV_MINMAX);

  return oimg;
}

/*
  iFFT function - takes inverse fourier transform on input matrice
  and return normalized real part of processed matrice
*/
Mat iFFT(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_32F);

  dft(img, oimg, DFT_INVERSE);

  Mat planes[2];
  split(oimg, planes);
  Mat magI = planes[0];

  normalize(magI, magI, 0, 1, CV_MINMAX);

  return magI;
}

// Gradient Function
Mat gradientFilter(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_64F);
  Mat tgx, tgy;

  Mat gx = (Mat_<float>(1,2) << -1, 1);
  Mat gy = (Mat_<float>(2,1) << -1, 1);

  // Filtering and getting magnitude in 64bit space
  filter2D(img, tgx, CV_64F, gx);
  filter2D(img, tgy, CV_64F, gy);

  magnitude(tgx, tgy, oimg);

  // Converting input back to 8bit space
  oimg.convertTo(oimg, CV_8U);

  return oimg;
}

Mat gradientAngle(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8U);
  Mat tgx, tgy;
  float temp = 0.0f;
  float max = 0.0f;
  float scalar = 180.0f / 3.14159265f;

  Mat gx = (Mat_<float>(1,2) << -1, 1);
  Mat gy = (Mat_<float>(2,1) << -1, 1);

  // Filtering and getting magnitude in 64bit space
  filter2D(img, tgx, CV_64F, gx);
  filter2D(img, tgy, CV_64F, gy);

  // Angle of the gradient on the (i, j) position: arctan((tgx(i, j) / tgy(i, j)) * 180 / PI)
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (tgy.at<uchar>(i, j) != 0) {
        oimg.at<uchar>(i, j) = atan(tgx.at<uchar>(i, j) / tgy.at<uchar>(i, j)) * scalar;
        if (oimg.at<uchar>(i, j) > max) {
          max = oimg.at<uchar>(i, j);
        }
      }
    }
  }

  // Contrast streching for gradient angles
  oimg = oimg / max * 255.0f;

  return oimg;
}

// Roberts Gradient Function
Mat robertsGF(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_64FC1);
  Mat tgx, tgy;

  Mat gx = (Mat_<float>(2,2) << -1, 0, 0, 1);
  Mat gy = (Mat_<float>(2,2) << 0, -1, 1, 0);

  // Filtering and getting magnitude in 64bit space
  filter2D(img, tgx, CV_64F, gx);
  filter2D(img, tgy, CV_64F, gy);

  magnitude(tgx, tgy, oimg);

  // Converting input back to 8bit space
  oimg.convertTo(oimg, CV_8U);

  return oimg;
}

// Sobel Gradient Function
Mat sobelGF(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_64FC1);
  Mat tgx, tgy;

  Mat gx = (Mat_<float>(3,3) <<
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1);

  Mat gy = (Mat_<float>(3,3) <<
  -1, -2, -1,
  0, 0, 0,
  1, 2, 1);

  // Filtering and getting magnitude in 64bit space
  filter2D(img, tgx, CV_64F, gx);
  filter2D(img, tgy, CV_64F, gy);

  magnitude(tgx, tgy, oimg);

  // Converting input back to 8bit space
  oimg.convertTo(oimg, CV_8U);

  return oimg;
}

// Grayscale to Binary conversion function
Mat grayToBinary(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, img.type());

  // Calculating average of the image
  float avg;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      avg += img.at<uchar>(i,j);
    }
  }
  avg /= img.rows * img.cols;

  // Applying average threshold
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<uchar>(i,j) > avg) {
        oimg.at<uchar>(i,j) = 255.0f;
      } else {
        oimg.at<uchar>(i,j) = 0.0f;
      }
    }
  }

  return oimg;
}

// Chessboard pattern
Mat cbPattern(int size){
  Mat oimg = Mat(size, size, CV_8U);
  unsigned char color=0;

  for (int i = 0; i < size; i+=size/8) {
    color=~color;

    for(int j=0; j<size; j+=size/8){
      Mat ROI = oimg(Rect(i, j, size/8, size/8));
      ROI.setTo(Scalar::all(color));
      color=~color;
    }
  }

  return oimg;
}

// Function to merge several images as 1 image
Mat mergeImages(Mat imgList[], int rows, int cols){
  int size = 256; // Standart resim ebatı
  Mat oimg = Mat(size * rows, size * cols, CV_8U);

  for (int i = 0; i < rows * cols; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        oimg.at<uchar>((256 * (i/cols)) + j, (256*(i%cols)) + k) = imgList[i].at<uchar>(j, k);
      }
    }
  }

  return oimg;
}

// RGB2Grayscale conversion function
Mat convertImage(Mat img){
  Mat oimg = Mat(img.rows, img.cols, CV_8UC1);

  Mat c[3];
  split(img,c);

  /*
    Inputted 3 channel image is splitted and each channel is
    multiplied with corresponding scalar to get grayscale version of image.
  */

  c[0] *= 0.299;
  c[1] *= 0.587;
  c[2] *= 0.114;

  oimg = c[0] + c[1] + c[2];

  return oimg;
}

int main(int argc, char** argv ){

  // Load Images
  Mat glena = imread("lena.jpg", 0 );

  Mat gcat = imread("building.tif", 0);

  Mat gcam = imread("cameraman.png", 0 );

  // Q1: Gradient Magnitude & Angel Outputs
  Mat gfList[6];

  gfList[0] = glena;
  gfList[1] = gradientFilter(gfList[0]);
  gfList[2] = gradientAngle(gfList[0]);
  gfList[3] = cbPattern(256);
  gfList[4] = gradientFilter(gfList[3]);
  gfList[5] = gradientAngle(gfList[3]);

  Mat q1 = mergeImages(gfList, 2, 3);
  imwrite("q1.png", q1);
  imshow("Image, Gradient Mag, Gradient Angle", q1);

  // Q2: Gradient-Roberts-Sobel Magnitude Outputs
  Mat rsFList[8];

  rsFList[0] = glena;
  rsFList[1] = gradientFilter(rsFList[0]);
  rsFList[2] = robertsGF(rsFList[0]);
  rsFList[3] = sobelGF(rsFList[0]);

  rsFList[4] = cbPattern(256);
  rsFList[5] = gradientFilter(rsFList[4]);
  rsFList[6] = robertsGF(rsFList[4]);
  rsFList[7] = sobelGF(rsFList[4]);

  Mat q2 = mergeImages(rsFList, 2, 4);
  imwrite("q2.png", q2);
  imshow("Image, GradientI, RobertsI, SobelI", q2);

  // Q3: Converting grayScale Roberts & Sobel Outputs to binary
  Mat gtb[4];

  gtb[0] = rsFList[1];
  gtb[1] = grayToBinary(gtb[0]);
  gtb[2] = rsFList[2];
  gtb[3] = grayToBinary(gtb[2]);

  Mat q3 = mergeImages(gtb, 2, 2);
  imwrite("q3.png", q3);
  imshow("q3", q3);

  /*
    Q4: Apply FFT to an image, then apply FFT Shift to render. Apply
    iFFT to return to pixel space and render it again
  */

  imshow("lena2", glena);
  Mat lenadft = shiftTransform(showComplexI(fastFT(glena)));
  imshow("lenadft", lenadft);
  Mat lenaidft = iFFT(fastFT(glena));
  imshow("lenaidft", lenaidft);

  imshow("gc1", gcat);
  Mat catdft = shiftTransform(showComplexI(fastFT(gcat)));
  imshow("catdft", catdft);
  Mat catidft = iFFT(fastFT(gcat));
  imshow("catidft", catidft);

  imshow("cam1", gcam);
  Mat camdft = shiftTransform(showComplexI(fastFT(gcam)));
  imshow("camdft", camdft);
  Mat camidft = iFFT(fastFT(gcam));
  imshow("camidft", camidft);

  // Q5: Synthetic Rect Shape FFT
  Mat sr = Mat::zeros(256, 256, CV_8UC1);
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      if (i < 160 && i > 96 && j > 96 && j < 160) {
        sr.at<uchar>(i, j) = 255.0f;
      }
    }
  }

  imshow("Synthetic Rectangle", sr);
  Mat srdft = shiftTransform(showComplexI(fastFT(sr)));
  imshow("Synt Rect FFT", srdft);
  Mat sridft = iFFT(fastFT(sr));
  imshow("Synt Rect IDFT", sridft);

  waitKey();
  return 0;
}
