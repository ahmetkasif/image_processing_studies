#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <random>
#include <chrono>
#include <cmath>

using std::cout;
using std::cin;
using std::endl;

using namespace std;
using namespace cv;

#define PI 3.14159265

void mean();
void sobelx();
void laplacian();

Mat fastFT(Mat img);
Mat iFFT(Mat img);
Mat shiftI(Mat img);
Mat getImageFourier(Mat img);
Mat getComplexFourier(Mat img);
void plotProfile(Mat img, string name);

Mat convertImage(Mat img);
Mat shiftTransform(Mat img);
Mat grayToBinary(Mat img);
Mat mergeImages(Mat imgList[], int rows, int cols);

void mean(){
  Mat meanExt = Mat::zeros(512, 512, CV_32F);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      meanExt.at<float>(i, j) = 1/9.0f;
    }
  }

  Mat fmeanExt = getImageFourier(meanExt);

  log(fmeanExt, fmeanExt);
  normalize(fmeanExt, fmeanExt, 0, 1, CV_MINMAX);
  imshow("mean filter abs", fmeanExt);

  plotProfile(fmeanExt, "mean filt profile");
}

void sobelx(){
  Mat sobelxExt = Mat::zeros(512, 512, CV_32F);

  sobelxExt.at<float>(0, 0) = -1.0f;
  sobelxExt.at<float>(0, 2) = 1.0f;
  sobelxExt.at<float>(1, 0) = -2.0f;
  sobelxExt.at<float>(1, 2) =  2.0f;
  sobelxExt.at<float>(2, 0) = -1.0f;
  sobelxExt.at<float>(2, 2) = 1.0f;

  Mat fsobelxExt = getImageFourier(sobelxExt);
  log(fsobelxExt, fsobelxExt);
  imshow("sobel filter abs", fsobelxExt);

  plotProfile(fsobelxExt, "sobel filt profile");
}

void laplacian(){
  Mat laplacianExt = Mat::zeros(512, 512, CV_32F);

  laplacianExt.at<float>(0, 1) = -1.0f;
  laplacianExt.at<float>(1, 0) = -1.0f;
  laplacianExt.at<float>(1, 1) =  4.0f;
  laplacianExt.at<float>(1, 2) = -1.0f;
  laplacianExt.at<float>(2, 1) = -1.0f;

  Mat flaplacianExt = getImageFourier(laplacianExt);
  log(flaplacianExt, flaplacianExt);
  imshow("laplacian filter abs", flaplacianExt);

  plotProfile(flaplacianExt, "laplacian filt profile");
}

Mat idealBP(Mat img, float r, int w){
  Mat filter = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_32F);
  Point centre = Point(img.rows / 2, img.cols / 2);
	double radius; // Radius to be calculated in each pos.
  Mat planes[2]; // Input planes
  Mat oplanes[2] = {Mat::zeros(img.size(), CV_32F),
    Mat::zeros(img.size(), CV_32F)};

  split(img, planes);

  for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
      radius=(double)sqrt(pow((i-centre.x),2.0)+pow((double)(j-centre.y),2.0));

      if (radius > r - w/2 && radius < r + w/2) {
        oplanes[0].at<float>(i, j) = 0.0f;
        oplanes[1].at<float>(i, j) = 0.0f;
      }else{
        oplanes[0].at<float>(i, j) = planes[0].at<float>(i, j);
        oplanes[1].at<float>(i, j) = planes[1].at<float>(i, j);
      }
		}
	}

  merge(oplanes, 2, oimg);

  return oimg;
}

Mat butterworthBP(Mat img, float r, int n, int w){
  Mat filter = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_32F);
  Point centre = Point(img.rows / 2, img.cols / 2);
	double radius; // Radius to be calculated in each pos.
  Mat planes[2]; // Input planes
  Mat oplanes[2] = {Mat::zeros(img.size(), CV_32F),
    Mat::zeros(img.size(), CV_32F)}; // Output planes

  split(img, planes);

  for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
      radius=(double)sqrt(pow((i-centre.x),2.0)+pow((double)(j-centre.y),2.0));

      oplanes[0].at<float>(i,j) = planes[0].at<float>(i, j) *
      (float)(1/(1 +
        pow((double)((r*w)/(r*r - radius * radius)), (double)(2 * n))));

      oplanes[1].at<float>(i,j) = planes[1].at<float>(i, j) *
      (float)(1/(1 +
        pow((double)((r*w)/(r*r - radius * radius)), (double)(2 * n))));
		}
	}

  merge(oplanes, 2, oimg);

  return oimg;
}

// FFT Fonksiyonu
Mat fastFT(Mat img){
  /*
    Take real part from img, add complex part
    as zero matrice and merge them, then take dft
  */

  Mat oimg;
  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
  merge(planes, 2, oimg);

  dft(oimg, oimg);

  return oimg;
}

Mat getImageFourier(Mat img){
  /*
    Function to show frequency space transform with shifted axis
    Splits complex and real parts of image
    and returns real part ready to be scaled logarithmically
    if requested
  */

  Mat oimg;
  Mat planes[2] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
  merge(planes, 2, oimg);

  dft(oimg, oimg);

  split(oimg, planes);
  magnitude(planes[0], planes[1], planes[0]);
  oimg = planes[0];

  oimg = shiftTransform(oimg);

  return oimg;
}

Mat getComplexFourier(Mat fimg){
  /*
    Function to show frequency space transform with shifted axis split complex and real parts of image
    and return real part's logarihtmic scale
  */

  Mat planes[2];

  split(fimg, planes);
  magnitude(planes[0], planes[1], planes[0]);
  Mat sfimg = planes[0];

  return sfimg;
}

// Function to shift and normalize dft image for better visualization
Mat shiftI(Mat img){
  Mat planes[2];
  split(img, planes);

  for (int i = 0; i < 2; i++) {
    // origin as center
    int cx = planes[i].cols/2;
    int cy = planes[i].rows/2;

    Mat tl(planes[i], Rect(0, 0, cx, cy)); // Top-Left
    Mat tr(planes[i], Rect(cx, 0, cx, cy)); // Top-Right
    Mat bl(planes[i], Rect(0, cy, cx, cy)); // Bottom-Left
    Mat br(planes[i], Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
    tl.copyTo(tmp);
    br.copyTo(tl);
    tmp.copyTo(br);

    tr.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
    bl.copyTo(tr);
    tmp.copyTo(bl);
  }

  merge(planes, 2, img);

  return img;
}

// Function to shift image's only real part
Mat shiftTransform(Mat img){
  // origin as center
  int cx = img.cols/2;
  int cy = img.rows/2;

  Mat tl(img, Rect(0, 0, cx, cy)); // Top-Left
  Mat tr(img, Rect(cx, 0, cx, cy)); // Top-Right
  Mat bl(img, Rect(0, cy, cx, cy)); // Bottom-Left
  Mat br(img, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp; // swap quadrants (Top-Left with Bottom-Right)
  tl.copyTo(tmp);
  br.copyTo(tl);
  tmp.copyTo(br);

  tr.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
  bl.copyTo(tr);
  tmp.copyTo(bl);

  return img;
}

Mat iFFT(Mat img){
  /*
    iFFT function - takes inverse fourier transform on input matrice
    and return normalized real part of processed matrice
  */

  Mat oimg = Mat::zeros(img.rows, img.cols, CV_32F);

  dft(img, oimg, DFT_INVERSE);
  Mat planes[2];
  split(oimg, planes);
  oimg = planes[0];

  normalize(oimg, oimg, 0, 1, CV_MINMAX);

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

void plotProfile(Mat img, string name){
  Mat px = Mat::zeros(512, 512, CV_32F);
  Mat py = Mat::zeros(512, 512, CV_32F);
  int temp = 0;

  for (int i = -px.rows / 2; i < px.rows / 2; i++) {
    temp = (int)(img.at<float>(i+px.cols / 2, 256)*256);
    if (temp > 0) {
      for (int j = 0; j < temp; j++) {
        px.at<float>(511 - j, i+px.rows / 2) = 1.0f;
      }
    }
  }

  for (int i = -py.rows / 2; i < py.rows / 2; i++) {
    temp = (int)(img.at<float>(256, i + img.cols / 2)*256);
    if (temp > 0) {
      for (int j = 0; j < temp; j++) {
        py.at<float>(511 - j, i+py.rows / 2) = 1.0f;
      }
    }
  }

  imshow(name + "x", px);
  imshow(name + "y", py);
}

Mat periodicNoise(Mat img){
  float max = 0;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      float temp = 20*sin(2 * PI * i / 8) * sin(2 * PI * j / 8);

      if (img.at<uchar>(i, j) + temp < 0) {
        img.at<uchar>(i, j) = 0.0f;
      } else if(img.at<uchar>(i, j) + temp > 255){
        img.at<uchar>(i, j) = 255.0f;
      }else {
        img.at<uchar>(i, j) += temp;
      }
    }
  }

  return img;
}

int main(int argc, char** argv ){
  // Load Image
  Mat img = imread("a.jpeg", 0);
  imshow("Grayscale Image", img);

  img = periodicNoise(img);

  imshow("Periodic Noised Image", img);


  Mat fimg = fastFT(img);
  Mat sfimg = shiftI(fimg);

  Mat sfC = getComplexFourier(sfimg);
  sfC += Scalar::all(1);
  log(sfC, sfC);
  normalize(sfC, sfC, 0, 1, CV_MINMAX);
  imshow("Fourier of Periodic Noised Image", sfC);

  /*
    Q1A: mean filter analysis
  */

  mean();

  /*
    Q1B: sobelx filter analysis
  */

  sobelx();

  /*
    Q1C: laplacian filter analysis
  */

  laplacian();

  /*
    Q2: Adding periodic noise to image, then removing it with ideal
    and butterworth band pass filters in frequency domain
  */

  // Ideal Filter Noise Removal

  Mat iCl = idealBP(idealBP(idealBP(sfimg, 91, 4), 129, 4), 180, 4);
  Mat siCl = shiftI(iCl);
  iCl = iFFT(siCl);
  imshow("ideal band passed image", iCl);

  Mat inoise = Mat::zeros(img.rows, img.cols, CV_8UC1);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      inoise.at<uchar>(i, j) = img.at<uchar>(i, j) -  255.0f * iCl.at<float>(i, j);
    }
  }

  imshow("Ideal Noise", inoise);

  Mat clf = getImageFourier(iCl);
  clf += Scalar::all(1);
  log(clf, clf);
  normalize(clf, clf, 0, 1, CV_MINMAX);
  imshow("ideal bp image fourier", clf);

  // Butterworth Filter Noise Removal

  Mat bCl = butterworthBP(butterworthBP(butterworthBP(sfimg, 91, 1, 6), 129, 1, 4), 180, 1, 4);
  Mat sbCl = shiftI(bCl);
  bCl = iFFT(sbCl);
  imshow("butterworth bp image", bCl);

  Mat bnoise = Mat::zeros(img.rows, img.cols, CV_8UC1);
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      bnoise.at<uchar>(i, j) = img.at<uchar>(i, j) -  255.0f * bCl.at<float>(i, j);
    }
  }

  imshow("Butterworth Noise", bnoise);

  Mat clf2 = getImageFourier(bCl);
  clf2 += Scalar::all(1);
  log(clf2, clf2);
  normalize(clf2, clf2, 0, 1, CV_MINMAX);
  imshow("butterworth bp image fourier", clf2);

  waitKey();
  return 0;
}
