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

Mat idealFilter(Mat img, float r);
Mat butterworthFilter(Mat img, float r, int n);
Mat gaussianFilter(Mat img, int d);

Mat fastFT(Mat img);
Mat iFFT(Mat img);
Mat shiftI(Mat img);
Mat showFourier(Mat img);

Mat convertImage(Mat img);

Mat idealFilter(Mat img, float r){
  Mat filter = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_32F);
  Point centre = Point(img.rows / 2, img.cols / 2);
	double radius; // Radius to be calculated in each pos.
  Mat planes[2]; // Input planes
  Mat oplanes[] = {Mat::zeros(img.size(), CV_32F),
    Mat::zeros(img.size(), CV_32F)};; // Output planes

  split(img, planes);

  for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
      radius=(double)sqrt(pow((i-centre.x),2.0)+pow((double)(j-centre.y),2.0));

      if (radius <= r) {
        oplanes[0].at<float>(i, j) = planes[0].at<float>(i, j);
        oplanes[1].at<float>(i, j) = planes[1].at<float>(i, j);
      }
		}
	}

  merge(oplanes, 2, oimg);

  return oimg;
}

Mat butterworthFilter(Mat img, float r, int n){
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
      (float)(1 / (1 + pow((double) (radius / r), (double) (2 * n))));

      oplanes[1].at<float>(i,j) = planes[1].at<float>(i, j) *
      (float)(1 / (1 + pow((double) (radius / r), (double) (2 * n))));
		}
	}

  merge(oplanes, 2, oimg);

  return oimg;
}

Mat gaussianFilter(Mat img, int d){
  Mat filter = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_32F);
  Point centre = Point(img.rows / 2, img.cols / 2);
	double radius; // Radius to be calculated in each pos.
  Mat planes[2]; // Input planes
  Mat oplanes[] = {Mat::zeros(img.size(), CV_32F),
    Mat::zeros(img.size(), CV_32F)};; // Output planes

  split(img, planes);

  for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
      radius=(double)sqrt(pow((i-centre.x),2.0)+pow((double)(j-centre.y),2.0));
      oplanes[0].at<float>(i,j) = planes[0].at<float>(i, j) *
      exp((-radius*radius) / (2*d*d));

      oplanes[1].at<float>(i,j) = planes[1].at<float>(i, j) *
      exp((-radius*radius) / (2*d*d));
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

Mat showFourier(Mat img){
  /*
    Function to show frequency space transform with shifted axis split complex and real parts of image
    and return real part's logarihtmic scale
  */

  Mat oimg;
  Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
  merge(planes, 2, oimg);

  dft(oimg, oimg);

  oimg = shiftI(oimg);

  split(oimg, planes);
  magnitude(planes[0], planes[1], planes[0]);
  oimg = planes[0];

  log(oimg, oimg);
  normalize(oimg, oimg, 0, 1, CV_MINMAX);

  return oimg;
}

// Function to shift and normalize dft image for better visualization
Mat shiftI(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, img.type());
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

  merge(planes, 2, oimg);

  return oimg;
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

int main(int argc, char** argv ){

  // Load Image
  Mat cam = imread("cameraman.png", 1 );
  Mat gcam = convertImage(cam);

  imshow("Cameraman", gcam);
  imshow("gCam Freq. Image", showFourier(gcam));

  Mat gcamdft = fastFT(gcam);
  Mat scamdft = shiftI(gcamdft);

  /*
    Q1: Low pass ideal filter for r = size/6 & size/2
  */

  Mat ifcamdft = idealFilter(scamdft, gcam.rows/6);
  Mat isfcamdft = shiftI(ifcamdft);
  Mat q11 = iFFT(isfcamdft);
  imshow("if1", q11);

  Mat ifcamdft2 = idealFilter(scamdft, gcam.rows/2);
  Mat isfcamdft2 = shiftI(ifcamdft2);
  Mat q12 = iFFT(isfcamdft2);
  imshow("if2", q12);

  /*
    Q2: Low pass butterworth filter for r = size/3 & size*2/3 & d0 = 1,2,3
  */

  Mat bfcamdft = butterworthFilter(scamdft, gcam.rows/6, 1);
  Mat bsfcamdft = shiftI(bfcamdft);
  Mat q21 = iFFT(bsfcamdft);
  imshow("bf1", q21);

  Mat bfcamdft2 = butterworthFilter(scamdft, gcam.rows/2, 1);
  Mat bsfcamdft2 = shiftI(bfcamdft2);
  Mat q22 = iFFT(bsfcamdft2);
  imshow("bf2", q22);

  Mat bfcamdft3 = butterworthFilter(scamdft, gcam.rows/6, 2);
  Mat bsfcamdft3 = shiftI(bfcamdft3);
  Mat q23 = iFFT(bsfcamdft3);
  imshow("bf3", q23);

  Mat bfcamdft4 = butterworthFilter(scamdft, gcam.rows/2, 2);
  Mat bsfcamdft4 = shiftI(bfcamdft4);
  Mat q24 = iFFT(bsfcamdft4);
  imshow("bf4", q24);

  Mat bfcamdft5 = butterworthFilter(scamdft, gcam.rows/6, 3);
  Mat bsfcamdft5 = shiftI(bfcamdft5);
  Mat q25 = iFFT(bsfcamdft5);
  imshow("bf5", q25);

  Mat bfcamdft6 = butterworthFilter(scamdft, gcam.rows/2, 3);
  Mat bsfcamdft6 = shiftI(bfcamdft6);
  Mat q26 = iFFT(bsfcamdft6);
  imshow("bf6", q26);

  /*
    Q3: Low pass gaussian filter for r = size/3 & size*2/3 & l = 10, 20, 40
  */

  Mat gfcamdft = gaussianFilter(scamdft, 10);
  Mat gsfcamdft = shiftI(gfcamdft);
  Mat q31 = iFFT(gsfcamdft);
  imshow("gf1", q31);

  Mat gfcamdft3 = gaussianFilter(scamdft, 20);
  Mat gsfcamdft3 = shiftI(gfcamdft3);
  Mat q33 = iFFT(gsfcamdft3);
  imshow("gf3", q33);

  Mat gfcamdft5 = gaussianFilter(scamdft, 40);
  Mat gsfcamdft5 = shiftI(gfcamdft5);
  Mat q35 = iFFT(gsfcamdft5);
  imshow("gf5", q35);

  /*
    Q4: Mean mask filtering in frequency domain
  */

  Mat mask = Mat::zeros(gcam.size(), CV_32F);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      mask.at<float>(i, j) = 1/9.0;
    }
  }

  imshow("Q5: Mask", mask);

  Mat maskdft = fastFT(mask);
  Mat smaskdft = shiftI(maskdft);
  Mat q5;
  mulSpectrums(scamdft, smaskdft, q5, 1);
  smaskdft = shiftI(q5);
  maskdft = iFFT(smaskdft);

  imshow("Mask Filtered", maskdft);

  waitKey();
  return 0;
}
