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

void mse(Mat img1, Mat img2, string name);
Mat fastFT(Mat img);
Mat iFFT(Mat img);
Mat shiftI(Mat img);
Mat getImageFourier(Mat img);
Mat getComplexFourier(Mat img);

Mat convertImage(Mat img);
Mat shiftTransform(Mat img);
Mat mergeImages(Mat imgList[], int rows, int cols);

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
  magnitude(planes[0], planes[1], planes[0]);
  oimg = planes[0];

  normalize(oimg, oimg, 0, 1, CV_MINMAX);

  return oimg;
}

Mat inverseTurbulence(Mat img, float r, double k){
  Mat filter = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_32F);
  Point center = Point(img.rows / 2, img.cols / 2);
	double radius;
  Mat planes[2];

  split(img, planes);

  for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
      radius = sqrt(pow(i - center.x, 2) + pow(j - center.y, 2));
      float noise = 1 / exp(-k*(pow(i - center.x, 2) + pow(j - center.y, 2)));

      if (radius <= r) {
        planes[0].at<float>(i, j) *= noise;
        planes[1].at<float>(i, j) *= noise;
      } else {
        planes[0].at<float>(i, j) = 0;
        planes[1].at<float>(i, j) = 0;
      }
		}
	}

  merge(planes, 2, oimg);

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

Mat addNoise(Mat img, double k){
  Mat oimg;
  Point center = Point(img.rows / 2, img.cols / 2);

  Mat planes[2];
  split(img, planes);

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      float noise = exp(-k * (pow(i - center.x, 2) + pow(j - center.y, 2)));

      planes[0].at<float>(i, j) *= noise;
      planes[1].at<float>(i, j) *= noise;
    }
  }

  merge(planes, 2, oimg);

  return oimg;
}

Mat meanGauss(Mat img, int mean, int stddev, float size){
  // Applying mean filter
  Mat meanImg = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimgSF = Mat::zeros(img.rows, img.cols, CV_32F);

  for (int i = 0; i < (int)size; i++) {
    for (int j = 0; j < (int)size; j++) {
      meanImg.at<float>(i, j) = 1 / (size * size);
    }
  }

  Mat meanSF = shiftI(fastFT(meanImg));

  Mat planesF[2];
  split(meanSF, planesF);

  Mat planesI[2];
  split(img, planesI);

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      planesI[0].at<float>(i, j) *= planesF[0].at<float>(i, j);
      planesI[1].at<float>(i, j) *= planesF[0].at<float>(i, j);
    }
  }

  merge(planesI, 2, oimgSF);

  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);
  oimg = iFFT(shiftI(oimgSF));
  imshow("Noised Image", oimg);
  mse(iFFT(img), oimg, "mean gauss noised image's mse");

  // Q2:1 Restoring without using an epsilon value

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      if (planesF[0].at<float>(i, j) != 0) {
        planesI[0].at<float>(i, j) /= planesF[0].at<float>(i, j);
        planesI[1].at<float>(i, j) /= planesF[0].at<float>(i, j);
      }
    }
  }

  merge(planesI, 2, oimgSF);

  Mat resimg = Mat::zeros(img.rows, img.cols, CV_8UC1);
  resimg = iFFT(shiftI(oimgSF));
  imshow("Restored Image", resimg);
  mse(iFFT(img), resimg, "mean gauss noised and restored image's mse");

  // Q2:2 Restoring using an epsilon value

  // Noise image again

  split(img, planesI);

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      planesI[0].at<float>(i, j) *= planesF[0].at<float>(i, j);
      planesI[1].at<float>(i, j) *= planesF[0].at<float>(i, j);
    }
  }

  merge(planesI, 2, oimgSF);

  oimg = iFFT(shiftI(oimgSF));

  double epsilon = 0.0000000001;

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      if (planesF[0].at<float>(i, j) != 0) {
        if (abs(planesF[0].at<float>(i, j)) < epsilon) {
          planesI[0].at<float>(i, j) /= 0.1f;
          planesI[1].at<float>(i, j) /= 0.1f;
        } else {
          planesI[0].at<float>(i, j) /= planesF[0].at<float>(i, j);
          planesI[1].at<float>(i, j) /= planesF[0].at<float>(i, j);
        }
      }
    }
  }

  merge(planesI, 2, oimgSF);

  Mat resimgwithe = Mat::zeros(img.rows, img.cols, CV_8UC1);
  resimgwithe = iFFT(shiftI(oimgSF));
  imshow("Restored Image With Epsilon", resimgwithe);
  mse(iFFT(img), resimgwithe, "mean gauss noised and epsilon restored image's mse");

  return oimg;
}

Mat restoreMean(Mat img, float size){
  // Applying mean filter
  Mat meanImg = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimgSF = Mat::zeros(img.rows, img.cols, CV_32F);

  for (int i = 0; i < (int)size; i++) {
    for (int j = 0; j < (int)size; j++) {
      meanImg.at<float>(i, j) = 1 / (size * size);
    }
  }

  Mat meanSF = shiftI(fastFT(meanImg));

  Mat planesF[2];
  split(meanSF, planesF);

  Mat planesI[2];
  split(img, planesI);

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      if (planesF[0].at<float>(i, j) != 0) {
        planesI[0].at<float>(i, j) /= planesF[0].at<float>(i, j);
        planesI[1].at<float>(i, j) /= planesF[0].at<float>(i, j);
      }
    }
  }

  merge(planesI, 2, oimgSF);

  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);
  oimg = iFFT(shiftI(oimgSF));

  return oimg;
}

void wiener(Mat img, float size){
  double k = 0.000000001;

  // Applying mean filter
  Mat meanImg = Mat::zeros(img.rows, img.cols, CV_32F);
  Mat oimgSF = Mat::zeros(img.rows, img.cols, CV_32F);

  for (int i = 0; i < (int)size; i++) {
    for (int j = 0; j < (int)size; j++) {
      meanImg.at<float>(i, j) = 1 / (size * size);
    }
  }

  Mat meanSF = shiftI(fastFT(meanImg));
  Mat planesF[2];
  split(meanSF, planesF);

  Mat planesI[2];
  split(img, planesI);

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      planesI[0].at<float>(i, j) *= planesF[0].at<float>(i, j);
      planesI[1].at<float>(i, j) *= planesF[0].at<float>(i, j);
    }
  }

  merge(planesI, 2, oimgSF);

  // Q3 Restoring using wiener

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      if (planesF[0].at<float>(i, j) != 0 && planesF[1].at<float>(i, j) != 0) {
        planesI[0].at<float>(i, j) *= -planesF[1].at<float>(i, j) /
        (abs(planesF[0].at<float>(i, j)) * abs(planesF[1].at<float>(i, j)) * k);
        
        planesI[1].at<float>(i, j) *= -planesF[1].at<float>(i, j) /
        (abs(planesF[0].at<float>(i, j)) * abs(planesF[1].at<float>(i, j)) * k);
      }
    }
  }

  merge(planesI, 2, oimgSF);

  Mat resimg = Mat::zeros(img.rows, img.cols, CV_8UC1);
  resimg = iFFT(shiftI(oimgSF));
  imshow("Restored Image", resimg);
  mse(iFFT(img), resimg, "mean gauss noised and restored image's mse");
}

void mse(Mat img1, Mat img2, string name){
  int mse = 0;
  for (int i = 0; i < img1.rows; i++) {
    for (int j = 0; j < img1.cols; j++) {
      mse += pow(img1.at<uchar>(i, j) - img2.at<uchar>(i, j), 2);
    }
  }

  cout << name << ": " << sqrt(mse) << endl;
}

int main(int argc, char** argv ){
  Mat img = imread("aerial.tif", 0);
  imshow("Grayscale Image", img);

  Mat fimg = fastFT(img);
  Mat sfimg = shiftI(fimg);

  /*
    Q1: Add turbulence noise to an image, then acquire R(u, v) finally restore image
  */

  // Turbulence images
/*  Mat lowTF = addNoise(sfimg, 0.00025);

  Mat lowT = iFFT(lowTF);
  imshow("Low Turbulence", lowT);

  // MSE Calculation of Images
  mse(img, lowT, "Low");

  // Restoration of high turbulenced image

  Mat resahighTF = inverseTurbulence(lowTF, 240, 0.00025);
  Mat resahighT = iFFT(resahighTF);
  normalize(resahighT, resahighT, 0, 1, CV_MINMAX);
  imshow("Res. High Turb. with 240 Cut Off Freq.", resahighT);

  mse(img, resahighT, "Restored HTa");

  Mat resbhighTF = inverseTurbulence(lowTF, 40, 0.00025);
  Mat resbhighT = iFFT(resbhighTF);
  imshow("Res. High Turb. with 40 Cut Off Freq.", resbhighT);

  mse(img, resbhighT, "Restored HTb");

  Mat reschighTF = inverseTurbulence(lowTF, 87, 0.00025);
  Mat reschighT = iFFT(reschighTF);
  imshow("Res. High Turb. with 87 Cut Off Freq.", reschighT);

  mse(img, reschighT, "Restored HTc");*/

  /*
    Q2: Add gaussian noise and apply blurring to an image, then acquire R(u, v) finally restore image
  */

  // Mat mgn = meanGauss(sfimg, 0, 1, 7.0f);

  /*
    Q3: Add gaussian noise and apply blurring to an image, then apply Wiener filter to restore image
  */

  wiener(sfimg, 7);

  waitKey();
  return 0;
}
