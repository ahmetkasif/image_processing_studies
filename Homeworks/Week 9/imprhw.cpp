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

void convolution(Mat img, Mat kernel);
void correlation(Mat img, Mat kernel);

Mat fastFT(Mat img);
Mat iFFT(Mat img);
Mat shiftI(Mat img);
Mat showFourier(Mat img);

Mat convertImage(Mat img);
Mat grayToBinary(Mat img);
Mat mergeImages(Mat imgList[], int rows, int cols);

void convolution(Mat img, Mat kernel){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);
  Point centre = Point(kernel.rows / 2, kernel.cols / 2);
  Mat kernelRt;

  // Creating a rotation matrix for the center of filter image
  Mat rotM = getRotationMatrix2D(centre, 180, 1);
  // Apply affine transformation to image with rotated kernel image
  warpAffine(kernel, kernelRt, rotM, kernel.size());
  // Show rotated kernel image
  imshow("kernelrt", kernelRt);

  int iX = img.rows;
  int iY = img.cols;

  int kX = kernel.rows;
  int kY = kernel.cols;

  double sum;

  for(int x = kX / 2; x < iX - kX / 2; x++){
    for(int y = kY / 2; y < iY - kY / 2; y++){
      sum = 0.0;
      for(int j = -kX / 2; j <= kX / 2; j++){
        for(int k = -kY / 2; k <= kY / 2; k++){
          sum += kernel.at<uchar>(j+kX/2, k+kY/2)*img.at<uchar>(x + j, y + k);
        }
      }
      if (sum / (iX * iY) > 10) {
        oimg.at<uchar>(x,y) = 255.0f;
      }
    }
  }

  // Render image
  imshow("Convolution Output", oimg);
}

void correlation(Mat img, Mat kernel){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);

  int iX = img.rows;
  int iY = img.cols;

  int kX = kernel.rows;
  int kY = kernel.cols;

  double sum;

  for(int x = kX / 2; x < iX - kX / 2; x++){
    for(int y = kY / 2; y < iY - kY / 2; y++){
      sum = 0.0;
      for(int j = -kX / 2; j <= kX / 2; j++){
        for(int k = -kY / 2; k <= kY / 2; k++){
          sum += kernel.at<uchar>(j+kX/2, k+kY/2)*img.at<uchar>(x + j, y + k);
        }
      }
      if (sum / (iX * iY) > 10) {
        oimg.at<uchar>(x,y) = 255.0f;
      }
    }
  }

  // Render image
  imshow("Corelation Output", oimg);
}

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

      if (radius > r) {
        oplanes[0].at<float>(i, j) = planes[0].at<float>(i, j);
        oplanes[1].at<float>(i, j) = planes[1].at<float>(i, j);
      } else {
        oplanes[0].at<float>(i, j) = 0.0f;
        oplanes[1].at<float>(i, j) = 0.0f;
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
      (float)(1 / (1 + pow((double) (r / radius), (double) (2 * n))));

      oplanes[1].at<float>(i,j) = planes[1].at<float>(i, j) *
      (float)(1 / (1 + pow((double) (r / radius), (double) (2 * n))));
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
      (1 - exp((-radius*radius) / (2*d*d)));

      oplanes[1].at<float>(i,j) = planes[1].at<float>(i, j) *
      (1 - exp((-radius*radius) / (2*d*d)));
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
  Mat oimg;
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

Mat threshold(Mat img, float t){
  Mat oimg = Mat::zeros(img.rows, img.cols, img.type());

  // Applying average threshold
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if (img.at<float>(i,j) > t) {
        oimg.at<float>(i,j) = 1.0f;
      } else {
        oimg.at<float>(i,j) = 0.0f;
      }
    }
  }

  return oimg;
}

// Resimleri tek resim halinde birleştiren fonksiyon
Mat mergeImages(Mat imgList[], int rows, int cols){
  int size = 256; // Standart resim ebatı
  Mat oimg = Mat::zeros(size * rows, size * cols, CV_32F);

  // 12 resmin tek resimde birlestirilmesi
  for (int i = 0; i < rows * cols; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        oimg.at<float>((256 * (i/cols)) + j, (256*(i%cols)) + k) = imgList[i].at<uchar>(j, k);
      }
    }
  }

  return oimg;
}

int main(int argc, char** argv ){

  // Load Image
  Mat img = imread("characters.tif", 1 );
  Mat gimg = convertImage(img);
  Mat bimg = grayToBinary(gimg);

//  imshow("Image", gimg);
//  imshow("Images", bimg);

  Mat gcamdft = fastFT(gimg);
  Mat scamdft = shiftI(gcamdft);

  Mat text = imread("chars.png", 1);
  Mat gtext = convertImage(text);
  imshow("Text Image", gtext);

  Mat gtest = Mat::zeros(text.rows, text.cols, CV_8UC1);
  Mat roi = gtext(Rect(157, 140, 11, 11));
  roi.copyTo(gtest(Rect(0, 0, 11, 11)));

  imshow("Character", gtest);

  // 157 140, 11-11

  /*
    Q1A: Correlate Image with a character in it
  */

  correlation(gtext, roi);

  /*
    Q1B: Convolve Image with a character in it
  */

  clock_t begin = clock();
  convolution(gtext, roi);
  clock_t end = clock();

  /*
    Q2A: Print output time of Q1B
  */

  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  cout << "Elapsed time implementing convolution in spatial domain: " << elapsed_secs << "\n";

  /*
    Q2B: Print output time of Q1B, this time computed in frequency domain
  */

  clock_t begin1 = clock();
  Mat maskdft = fastFT(gtest);
  Mat smaskdft = shiftI(maskdft);

  Mat q5;
  Mat planes[] = {Mat_<float>(q5), Mat::zeros(q5.size(), CV_32F)};
  merge(planes, 2, q5);

  Mat charsdft = shiftI(fastFT(convertImage(gtext)));
  mulSpectrums(charsdft, smaskdft, q5, 1);
  smaskdft = shiftI(q5);
  maskdft = iFFT(smaskdft);
  imshow("Freq. Domain Char. Rec.", maskdft);
  clock_t end1 = clock();

  double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
  cout << "Elapsed time implementing convolution in frequency domain: " << elapsed_secs1 << "\n";

  /*
    Q3A: Low pass ideal filter for r = 30
  */

  Mat ifcamdft = idealFilter(scamdft, 30);
  Mat isfcamdft = shiftI(ifcamdft);
  Mat q11 = iFFT(isfcamdft);
  imshow("if1", q11);

  Mat ifcamdft2 = idealFilter(scamdft, 80);
  Mat isfcamdft2 = shiftI(ifcamdft2);
  Mat q12 = iFFT(isfcamdft2);
  imshow("if12", q12);

  /*
    Q3B: Low pass butterworth filter for r = size/3 & size*2/3 & d0 = 1,2,3
  */

  Mat bfcamdft = butterworthFilter(scamdft, 30, 1);
  Mat bsfcamdft = shiftI(bfcamdft);
  Mat q21 = iFFT(bsfcamdft);
  imshow("bf1", q21);

  Mat bfcamdft3 = butterworthFilter(scamdft, 60, 1);
  Mat bsfcamdft3 = shiftI(bfcamdft3);
  Mat q23 = iFFT(bsfcamdft3);
  imshow("bf3", q23);

  Mat bfcamdft5 = butterworthFilter(scamdft, 160, 1);
  Mat bsfcamdft5 = shiftI(bfcamdft5);
  Mat q25 = iFFT(bsfcamdft5);
  imshow("bf5", q25);

  /*
    Q3C: Low pass gaussian filter for r = size/3 & size*2/3 & l = 10, 20, 40
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

  waitKey();
  return 0;
}
