#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>
#include <iostream>

using namespace cv::ml;
using namespace cv;
using namespace std;

HOGDescriptor hog(
  Size(20,20), //winSize
  Size(10, 10), //blocksize
  Size(5, 5), //blockStride,
  Size(10, 10), //cellSize,
  9, //nbins,
  1, //derivAper,
  -1, //winSigma,
  0, //histogramNormType,
  0.2, //L2HysThresh,
  0,//gammal correction,
  64,//nlevels=64
  1);

// Saves characters in to memory via Mat vector, save labels into int vector
void loadTrainImage(Mat img, vector<Mat> &trainImage, vector<int> &trainLabels, int size);
// Saves characters in to memory via Mat vector
void loadTestImage(Mat img, vector<Mat> &testImage, vector<int> &testLabels);
// Preprocesses character matrices and saves them on another Mat vector
void preprocess(vector<Mat> &img, vector<Mat> &processedImg, int size);
// Creates feature vector via HOG method and saves it into a 2d vector structure
void hogFV(vector<Mat> &processedImg, vector<vector<float> > &trainHOG);
// Moves feature vectors from 2d vector to matrice
void vec2mat(vector<vector<float> > &imageHOG, Mat &imageMat);
// Deskewing Function
Mat deskew(Mat& img, int size);
// Grayscale to Binary conversion function
void gtb(Mat &img);
// SVM Parameters logging function
void getSVMParams(SVM *svm);

void loadTrainImage(Mat img, vector<Mat> &trainImage, vector<int> &trainLabels, int size){
  Mat digitImg;
  int ImgCount = 0;

  // cloning images every 20px x 20px, because size = 20px
  for(int i=0; i<img.rows; i=i+size){
    for(int j=0; j<img.cols; j=j+size){
      digitImg = (img.colRange(j, j+size).rowRange(i, i+size)).clone();
      trainImage.push_back(digitImg);
      ImgCount++;
    }
  }

  cout << "Train Image Count: " << ImgCount << endl;

  /*
    Coding labels of train images, changing code value for every 500 image
  */

  float digitValue = 0;
  for(int z=0; z<int(ImgCount); z++){
    if(z % 500 == 0 && z != 0){
      digitValue = digitValue + 1;
    }
    trainLabels.push_back(digitValue);
  }
}

void loadTestImage(Mat img, vector<Mat> &testImage, vector<int> &testLabels){
  int MIN_CONTOUR_AREA = 20;
  int RESIZED_IMAGE_WIDTH = 20;
  int RESIZED_IMAGE_HEIGHT = 20;

  std::vector<std::vector<cv::Point> > contourPoint;
  std::vector<cv::Vec4i> v4iHierarchy;

  // Finds contours and loads into contourPoint
  cv::findContours(img, contourPoint, v4iHierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  cv::Mat imgTrainingNumbers;
  int testImgCount = 0;

  for (int i = 0; i < contourPoint.size(); i++) { // for each contour
    if (cv::contourArea(contourPoint[i]) > MIN_CONTOUR_AREA) { // if contour is big enough to consider
      cv::Rect boundingRect = cv::boundingRect(contourPoint[i]); // get the bounding rect

      cv::Mat matROI = img(boundingRect); // get ROI image of bounding rect

      cv::Mat matROIResized;
      cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));
      cv::imshow("matROI", matROIResized);
      testImage.push_back(matROIResized);
      testImgCount++;

      int temp = cv::waitKey(0) - 176;

      testLabels.push_back(temp);
      cout << "Entered " << temp << endl;
    }
  }

  cout << "Test Image Count : " << testImgCount << endl;
}

void preprocess(vector<Mat> &img, vector<Mat> &processedImg, int size){
  // apply deskewing to images present in img, save into processedImg
  for(int i=0; i<img.size(); i++){
    Mat temp = deskew(img[i], size);
    processedImg.push_back(temp);
  }
}

void hogFV(vector<Mat> &processedImg, vector<vector<float> > &imageHOG){
  vector<float> descriptors;
  for(int y=0; y<processedImg.size(); y++){
    hog.compute(processedImg[y],descriptors);
    imageHOG.push_back(descriptors);
  }
}

void vec2mat(vector<vector<float> > &imageHOG, Mat &imageMat){
  int descriptorSize = imageHOG[0].size();

  for(int i = 0; i<imageHOG.size(); i++){
    for(int j = 0; j<descriptorSize; j++){
      imageMat.at<float>(i,j) = imageHOG[i][j];
    }
  }
}

Mat deskew(Mat& img, int size){
  float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;
  Moments m = moments(img);
  // return image, if it does not need deskewing
  if(abs(m.mu02) < 1e-2){
    return img.clone();
  }

  // calculate skewness
  float skew = m.mu11/m.mu02;
  // calculate corresponding warpMat for correcting skewness
  Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5 * size * skew, 0, 1, 0);

  // Apply affine transformation to img and save into oimg
  Mat oimg = Mat::zeros(img.rows, img.cols, img.type());
  warpAffine(img, oimg, warpMat, oimg.size(), affineFlags);

  return oimg;
}

void gtb(Mat &img){
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
      if (img.at<uchar>(i,j) > avg * 0.8f) {
        img.at<uchar>(i,j) = 0.0f;
      } else {
        img.at<uchar>(i,j) = 255.0f;
      }
    }
  }
}

void getSVMParams(SVM *svm){
  cout << "Kernel type     : " << svm->getKernelType() << endl;
  cout << "Type            : " << svm->getType() << endl;
  cout << "C               : " << svm->getC() << endl;
  cout << "Degree          : " << svm->getDegree() << endl;
  cout << "Nu              : " << svm->getNu() << endl;
  cout << "Gamma           : " << svm->getGamma() << endl;
}

int main(){
  int size = 20;

  vector<Mat> trainImages; // train images 20*20
  vector<int> trainLabels; // train image labels in between 0-9, integer

  // --------- TRAINING -------------

  Mat img = imread("digits.png", CV_8UC1);

  // Loads Train Image and corresponding labels
  loadTrainImage(img, trainImages, trainLabels, size);

  // Preprocessing Train Image
  vector<Mat> processedtrainImages;
  preprocess(trainImages, processedtrainImages, size);

  // Feature Vector of Train Image via HOG
  std::vector<std::vector<float> > trainHOG;
  hogFV(processedtrainImages, trainHOG);

  int descriptorSize = trainHOG[0].size();
  cout << "Descriptor Size : " << descriptorSize << endl;

  // Move from 2d vector to Mat
  Mat trainMat(trainHOG.size(), descriptorSize, CV_32FC1);
  vec2mat(trainHOG, trainMat);

  // Train SVM
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(0.50625);
  svm->setC(12.5);
  svm->setKernel(SVM::RBF);
  svm->setType(SVM::C_SVC);
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  svm->train(td);
  // Save svm model in a yml file
  svm->save("model4.yml");

  // --------- TESTING -------------

  vector<Mat> testImages; // test images 20*20
  vector<int> testLabels; // test image labels in between 0-9, integer

  // Loads Test Image and finds numbers
  Mat testimg = imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  Mat testimgb;
  cv::GaussianBlur(testimg, testimgb, cv::Size(7, 7), 0) ;
  gtb(testimgb);

  imshow("Test Image", testimgb);

  loadTestImage(testimgb, testImages, testLabels);

  // post-processing Test Images
  vector<Mat> processedtestImages;
  preprocess(testImages, processedtestImages, size);

  // Feature Vector of Test Image via HOG
  std::vector<std::vector<float> > testHOG;
  hogFV(processedtestImages, testHOG);

  // Move from 2d vector to Mat
  Mat testMat(testHOG.size(), descriptorSize, CV_32FC1);
  vec2mat(testHOG, testMat);

  // Predict test using trained SVM
  Mat testResponse;
  svm->predict(testMat, testResponse);
  getSVMParams(svm);

  for(int i=0; i<testResponse.rows; i++){
      cout << "Real value: " << testLabels[i] << ", predicted value: "
      << testResponse.at<float>(0, i) << endl;
      if (testLabels[i] != testResponse.at<float>(0, i)) {
        cout << "Values do not match!!" << endl;
      }
  }

  return 0;
}
