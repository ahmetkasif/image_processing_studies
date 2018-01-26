#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <random>
#include <chrono>

using std::cout;
using std::cin;
using std::endl;

using namespace std;
using namespace cv;

Mat addGaussianNoise(Mat img, double mean, double stddev);
Mat meanFilter(Mat img, float maskSize);
Mat mergeImages(Mat imgList[], int rows, int cols);
Mat convertImage(Mat img);
Mat gaussianExp(Mat img, float experiments, float stddev);
Mat verhorfilter(Mat img);
Mat diafilter(Mat img);
Mat comfilter(Mat img);

Mat gaussianExp(Mat img, float experiments, float stddev){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);;
  Mat gaussNoiseList[(int)experiments];

  // Belirtilen sayıda örneklenmiş normal dağılım gürültülü imgenin ortalaması hesaplanır.
  for (int k = 0; k < (int)experiments; k++) {
    gaussNoiseList[k] = addGaussianNoise(img, 0, stddev);
    oimg += (gaussNoiseList[k])/(int)experiments;
  }

  return oimg;
}

// Gaussian Gürültü Fonksiyonu
Mat addGaussianNoise(Mat img, double mean, double stddev){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);

  /*
    Bilgisayar zamanı tabanlı bir tohumlama yapılarak,
    örnekleme yapılırken her örnek için farklı bir gürültü hesaplaması simüle ediliyor
  */

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::normal_distribution<double>distribution(mean, stddev);

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      oimg.at<uchar>(i, j) = img.at<uchar>(i, j) + distribution(generator);
    }
  }

  return oimg;
}

Mat verhorfilter(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);

  // Yatay ve dikeyde filtreleme
  Mat kernel = (Mat_<float>(3,3) <<
    0, 1, 0,
    1, -4, 1,
    0, 1, 0);

  filter2D(img, oimg, CV_8U, kernel);

  return oimg;
}

Mat diafilter(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);

  Mat kernel = (Mat_<float>(3,3) <<
    1, 0, 1,
    0, -4, 0,
    1, 0, 1);
  filter2D(img, oimg, CV_8U, kernel);

  return oimg;
}

Mat comfilter(Mat img){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1);

  Mat kernel = (Mat_<float>(3,3) <<
    1, 1, 1,
    1, -8, 1,
    1, 1, 1);
  filter2D(img, oimg, CV_8U, kernel);

  return oimg;
}

// Aritmetik Ortalama Filtresi, istenen maske boyutunda
Mat meanFilter(Mat img, float maskSize){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1); // Çıktı matrisi
  Mat temp = Mat::zeros(img.rows + (int)maskSize-1, img.cols + (int)maskSize-1, CV_8UC1); // Gecici matris
  float sum;
  float mask[(int)maskSize][(int)maskSize];

  // Maskenin hazirlanmasi
  for (int i = 0; i < (int)maskSize; i++) {
    for (int j = 0; j < (int)maskSize; j++) {
      mask[i][j] = 1/(maskSize*maskSize);
    }
  }

  // Geçici matrisin genisletilmesi
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      temp.at<uchar>(i+(int)maskSize/2, j+(int)maskSize/2) = img.at<uchar>(i, j);
    }
  }

  // Dikeyde genişletilmiş resmin aynasını alma işlemi
  for (int i = 0; i < (int)maskSize/2; i++) {
    for (int j = 0; j < temp.cols - (int)maskSize; j++) {
      temp.at<uchar>((int)(maskSize/2)-i-1, j+(int)maskSize/2) = img.at<uchar>(i, j);
      temp.at<uchar>(img.rows + (int)(maskSize/2)+i, j+(int)maskSize/2) = img.at<uchar>(img.rows-i-1, j);
    }
  }

  // Yatayda genişletilmiş resmin aynasını alma işlemi
  for (int i = 0; i < temp.cols - (int)maskSize; i++) {
    for (int j = 0; j < (int)maskSize/2; j++) {
      temp.at<uchar>(i+(int)maskSize/2, (int)(maskSize/2)-j-1) = img.at<uchar>(i, j);
      temp.at<uchar>(i+(int)maskSize/2, img.cols + (int)(maskSize/2)+j) = img.at<uchar>(i, img.cols-j-1);
    }
  }

  // Maskenin temp matrisine uygulanmasi
  for(int y = (int)(maskSize)/2; y < temp.rows; y++){ // MaskSize=3 için y=1'den baslar
    for(int x = (int)(maskSize)/2; x < temp.cols; x++){ // MaskSize=3 için x=1'den baslar
      sum = 0.0;
      for(int k = -(int)(maskSize)/2; k <= (int)(maskSize)/2;k++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
        for(int j = -(int)(maskSize)/2; j <=(int)(maskSize)/2; j++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
          // maskenin ilgili indisini o an seçili bölgenin ilgili indisi ile çarpip toplama ekler
          sum = sum + mask[j+(int)(maskSize/2)][k+(int)(maskSize/2)]*img.at<uchar>(y - j, x - k);
        }
      }
      temp.at<uchar>(y,x) = sum; // Elde edilen toplamin yazilmasi
    }
  }

  // Büyültülmüs geçici matrisin tekrar daraltilmasi
  for (int i = 0; i < oimg.rows; i++) {
    for (int j = 0; j < oimg.cols; j++) {
      oimg.at<uchar>(i, j) = temp.at<uchar>(i+((int)maskSize/2), j+((int)maskSize/2));
    }
  }

  return oimg;
}

// Resimleri tek resim halinde birleştiren fonksiyon
Mat mergeImages(Mat imgList[], int rows, int cols){
  int size = 256; // Standart resim ebatı
  Mat oimg = Mat(size * rows, size * cols, CV_8UC1);

  // 12 resmin tek resimde birlestirilmesi
  for (int i = 0; i < rows * cols; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        oimg.at<uchar>((256 * (i/cols)) + j, (256*(i%cols)) + k) = imgList[i].at<uchar>(j, k);
      }
    }
  }

  return oimg;
}

// RGB2Grayscale dönüşüm fonksiyonu
Mat convertImage(Mat img){
  Mat oimg = Mat(img.rows, img.cols, CV_8UC1);

  Mat c[3];
  split(img,c); // Girdi resmi kanallarına ayrılıp c dizisine yüklenir.

  c[0] *= 0.299;
  c[1] *= 0.587;
  c[2] *= 0.114;

  oimg = c[0] + c[1] + c[2];

  return oimg;
}

int main(int argc, char** argv ){
  if ( argc != 2 ){
      printf("usage: DisplayImage.out <Image_Path>\n"); // Resim konumu kullanıcıdan alınır.
      return -1;
  }

  Mat img = imread( argv[1], 1 );
  Mat gimg = convertImage(img);

  // Gaussian noise ekle, mean filter, median filter ile tara
  Mat gaussNoise[14];

  gaussNoise[0] = gimg;
  gaussNoise[1] = addGaussianNoise(gimg, 0, 10.0);
  gaussNoise[2] = gaussianExp(gimg, 5.0f, 10.0);
  gaussNoise[3] = gaussianExp(gimg, 10.0f, 10.0);
  gaussNoise[4] = gaussianExp(gimg, 20.0f, 10.0);
  gaussNoise[5] = gaussianExp(gimg, 50.0f, 10.0);
  gaussNoise[6] = gaussianExp(gimg, 100.0f, 10.0);

  gaussNoise[7] = gimg;
  gaussNoise[8] = addGaussianNoise(gimg, 0, 20.0);
  gaussNoise[9] = gaussianExp(gimg, 5.0f, 20.0);
  gaussNoise[10] = gaussianExp(gimg, 10.0f, 20.0);
  gaussNoise[11] = gaussianExp(gimg, 20.0f, 20.0);
  gaussNoise[12] = gaussianExp(gimg, 50.0f, 20.0);
  gaussNoise[13] = gaussianExp(gimg, 100.0f, 20.0);

  Mat q1 = mergeImages(gaussNoise, 2, 7);
  imwrite("gaussian.png", q1);
  imshow("Gaussian Noise", q1);

  // Keskinleştirme filtresi uygula, orjinal ve art. ort. filt. 3x3, 5x5 ve 7x7 için

  Mat vhf[12];

  vhf[0] = gimg;
  vhf[1] = meanFilter(gimg, 3.0);
  vhf[2] = meanFilter(gimg, 5.0);
  vhf[3] = meanFilter(gimg, 7.0);
  vhf[4] = verhorfilter(gimg);
  vhf[5] = verhorfilter(vhf[1]) * 3;
  vhf[6] = verhorfilter(vhf[2]) * 4;
  vhf[7] = verhorfilter(vhf[3]) * 5;
  vhf[8] = gimg - verhorfilter(gimg);
  vhf[9] = vhf[1] - verhorfilter(vhf[1]);
  vhf[10] = vhf[2] - verhorfilter(vhf[2]);
  vhf[11] = vhf[3] - verhorfilter(vhf[3]);

  Mat q2 = mergeImages(vhf, 3, 4);
  imwrite("dikey-yatay-kesk-filt.png", q2);
  imshow("Dikey-Yatay Kesk. Filt.", q2);

  Mat df[12];

  df[0] = gimg;
  df[1] = meanFilter(gimg, 3.0);
  df[2] = meanFilter(gimg, 5.0);
  df[3] = meanFilter(gimg, 7.0);
  df[4] = diafilter(gimg);
  df[5] = diafilter(df[1]) * 3;
  df[6] = diafilter(df[2]) * 4;
  df[7] = diafilter(df[3]) * 5;
  df[8] = gimg - diafilter(gimg);
  df[9] = df[1] - diafilter(df[1]);
  df[10] = df[2] - diafilter(df[2]);
  df[11] = df[3] - diafilter(df[3]);

  Mat q2_1 = mergeImages(df, 3, 4);
  imwrite("capraz-kesk-filt.png", q2_1);
  imshow("Çapraz Kesk. Filt.", q2_1);

  Mat cf[12];

  cf[0] = gimg;
  cf[1] = meanFilter(gimg, 3.0);
  cf[2] = meanFilter(gimg, 5.0);
  cf[3] = meanFilter(gimg, 7.0);
  cf[4] = comfilter(gimg);
  cf[5] = comfilter(cf[1]) * 3;
  cf[6] = comfilter(cf[2]) * 4;
  cf[7] = comfilter(cf[3]) * 5;
  cf[8] = gimg - comfilter(gimg);
  cf[9] = cf[1] - comfilter(cf[1]);
  cf[10] = cf[2] - comfilter(cf[2]);
  cf[11] = cf[3] - comfilter(cf[3]);

  Mat q2_2 = mergeImages(cf, 3, 4);
  imwrite("tam-kesk-filt.png", q2_2);
  imshow("Tam Kesk. Filt.", q2_2);

  waitKey();
  return 0;
}
