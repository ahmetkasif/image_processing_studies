#include <stdio.h>
#include <opencv2/opencv.hpp>

using std::cout;
using std::cin;
using std::endl;

using namespace std;
using namespace cv;

// Minimum fonksiyonu
void findMin(Mat img, int (&x)[3]){
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      for (int k = 0; k < 3; k++) {
        if (x[k] > (int)img.at<Vec3b>(i,j).val[k]) {
          x[k] = (int)img.at<Vec3b>(i,j).val[k];
        }
      }
    }
  }
}

// Maximum fonksiyonu
void findMax(Mat img, int (&x)[3]){
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      for (int k = 0; k < 3; k++) {
        if (x[k] < (int)img.at<Vec3b>(i,j).val[k]) {
          x[k] = (int)img.at<Vec3b>(i,j).val[k];
        }
      }
    }
  }
}

// Ortalama bulan fonksiyon
void findAvg(Mat img, int (&x)[3]){
  int temp[3] = {0};
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      for (int k = 0; k < 3; k++) {
        temp[k] += (int)img.at<Vec3b>(i,j).val[k];
      }
    }
  }

  for (int i = 0; i < 3; i++) {
    x[i] = temp[i]/(img.rows*img.cols);
  }
}

int main(int argc, char** argv ){
  int min[3]={256, 256, 256};
  int max[3]={0};
  int avg[3]={0};

  if ( argc != 2 ){
      printf("usage: DisplayImage.out <Image_Path>\n"); // Resim konumu kullanıcıdan alınır.
      return -1;
  }

  Mat img = imread( argv[1], 1 );
  imshow("Original Image", img);

  findMin(img, min);
  findMax(img, max);
  findAvg(img, avg);

  std::cout << "Minimum Değerleri" << '\n';

  for (int i = 0; i < 3; i++) {
    std::cout << "min: " << min[i] << '\n';
  }

  std::cout << "Maksimum Değerleri" << '\n';

  for (int i = 0; i < 3; i++) {
    std::cout << "max: " << max[i] << '\n';
  }

  std::cout << "Ortalama Değerleri" << '\n';

  for (int i = 0; i < 3; i++) {
    std::cout << "avg: " << avg[i] << '\n';
  }



  waitKey();
  return 0;
}
