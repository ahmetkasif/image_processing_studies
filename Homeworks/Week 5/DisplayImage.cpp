#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <random>

using std::cout;
using std::cin;
using std::endl;

using namespace std;
using namespace cv;

Mat locHisEq(Mat img, int maskSize);
Mat addSPNoise(Mat img);
Mat addGaussianNoise(Mat img, double mean, double stddev);
Mat stdDevFilter(Mat img, float maskSize);
Mat meanFilter(Mat img, float maskSize);
Mat medianFilter(Mat img, float maskSize);
Mat mergeImages(Mat imgList[], int rows, int cols);
Mat convertImage(Mat img);
Mat histEq(Mat img);
void insertionSort(int list[]);

// Bölgesel histogram eşitleme, istenen maske boyutunda
Mat locHisEq(Mat img, int maskSize){
  Mat oimg = Mat(img.rows, img.cols, CV_8UC1);
  Mat t = Mat::zeros(maskSize, maskSize,CV_8UC1);
  Mat t2 = Mat::zeros(maskSize, maskSize, CV_8UC1);

  for (int x = 0; x < img.rows-maskSize; x+=maskSize) {
    for (int y = 0; y < img.cols-maskSize; y+=maskSize) {
      // İslenecek kismin gecici matrise aktarilmasi
      for (int i = 0; i < maskSize; i++) {
        for (int j = 0; j < maskSize; j++) {
          t.at<uchar>(i,j) = img.at<uchar>(i+x,j+y);
        }
      }

      //  geçici matrisin histogram eşitlenerek geri yazılması.
      for (int i = 0; i < maskSize; i++) {
        for (int j = 0; j < maskSize; j++) {
          oimg.at<uchar>(i+x,j+y) = histEq(t).at<uchar>(i,j);
        }
      }
    }
  }

  return oimg;
}

// Salt & Pepper Gürültüsü Fonksiyonu
Mat addSPNoise(Mat img, double corrRatio){
  Mat temp = Mat::ones(img.rows, img.cols, CV_8UC1); // Piksel ilk değerleri 1
  Mat oimg = Mat(img.rows, img.cols, CV_8UC1);
  int a = 0;
  int b = 0;

  // S&P matrisi bir filtre matrisi gibi oluşturulur.
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      a = rand() % (int)(1/corrRatio); // Bozulacak piksel oranı
      b = rand() % 2; // Piksel is salt mu pepper mı

      if (a == 1) { // tek ise bozuyoruz
        if (b == 0) {
          temp.at<uchar>(i,j) = 0.0f;
        } else if (b == 1) {
          temp.at<uchar>(i,j) = 255.0f;
        }
      }
    }
  }

  // S&P filtresi (temp) resme uygulanıyor ve çıktı döndürülüyor
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      if ((int)temp.at<uchar>(i,j) == 0) {
        oimg.at<uchar>(i,j) = 0;
      } else if ((int)temp.at<uchar>(i,j) == 255) {
        oimg.at<uchar>(i,j) = 255;
      } else {
        oimg.at<uchar>(i,j) = img.at<uchar>(i,j);
      }
    }
  }

  return oimg;
}

// Gaussian Gürültü Fonksiyonu
Mat addGaussianNoise(Mat img, double mean, double stddev){
  Mat oimg = Mat(img.rows, img.cols, CV_8UC1);

  const int nrolls=10000;  // number of experiments
  const int nstars=100;    // maximum number of stars to distribute

  std::default_random_engine generator;
  std::normal_distribution<double>distribution(mean, stddev);

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      oimg.at<uchar>(i, j) = img.at<uchar>(i, j) + distribution(generator);
    }
  }

  return oimg;
}

// Standart Sapma Fİltresi, istenen maske boyutunda
Mat stdDevFilter(Mat img, float maskSize){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1); // Çıktı matrisi
  Mat temp = Mat::zeros(img.rows + (int)maskSize-1, img.cols + (int)maskSize-1, CV_8UC1); // Gecici matris
  float mean, sum, max;
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

  // Standart sapmanın piksel piksel hesaplanması
  for(int y = (int)(maskSize)/2; y < temp.rows; y++){ // MaskSize=3 için y=1'den baslar
    for(int x = (int)(maskSize)/2; x < temp.cols; x++){ // MaskSize=3 için x=1'den baslar
      sum = 0.0f;
      mean = 0.0f;

      for(int k = -(int)(maskSize)/2; k <= (int)(maskSize)/2;k++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
        for(int j = -(int)(maskSize)/2; j <=(int)(maskSize)/2; j++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
          mean = mean + mask[j+(int)(maskSize/2)][k+(int)(maskSize/2)]*img.at<uchar>(y - j, x - k); // maskenin ilgili indisini o an seçili bölgenin ilgili indisi ile çarpip toplama ekler
        }
      }

      // Ortalamaya olan uzaklıkların karelerin toplamının sum'a aktarılması
      for(int k = -(int)(maskSize)/2; k <= (int)(maskSize)/2;k++){
        for(int j = -(int)(maskSize)/2; j <=(int)(maskSize)/2; j++){
          sum += ((int)img.at<uchar>(y - j, x - k) - mean) * ((int)img.at<uchar>(y - j, x - k) - mean);
        }
      }

      // Toplamın maskedeki terim sayısının bir eksiğine bölünmesi
      sum = sum/(pow(maskSize, 2)-1);
      // Toplamın karekökünün alınması
      sum = sqrt(sum);
      // Elde edilen sayıların, daha sonra 0-255 arasına aktarılması için elde edilen en büyük değerin hesaplanması
      if (max < sum) {
        max = sum;
      }

      temp.at<uchar>(y,x) = sum; // Elde edilen toplamin yazilmasi
    }
  }

  // Büyültülmüs geçici matrisin tekrar daraltilmasi
  for (int i = 0; i < oimg.rows; i++) {
    for (int j = 0; j < oimg.cols; j++) {
      // Piksel değeri aktarılırken aynı zamanda max değişkeni kullanılarak, 0-255 arasına aktarılıyor.
      oimg.at<uchar>(i, j) = temp.at<uchar>(i+((int)maskSize/2), j+((int)maskSize/2))/sum*255.0f;
    }
  }

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

// Medyan Filtresi, istenen maske boyutunda
Mat medianFilter(Mat img, float maskSize){
  Mat oimg = Mat::zeros(img.rows, img.cols, CV_8UC1); // Çıktı matrisi
  Mat temp = Mat::zeros(img.rows + (int)maskSize-1, img.cols + (int)maskSize-1, CV_8UC1); // Gecici matris
  int list[(int)maskSize * (int)maskSize];

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

  // Değerlerin tek boyutlu bir diziye aktarılıp, sıralama işlemi yapılması, ortadaki elemanın çekilmesi
  for(int y = (int)(maskSize)/2; y < temp.rows; y++){ // MaskSize=3 için y=1'den baslar
    for(int x = (int)(maskSize)/2; x < temp.cols; x++){ // MaskSize=3 için x=1'den baslar
      int c = 0;
      for(int k = -(int)(maskSize)/2; k <= (int)(maskSize)/2;k++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
        for(int j = -(int)(maskSize)/2; j <=(int)(maskSize)/2; j++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
          list[c] = temp.at<uchar>(y - j, x - k);
          c++;
        }
      }
      insertionSort(list);

      temp.at<uchar>(y,x) = list[4]; // Medyan değerinin geri yazilmasi
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

Mat histEq(Mat img){
  Mat o_img = Mat::zeros(img.rows, img.cols, CV_8UC1); // Tek kanallı histogram matrisi

  // Histogramın oluşturulması
  double hist[256] = {0};
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      hist[(int)img.at<uchar>(i, j)] += 1.0;
    }
  }

  // PMF dizisi hesaplanır.
  double pmf[256];
  for (int i = 0; i < 256; i++) {
    pmf[i] = hist[i] / (img.rows * img.cols);
  }

  // CDF fonksiyonu hesaplanır
  double cdf[256];
  cdf[0] = pmf[0];
  for(int i = 1; i < 256; i++){
    cdf[i] = pmf[i] + cdf[i-1];
  }

   // CDF fonksiyonu 0-1 aralığından 0-255 aralığına aktarılıyor
  for (int i = 0; i < 256; i++) {
    cdf[i] =  (int)(cdf[i]*255);
  }

  // Resmin piksellerinin parlaklık değerleri, cdf fonksiyonundaki parlaklık değerleriyle değiştirilir.
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      o_img.at<uchar>(i,j) = cdf[(int)img.at<uchar>(i,j)];
    }
  }

  return o_img;
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

// Insertion sort, sıralama algoritma fonksiyonu
void insertionSort(int list[]){
  int temp, i , j;
  for(i = 0; i < 9; i++){
    temp = list[i];
    for(j = i-1; j >= 0 && temp < list[j]; j--){
      list[j+1] = list[j];
    }
    list[j+1] = temp;
  }
}

int main(int argc, char** argv ){
  if ( argc != 2 ){
      printf("usage: DisplayImage.out <Image_Path>\n"); // Resim konumu kullanıcıdan alınır.
      return -1;
  }

  Mat img = imread( argv[1], 1 );
  Mat gimg = convertImage(img);

  // Local histogram equalization by 4x4, 8x8, 16x16

  Mat lochiseq[4];
  lochiseq[0] = gimg;
  lochiseq[1] = locHisEq(gimg, 2);
  lochiseq[2] = locHisEq(gimg, 8);
  lochiseq[3] = locHisEq(gimg, 16);
  Mat q1 = mergeImages(lochiseq, 1, 4);
  imwrite("loc-hist-eq.png", q1);
  imshow("Local Hist Eq", q1);

  // Salt & Pepper noise, mean filter, median filter ile tarama
  Mat spNoise[8];
  spNoise[0] = gimg;
  spNoise[1] = addSPNoise(gimg, 0.5);
  spNoise[2] = meanFilter(spNoise[1], 3);
  spNoise[3] = medianFilter(spNoise[1], 3);
  spNoise[4] = gimg;
  spNoise[5] = addSPNoise(gimg, 0.25);
  spNoise[6] = meanFilter(spNoise[5], 3);
  spNoise[7] = medianFilter(spNoise[5], 3);
  Mat q2 = mergeImages(spNoise, 2, 4);
  imwrite("salt-pepper.png", q2);
  imshow("Salt & Pepper Noise", q2);

  // Gaussian noise ekle, mean filter, median filter ile tara

  Mat gaussNoise[8];
  gaussNoise[0] = gimg;
  gaussNoise[1] = addGaussianNoise(gimg, 0, 2.0);
  gaussNoise[2] = meanFilter(gaussNoise[1], 3);
  gaussNoise[3] = medianFilter(gaussNoise[1], 3);
  gaussNoise[4] = gimg;
  gaussNoise[5] = addGaussianNoise(gimg, 0, 8.0);
  gaussNoise[6] = meanFilter(gaussNoise[5], 3);
  gaussNoise[7] = medianFilter(gaussNoise[5], 3);
  Mat q3 = mergeImages(gaussNoise, 2, 4);
  imwrite("gaussian.png", q3);
  imshow("Gaussian Noise", q3);

  // Standart sapma filtre uygula 3x3, 5x5, 7x7 için
  Mat stdDevFil[4];
  stdDevFil[0] = gimg;
  stdDevFil[1] = stdDevFilter(gimg, 3);
  stdDevFil[2] = stdDevFilter(gimg, 5);
  stdDevFil[3] = stdDevFilter(gimg, 7);
  Mat q4 = mergeImages(stdDevFil, 1, 4);
  imwrite("stdevfilt.png", q4);
  imshow("Standart Deviation Filter", q4);

  waitKey();
  return 0;
}
