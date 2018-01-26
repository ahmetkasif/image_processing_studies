#include <stdio.h>
#include <opencv2/opencv.hpp>

using std::cout;
using std::cin;
using std::endl;

using namespace std;
using namespace cv;

Mat convertImage(Mat img){
  Mat o_img = Mat(img.rows, img.cols, CV_8UC1); // Çıktı için aynı boyutta yeni bir matris oluşturulur.

  Mat c[3];
  split(img,c); // Girdi resmi kanallarına ayrılıp c dizisine yüklenir.

  // RGB[A] to Gray: Y <- 0.299 * G, + 0.587 * R, + 0.114 * B , Kanallar ilgili değerlerle çarılarak toplanır ve tek kanallı new_image matrisine aktarılır.

  c[0] *= 0.299;
  c[1] *= 0.587;
  c[2] *= 0.114;

  o_img = c[0] + c[1] + c[2];

  return o_img;
}

Mat plotHistogram(Mat img){
  Mat o_img = Mat::zeros(256, 256, CV_8UC1); // Tek kanallı histogram matrisi
  int hist[256] = {0}; // Histogram verisinin tutulacağı dizi
  float temp = 0.0f; // En yüksek olasılığın tutulduğu temp değişkeni

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      hist[(int)img.at<uchar>(i, j)]++;  // Tüm matris dolaşılarak histogram verisi toplanır
    }
  }

  // Temp değişkeni hesaplanır
  for (int i = 0; i < 256; i++) {
    if (hist[i] > temp) {
      temp = hist[i];
    }
  }

  // Tüm değerler, temp kullanılarak 0-256 arasında bir değere aktarılır
  for (int i = 0; i < 256; i++) {
    hist[i] = hist[i] / temp * 256.0f;
  }

  // 0'dan histogram değerine kadar pikseller beyaza boyanır
  for (int j = 0; j < 256; j++) {
    for (int k = 0; k < hist[j]; k++) {
      o_img.at<uchar>(255-k, j) = 255;
    }
  }

  return o_img;
}

Mat gammaCorrection(Mat img, float fGamma){
  unsigned char lut[256];
	for (int i = 0; i < 256; i++){
		lut[i] = pow((float)(i / 255.0), 1/fGamma) * 255.0f; // Lut dizisinin oluşturulması (i/255)^(1/fGamma)
	}
  Mat o_img = Mat(img.rows, img.cols, img.type()); // Çıktı için aynı boyutta yeni bir matris oluşturulur.

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      o_img.at<uchar>(i, j) = lut[(int)(img.at<uchar>(i,j))]; // Çıktı resmindeki her pikselin r,g,b kanallarına ayrı ayrı lut katsayısı uygulanır.
    }
  }

  return o_img;
}

Mat histEqualize(Mat img){
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

void q1(Mat img){
  Mat heq = Mat::zeros(256, 256, CV_8UC1); // Histogram esitlenmis resim
  Mat heq_his = Mat::zeros(256, 256, CV_8UC1); // Histogram esitlenmis resmin histogrami
  Mat out = Mat::zeros(256, 256, CV_8UC1); // Histogram uyarlanmis resim
  Mat out_his = Mat::zeros(256, 256, CV_8UC1); // Histogram uyarlanmis resmin histogrami
  Mat all = Mat::zeros(512, 512, CV_8UC1); // Birlestirilmis matris

  float temp = 0.0f; // En yüksek histogram degerini tutar

  heq = histEqualize(img); // Histogram eşitlenir
  heq_his = plotHistogram(heq); // Eşitlenmiş histogramın grafiğe dönüştürülmesi

  // Hedef Histogram
  int cdf[256];
  for (int i = 0; i < 256; i++) {
    if (i<128) {
      cdf[i] = i*2;
    } else if(i>= 128){
      cdf[i] = (256-i)*2;
    }
  }

  // Resmin piksellerinin parlaklık değerleri, cdf fonksiyonundaki parlaklık değerleriyle değiştirilir.
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      out.at<uchar>(i,j) = cdf[(int)heq.at<uchar>(i,j)];
    }
  }

  out_his = plotHistogram(out);

  // 4 resmin tek resmen nakledilmesi
  for (int i = 0; i < 512; i++) {
    for (int j = 0; j < 512; j++) {
      if (i<256) {
        if (j<256) {
          all.at<uchar>(i,j) = heq.at<uchar>(i,j);
        } else if (j >= 256) {
          all.at<uchar>(i,j) = heq_his.at<uchar>(i,j-256);
        }
      } else if (i>=256) {
        if (j<256) {
          all.at<uchar>(i,j) = out.at<uchar>(i-256,j);
        } else if (j >= 256) {
          all.at<uchar>(i,j) = out_his.at<uchar>(i-256,j-256);
        }
      }
    }
  }

  imshow("Hist Eq & Spec", all);
}

void q2(Mat img){
  Mat all = Mat::zeros(768, 1024, CV_8UC1); // Birlestirilmis matris
  Mat g1 = gammaCorrection(img, 1.0f);
  Mat g1_his = plotHistogram(g1);
  Mat g1eq = histEqualize(g1);
  Mat g1eq_his = plotHistogram(g1eq);

  Mat g05 =  gammaCorrection(img, 0.5f);
  Mat g05_his = plotHistogram(g05);
  Mat g05eq = histEqualize(g05);
  Mat g05eq_his = plotHistogram(g05eq);

  Mat g20 = gammaCorrection(img, 2.0f);
  Mat g20_his = plotHistogram(g20);
  Mat g20eq = histEqualize(g20);
  Mat g20eq_his = plotHistogram(g20eq);

  // 12 resmin tek resimde birlestirilmesi
  for (int i = 0; i < 768; i++) {
    for (int j = 0; j < 1024; j++) {
      if (i<256) {
        if (j<256) {
          all.at<uchar>(i,j) = g1.at<uchar>(i,j);
        } else if (j >= 256 && j < 512) {
          all.at<uchar>(i,j) = g1_his.at<uchar>(i,j-256);
        } else if (j >= 512 && j < 768) {
          all.at<uchar>(i,j) = g1eq.at<uchar>(i,j-512);
        } else if (j >= 768) {
          all.at<uchar>(i,j) = g1eq_his.at<uchar>(i,j-768);
        }
      } else if (i>=256 && i<512) {
        if (j<256) {
          all.at<uchar>(i,j) = g05.at<uchar>(i-256,j);
        } else if (j >= 256 && j < 512) {
          all.at<uchar>(i,j) = g05_his.at<uchar>(i-256,j-256);
        } else if (j >= 512 && j < 768) {
          all.at<uchar>(i,j) = g05eq.at<uchar>(i-256,j-512);
        } else if (j >= 768) {
          all.at<uchar>(i,j) = g05eq_his.at<uchar>(i-256,j-768);
        }
      } else if (i>=512) {
        if (j<256) {
          all.at<uchar>(i,j) = g20.at<uchar>(i-512,j);
        } else if (j >= 256 && j < 512) {
          all.at<uchar>(i,j) = g20_his.at<uchar>(i-512,j-256);
        } else if (j >= 512 && j < 768) {
          all.at<uchar>(i,j) = g20eq.at<uchar>(i-512,j-512);
        } else if (j >= 768 && j < 1024) {
          all.at<uchar>(i,j) = g20eq_his.at<uchar>(i-512,j-768);
        }
      }
    }
  }

  imshow("Gamma Corr. & Hist Eq.", all); // Orjinal resmin ekrana basımı
}

Mat q3(Mat img){
  Mat temp = Mat::zeros(img.rows+2, img.cols+2, CV_8UC1); // Geçici matrisi
  Mat out = Mat::zeros(img.rows, img.cols, CV_8UC1); // Çikti matrisi
  Mat t = Mat::zeros(3,3,CV_8UC1);

  // Resmin 3'ün katina büyültülmesi
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      temp.at<uchar>(i+1, j+1) = img.at<uchar>(i, j);
    }
  }

  // Maske yatay ve dikeyde 3'er birim kaydirilarak ilerletiliyor
  for (int x = 1; x < img.rows; x+=3) {
    for (int y = 1; y < img.cols; y+=3) {
      // İslenecek kismin gecici matrise aktarilmasi
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          t.at<uchar>(i+1,j+1) = 0.0f;
          t.at<uchar>(i+1,j+1) = temp.at<uchar>(i+x,j+y);
        }
      }

      // t imgesine histogram esitleme uygulanir ve dönen matris ilgili yere yazilir.
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          temp.at<uchar>(i+x,j+y) = histEqualize(t).at<uchar>(i,j);
        }
      }
    }
  }

  // Büyültülmüs geçici matrisin tekrar daraltilmasi
  for (int i = 0; i < out.rows; i++) {
    for (int j = 0; j < out.cols; j++) {
      out.at<uchar>(i, j) = temp.at<uchar>(i+1, j+1);
    }
  }

  return out;
}

Mat q4(Mat img, float maskSize){
  Mat out = Mat::zeros(img.rows, img.cols, CV_8UC1); // Çıktı matrisi
  Mat temp = Mat::zeros(img.rows + (2*(int)maskSize), img.cols + (2*(int)maskSize), CV_8UC1); // Gecici matris
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
      temp.at<uchar>(i+((int)maskSize/2), j+((int)maskSize/2)) = img.at<uchar>(i, j);
    }
  }

  // Maskenin temp matrisine uygulanmasi
  for(int y = (int)(maskSize)/2; y < temp.rows; y++){ // MaskSize=3 için y=1'den baslar
    for(int x = (int)(maskSize)/2; x < temp.cols; x++){ // MaskSize=3 için x=1'den baslar
      sum = 0.0;
      for(int k = -(int)(maskSize)/2; k <= (int)(maskSize)/2;k++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
        for(int j = -(int)(maskSize)/2; j <=(int)(maskSize)/2; j++){ // MaskSize=3 için y=-1 ile y=+1 araliginda calisir
          sum = sum + mask[j+(int)(maskSize)/2][k+(int)(maskSize)/2]*img.at<uchar>(y - j, x - k); // maskenin ilgili indisini o an seçili bölgenin ilgili indisi ile çarpip toplama ekler
        }
      }
      temp.at<uchar>(y,x) = sum; // Elde edilen toplamin yazilmasi
    }
  }

  // Büyültülmüs geçici matrisin tekrar daraltilmasi
  for (int i = 0; i < out.rows; i++) {
    for (int j = 0; j < out.cols; j++) {
      out.at<uchar>(i, j) = temp.at<uchar>(i+((int)maskSize/2), j+((int)maskSize/2));
    }
  }

  return out;
}

int main(int argc, char** argv ){
  if ( argc != 2 ){
      printf("usage: DisplayImage.out <Image_Path>\n"); // Resim konumu kullanıcıdan alınır.
      return -1;
  }

  Mat img = imread( argv[1], 1 ); // Resim okunur.
  Mat g_img = convertImage(img); // Resim, RGB2GRAY dönüşümü için, convertImage fonksiyonuna gönderilir.

  q1(g_img); // Histogram Specification
  q2(g_img); // Gamma Correction
  Mat lhis = q3(g_img);
  imshow("Local Hist Equ.", lhis);

  Mat uc = q4(g_img, 3.0); // 3x3 mean filtering.
  Mat bes = q4(g_img, 5.0); // 5x5 mean filtering.
  Mat yedi = q4(g_img, 7.0); // 7x7 mean filtering.
  Mat q4 = Mat::zeros(256, 1024, CV_8UC1); // Geçici matrisi

  // 4 resmin tek resmen nakledilmesi
  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 1024; j++) {
      if (j<256) {
        q4.at<uchar>(i,j) = g_img.at<uchar>(i,j);
      } else if (j >= 256 && j < 512) {
        q4.at<uchar>(i,j) = uc.at<uchar>(i,j-256);
      } else if (j >= 512 && j < 768) {
        q4.at<uchar>(i,j) = bes.at<uchar>(i,j-512);
      } else if (j >= 768) {
        q4.at<uchar>(i,j) = yedi.at<uchar>(i,j-768);
      }
    }
  }

  imshow("Mean Filter", q4);

  waitKey();
  return 0;
}
