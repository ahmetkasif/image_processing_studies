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
  int hist[256] = {0}; // Histogram verisinin tutulacağı dizi
  float temp = 0.0f; // En yüksek olasılığın tutulduğu temp değişkeni

  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      hist[(int)img.at<uchar>(i, j)]++;  // Tüm matris dolaşılarak histogram verisi toplanır
    }
  }

  for (int i = 0; i < 256; i++) {
    if (hist[i] > temp) {
      temp = hist[i]; // Temp değişkeni en yüksek olasılığı tutar
    }
  }

  for (int i = 0; i < 256; i++) {
    hist[i] = hist[i] / temp * 256.0f; // Tüm değerler, 0-256 arasında bir değere aktarılır
  }

  Mat o_img = Mat::zeros(256, 256, CV_8UC1); // Tek kanallı histogram matrisi

  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 2; j++) {
      o_img.at<uchar>(255-j, i) = 255; // Grafiğin altında 2 piksellik bir dolgu oluşturuluyor
    }
  }
  for (int j = 0; j < 256; j++) { // Her parlaklık değeri için,
    for (int k = 2; k < hist[j] + 2; k++) { // Değer büyüklüğü uyarınca 2'den başlayarak
      o_img.at<uchar>(256-k, j) = 255; // Pikseller beyaza boyanır
    }
  }

  return o_img;
}

Mat gammaCorrection(Mat img, float fGamma){
  unsigned char lut[256];
	for (int i = 0; i < 256; i++){
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), 1/fGamma) * 255.0f); // Lut dizisinin oluşturulması (i/255)^(1/fGamma)
	}
  Mat o_img = Mat(img.rows, img.cols, img.type()); // Çıktı için aynı boyutta yeni bir matris oluşturulur.

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      for (int k = 0; k < 3; k++) {
        o_img.at<Vec3b>(i, j).val[k] = lut[(img.at<Vec3b>(i,j).val[k])]; // Çıktı resmindeki her pikselin r,g,b kanallarına ayrı ayrı lut katsayısı uygulanır.
      }
    }
  }

  return o_img;
}

Mat plotGammaHist(float fGamma){
  unsigned char lut[256];
  for (int i = 0; i < 256; i++){
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), 1/fGamma) * 255.0f); // Lut dizisinin oluşturulması (i/255)^(1/fGamma)
	}

  Mat luts = Mat::zeros(256, 256, CV_8UC1); // Tek kanallı histogram matrisi

  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 2; j++) {
      luts.at<uchar>(255-j, i) = 255; // Grafiğin altında 2 piksellik bir dolgu oluşturuluyor
    }
  }
  for (int j = 0; j < 256; j++) { // Her parlaklık değeri için,
    for (int k = 2; k < lut[j] + 2; k++) { // Değer büyüklüğü uyarınca 2'den başlayarak
      luts.at<uchar>(256-k, j) = 255; // Pikseller beyaza boyanır
    }
  }

  return luts;
}

void sliceBitPlanes(Mat img){
  int32_t rows(img.rows), cols(img.cols); // Görüntünün satır ve sütun sayısının hafızaya alımı
  Mat bit_planes[8]; // Elde edilecek bit planelerin saklanacağı dizi oluşturulur.
  for (int i = 0; i < 8; i++) {
    bit_planes[i] = Mat(img.rows, img.cols, CV_8UC1); // Dizideki matrisler tek kanallı resim matrisleri olarak ilişkilendirilir.
  }

  Mat bit_mask(Mat::ones(rows, cols, CV_8UC1)); // Resimle aynı boyutta tek kanallı maske matrisinin oluşturulması.
  Mat t_image(img.clone()); // Geçici resim matrisi oluşturulması

  for (int i = 0; i < 8; ++i) {
    Mat out; // Çıktı matrisinin hafızadaki yerinin hazırlanması
    cv::bitwise_and(t_image, bit_mask, out);  // resmi ve daha önceden hazırladığımız değerleri 1 olan matrisi Bitwise and işlemine tabi tutmamız, çıktının out matrisine yüklenmesi

    bit_planes[i] = out * 255; // out matrisinin 255 ile skaler çarpılarak 1 olan değerlerin 255 yapılması
    t_image /= 2; // Geçici matris değerlerinin ikiye bölünmesi
  }

  // Bit katmanlarının Ekrana Basılması
  imshow("Katman 0", bit_planes[0]);
  imshow("Katman 1", bit_planes[1]);
  imshow("Katman 2", bit_planes[2]);
  imshow("Katman 3", bit_planes[3]);
  imshow("Katman 4", bit_planes[4]);
  imshow("Katman 5", bit_planes[5]);
  imshow("Katman 6", bit_planes[6]);
  imshow("Katman 7", bit_planes[7]);
}

void histEqualize(Mat img){
  Mat o_img = Mat::zeros(256, 256, CV_8UC1); // Tek kanallı histogram matrisi

  // generate histogram
  double hist[256] = {0};
  for (int i = 0; i < img.cols; i++) {
    for (int j = 0; j < img.rows; j++) {
      hist[(int)img.at<uchar>(i, j)] += 1.0;  // Tüm matris dolaşılarak histogram verisi toplanır
    }
  }

  double pmf[256]; // probability mass function dizisi
  for (int i = 0; i < 256; i++) {
    pmf[i] = hist[i] / (img.rows * img.cols);  //  probability mass function dizisi hesaplanır
  }

  double cdf[256]; // cumulative distribution function dizisi
  cdf[0] = pmf[0]; // cdf dizisinin ilk elemanı pdf'ten alınır
  for(int i = 1; i < 256; i++){
    cdf[i] = pmf[i] + cdf[i-1]; // cdf dizisinin diğer elemanları hesaplanır
  }

  for (int i = 0; i < 256; i++) {
    cdf[i] =  (int)(cdf[i]*255); // cdf değerleri 0-255 tamsayı değerleri arasına map ediliyor
  }

  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
      o_img.at<uchar>(i,j) = cdf[(int)img.at<uchar>(i,j)]; // resmin piksellerinin parlaklık değerleri, cdf fonksiyonundaki parlaklık değerleriyle değiştirilir.
      hist[(int)img.at<uchar>(i, j)] += 1.0;  // Tüm matris dolaşılarak histogram verisi toplanır
    }
  }

  imshow("Histogram Equalized Image", o_img); // Orijinal resmin ekrana basımı

  Mat his_o_img = plotHistogram(o_img); // Orijinal histogramın hesaplanıp, matrise yüklenmesi
  imshow("His Equalized's Histogram", his_o_img); // Orijinal histogramın ekrana basılması
}

int main(int argc, char** argv ){
  if ( argc != 2 ){
      printf("usage: DisplayImage.out <Image_Path>\n"); // Resim konumu kullanıcıdan alınır.
      return -1;
  }

  Mat img = imread( argv[1], 1 ); // Resim okunur.
  imshow("Original Image", img); // Orijinal resmin ekrana basımı

  // **---------------RGB2GRAY-------------------**

  Mat g_img = convertImage(img); // Resim, RGB2GRAY dönüşümü için, convertImage fonksiyonuna gönderilir.
  imshow("Grayscale Image", g_img); // Grayscale resim ekrana basılır.

  histEqualize(g_img);

  Mat his_o = plotHistogram(g_img); // Orijinal histogramın hesaplanıp, matrise yüklenmesi
  imshow("Original His", his_o); // Orijinal histogramın ekrana basılması

  // **--------------GAMMA CORRECTION----------------**

  /* 0.5f Gamma */

  Mat gamma_0_5 = gammaCorrection(img, 0.5f); // Resme gamma değerinin uygulanması
  imshow("0.5f Gamma Image", gamma_0_5); // 0.5 Gamma uygulanmış imge

  Mat gamma_0_5_curve = plotGammaHist(0.5f); // 0.5f gammanın eğrisinin çizimi
  imshow("0.5f Gamma Curve", gamma_0_5_curve); // 0.5 Gamma eğrisi

  Mat his_0_5 = plotHistogram(gamma_0_5); // 0.5 Histogramın hesaplanıp, matrise yüklenmesi
  imshow("0.5f Gamma Applied His", his_0_5); // 0.5 Gamma uygulanmış imgenin histogramı

  /* 0.75f Gamma */

  Mat gamma_0_75 = gammaCorrection(img, 0.75f); // Resme gamma değerinin uygulanması
  imshow("0.75 Gamma Image", gamma_0_75); // 0.75 Gamma uygulanmış imge

  Mat gamma_0_75_curve = plotGammaHist(0.75f); // 0.75f gammanın eğrisinin çizimi
  imshow("0.75f Gamma Curve", gamma_0_75_curve); // 0.75 Gamma eğrisi

  Mat his_0_75 = plotHistogram(gamma_0_75); // 0.75 Histogramın hesaplanıp, matrise yüklenmesi
  imshow("0.75f Gamma Applied His", his_0_75); // 0.75 Gamma uygulanmış imgenin histogramı

  /* 1.0f Gamma */

  Mat gamma_1 = gammaCorrection(img, 1.0f); // Resme gamma değerinin uygulanması
  imshow("1.0f Gamma Image", gamma_1); // 1.0 Gamma uygulanmış imge

  Mat gamma_1_0_curve = plotGammaHist(1.0f); // 1.0f gammanın eğrisinin çizimi
  imshow("1.0f Gamma Curve", gamma_1_0_curve); // 1.0 Gamma eğrisi

  Mat his_1 = plotHistogram(gamma_1); // 1.0 Histogramın hesaplanıp, matrise yüklenmesi
  imshow("1.0f Gamma Applied His", his_1); // 1.0 Gamma uygulanmış imgenin histogramı

  /* 1.5f Gamma */

  Mat gamma_1_5 = gammaCorrection(img, 1.5f); // Resme gamma değerinin uygulanması
  imshow("1.5f Gamma Image", gamma_1_5); // 1.5f Gamma uygulanmış imge

  Mat gamma_1_5_curve = plotGammaHist(1.5f); // 1.5f gammanın eğrisinin çizimi
  imshow("1.5f Gamma Curve", gamma_1_5_curve); // 1.5 Gamma eğrisi

  Mat his_1_5 = plotHistogram(gamma_1_5); // 1.5f Histogramın hesaplanıp, matrise yüklenmesi
  imshow("1.5f Gamma Applied His", his_1_5); // 1.5f Gamma uygulanmış imgenin histogramı

  /* 2.0f Gamma */

  Mat gamma_2_0 = gammaCorrection(img, 2.0f); // Resme gamma değerinin uygulanması
  imshow("2.0f Gamma Image", gamma_2_0); // 2.0f Gamma uygulanmış imge


  Mat his_2_0 = plotHistogram(gamma_2_0); // 2.0f Histogramın hesaplanıp, matrise yüklenmesi
  imshow("2.0f Gamma Applied His", his_2_0); // 2.0f Gamma uygulanmış imgenin histogramı


  // **-----------BIT PLANE SLICING--------------**

  sliceBitPlanes(g_img); // Resim Bit Planelerine ayrılması için sliceBitPlanes fonksiyonuna gönderilir.
  waitKey();
  return 0;
}
