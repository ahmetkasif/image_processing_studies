#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat convertImage(Mat img){
  Mat new_image = Mat(img.rows, img.cols, CV_8UC1); // Çıktı için aynı boyutta yeni bir matris oluşturulur.

  Mat c[3];
  split(img,c); // Girdi resmi kanallarına ayrılıp c dizisine yüklenir.

  // RGB[A] to Gray: Y <- 0.299 * G, + 0.587 * R, + 0.114 * B , Kanallar ilgili değerlerle çarılarak toplanır ve tek kanallı new_image matrisine aktarılır.

  c[0] *= 0.299;
  c[1] *= 0.587;
  c[2] *= 0.114;

  new_image = c[0] + c[1] + c[2];

  return new_image;
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

  for (int i = 0; i < 8; i++) {
    imshow(i + ". Plane", bit_planes[i]); // Bit Planelerin Ekrana Basılması
  }
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n"); // Resim konumu kullanıcıdan alınır.
        return -1;
    }

    Mat image = imread( argv[1], 1 ); // Resim okunur.

    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image); // Orijinal resmin ekrana basımı

    // **---------------RGB2GRAY-------------------**

    Mat g_img = convertImage(image); // Resim, RGB2GRAY dönüşümü için, convertImage fonksiyonuna gönderilir.

    namedWindow("Gray Image", WINDOW_AUTOSIZE);
    imshow("Gray Image", g_img); // Grayscale resim ekrana basılır.

    // **-----------BIT PLANE SLICING--------------**

    sliceBitPlanes(g_img); // Resim Bit Planelerine ayrılması için sliceBitPlanes fonksiyonuna gönderilir.

    // **--------RESOLUTION REDUCING------------**

    Mat lro_img = Mat(g_img.rows, g_img.cols, CV_8UC1); // Resim çözünürlüğünün yarısı boyutunda bir matris oluşturulur.

    int cols = lro_img.cols;
    int rows = lro_img.rows;

    for (int i = 0; i < cols; i+=2) {
      for (int j = 0; j < rows; j+=2) {
        lro_img.at<uchar>(i,j) = g_img.at<uchar>(i, j);
        lro_img.at<uchar>((i)+1,j) = g_img.at<uchar>(i, j);
        lro_img.at<uchar>(i,(j)+1) = g_img.at<uchar>(i, j);
        lro_img.at<uchar>((i)+1,(j)+1) = g_img.at<uchar>(i, j);
      }
    }

    namedWindow("Lower Resolution Image", WINDOW_AUTOSIZE);
    imshow("Lower Resolution Image", lro_img); // Daha düşük çözünürlüklü resim ekrana basılır.

    waitKey();
    return 0;
}
