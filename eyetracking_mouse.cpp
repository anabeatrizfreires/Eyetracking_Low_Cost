#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>

// dentro do círculo encontrado pelo Hough desenha o círculo mais preto
cv::Vec3f Circulo_Mais_Escuro(cv::Mat &eye, std::vector<cv::Vec3f> &circles)
{
  std::vector<int> sums(circles.size(), 0);
  for (int y = 0; y < eye.rows; y++) //associa linha com olhos
  {
      uchar *ptr = eye.ptr<uchar>(y);
      for (int x = 0; x < eye.cols; x++) // associa coluna com olhos
      {
          int value = static_cast<int>(*ptr);
          for (int i = 0; i < circles.size(); i++) //pega os dados da função HoughCircles p desenhar o circulo no circulo mais preto
          {
              cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
              int radius = (int)std::round(circles[i][2]);
              if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
              {
                  sums[i] += value;
              }
          }
          ++ptr;
      }
  }
  int smallestSum = 9999999;
  int smallestSumIndex = -1;
  for (int i = 0; i < circles.size(); i++)
  {
      if (sums[i] < smallestSum)
      {
          smallestSum = sums[i];
          smallestSumIndex = i;
      }
  }
  return circles[smallestSumIndex];
}
// Determina qual olho controla cursor, no caso o esquerdo
cv::Rect Seleciona_Olho_Esquerdo(std::vector<cv::Rect> &eyes)
{
  int leftmost = 99999999;
  int leftmostIndex = -1;
  for (int i = 0; i < eyes.size(); i++)
  {
      if (eyes[i].tl().x < leftmost) //analisa em x p encontrar o olho posicionado mais a esquerda
      {
          leftmost = eyes[i].tl().x;
          leftmostIndex = i; //pixel a pixel até definir o esquerdo
      }
  }
  return eyes[leftmostIndex];
}

std::vector<cv::Point> centers;
cv::Point lastPoint;
cv::Point mousePoint;

//Para normalizar e estabilizar o número de frames utilizados
cv::Point Filtro_de_Media(std::vector<cv::Point> &points, int windowSize)
{
  float sumX = 0;
  float sumY = 0;
  int count = 0;
  for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
  {
      sumX += points[i].x; //analisa em x e em y
      sumY += points[i].y;
      ++count;
  }
  if (count > 0)
  {
      sumX /= count; //faz uma media simples para estabilizar
      sumY /= count;
  }
  return cv::Point(sumX, sumY);
}

//Função que utiliza haar cascades para fazer o reconhecimento de faces e depois de olhos
void Detecta_Olhos(cv::Mat &frame, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade)
{
  cv::Mat grayscale;
  cv::cvtColor(frame, grayscale, CV_BGR2GRAY); // converte a imagem p escalas de cinza
  cv::equalizeHist(grayscale, grayscale); // equaliza o histograma 
  
  std::vector<cv::Rect> faces; //armazena no vetor faces
  faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(150, 150)); //reconhecendo a face
  if (faces.size() == 0) return; 
  cv::Mat face = grayscale(faces[0]); 
  
  std::vector<cv::Rect> eyes; //armazena no vetor olhos
  eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30)); // reconhecendo os olhos    
  rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2); //se face encontrada desenha retangulo
  if (eyes.size() != 2) return; 
  for (cv::Rect &eye : eyes)
  {
      rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2); //se olhos encontrados desenha retangulo
  }
  cv::Rect eyeRect = Seleciona_Olho_Esquerdo(eyes);
  cv::Mat eye = face(eyeRect); 
  cv::equalizeHist(eye, eye); //equaliza histograma do olho recortado
  std::vector<cv::Vec3f> circles; //Função para Hough Circles
  
  cv::HoughCircles(eye, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
  if (circles.size() > 0)
  {
      cv::Vec3f eyeball = Circulo_Mais_Escuro(eye, circles); //Guarda no vetor o circulo mais escuro do olho
      cv::Point center(eyeball[0], eyeball[1]);
      centers.push_back(center);
      center = Filtro_de_Media(centers, 5);
      if (centers.size() > 1)
      {
          cv::Point diff;
          diff.x = (center.x - lastPoint.x) * 20;
          diff.y = (center.y - lastPoint.y) * -30;
          mousePoint += diff;
      }
      lastPoint = center;
      int radius = (int)eyeball[2];
      cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, cv::Scalar(0, 0, 255), 2);
      cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2);
  }
  cv::imshow("Eye", eye);
}

//Função que mexe o mouse a partir da posição do olho, utilizando xdotool
void Move_Mouse(cv::Mat &frame, cv::Point &location)
{
  if (location.x > frame.cols) location.x = frame.cols;
  if (location.x < 0) location.x = 0;
  if (location.y > frame.rows) location.y = frame.rows;
  if (location.y < 0) location.y = 0;
  system(("xdotool mousemove " + std::to_string(location.x) + " " + std::to_string(location.y)).c_str());
}
 //main do código
int main(int argc, char **argv)
{
  /*if (argc != 2)
  {
      std::cerr << "Usage: EyeDetector <WEBCAM_INDEX>" << std::endl;
      return -1;
  }*/
  cv::CascadeClassifier faceCascade; //carrega os haar
  cv::CascadeClassifier eyeCascade;
  if (!faceCascade.load("haarcascades/haarcascade_frontalface_alt.xml"))
  {
      std::cerr << "Haar Cascade Face Nao Encontrado" << std::endl;
      return -1;
  }    
  if (!eyeCascade.load("haarcascades/haarcascade_eye.xml"))
  {
      std::cerr << "Haar Cascade Olhos Nao Encontrado" << std::endl;
      return -1;
  }
  cv::VideoCapture cap(0); // Chama e nomeia camera 
 cap.open(0);// 
  if (!cap.isOpened())
  {
      std::cerr << "Webcam Nao Encontrada" << std::endl;
      return -1;
  }    
  cv::Mat frame; //Nomeia entrada da camera
  mousePoint = cv::Point(800, 800);
  while (1)
  {
      cap >> frame; // saida a partir da entrada
      if (!frame.data) break;
      Detecta_Olhos(frame, faceCascade, eyeCascade); //chama função a partir das variaveis
      Move_Mouse(frame, mousePoint); //chama função a partir das variaveis
      cv::imshow("Finish", frame); // saida da camera
      if (cv::waitKey(30) >= 0) break;  //para fechar pressione qualquer botao
  return 0;
}

