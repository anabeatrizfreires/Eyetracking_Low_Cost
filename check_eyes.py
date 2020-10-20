import cv2

# entra com as 'haarcascades'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Define a função que fará a detecção 
def detect(gray, frame):
   #Input = frame do vídeo
   #Output = frame com o retangulo demarcado
  
  # detecta os rostos usando o 'haarcascade'
  
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  
  #saídas:
  # x,y => coordenadas detectadas da face
  # width(w) do retangulo no rosto
  # height(h) do retangulo no rosto

  #parametros de entrada:
  # gray é a imagem de entrada (frame)
  # 1.3 é o tamanho do kernel usado para detecção
  # 5 é um valor de robustes na detecção do rosto, diminui os falsos positivos
  
  # Iteração para detecção continuada dos rostos
  for (x,y,w,h) in faces:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
    #parâmetros de entrada: imagem, coordenadas superiores, coordenadas laterais, cor,         #rectangle border thickness
    
    # Definimos duas regioes de interesse(ROI) uma gray e outra colorida, uma para detecção 
    # do olho e outra para traçar o retangulo
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    # Detecção do olho com 'haarcascade'
    eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(25,25))
    # Desenhando o retangulo ao redor dos olhos
    for (ex, ey, ew, eh) in eyes:
       cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
      
  return frame
    
# Capturando video da webcam 
video_capture = cv2.VideoCapture(0)
# loop infinito 
while True:
  # Lê cada frame
  _, frame = video_capture.read()
  # Converte o frame para escala de cinza para poder trabalhar no 'haarcascade'
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Chama a função de detecção para a imagem cinza e o frame capturado 
  canvas = detect(gray, frame)
  # Mostra a imagem 
  cv2.imshow("Video", canvas)
  # Coloca uma condição de parada
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
video_capture.release()
cv2.destroyAllWindows()
