'''
Nome:   Classe de estimativa de postura
Sobre:  Realiza a estimativa de postura humana, retorna 32 pontos referentes a articulações do corpo humano.
        Desenha na imagem os landmarks e linhas interligando os mesmos.
        Calcula o ângulo formado por três landmarks.
        Compara as coordenadas recebidas com zonas de interesse:
            boca, nariz, topo da cabeça, olhos, ouvidos, tronco, mãos e pés
Desenvolvedor: felipeSperb
'''

import cv2
import mediapipe as mp
import time
import math


class poseDetector():

    def __init__(self, mode=False, complexity=1, smooth=True, detectionCon=0.5, trackCon=0.5):

        '''
        mode:   Se definido como false, a solução trata as imagens de entrada como um fluxo de vídeo.
                Ele tentará detectar a pessoa mais proeminente nas primeiras imagens e,
                após uma detecção bem-sucedida, localizará ainda mais os marcos da pose.
                Em imagens subsequentes, ele simplesmente rastreia esses pontos de referência
                sem invocar outra detecção até que perca o rastreamento, reduzindo a computação e a latência.
                Se definido como true, a detecção de pessoas executa cada imagem de entrada,
                ideal para processar um lote de imagens estáticas, possivelmente não relacionadas.
                Padrão para false.

        complexety: Complexidade do modelo marco postura: 0, 1ou 2.
                    A precisão do ponto de referência, bem como a latência de inferência, geralmente aumentam
                    com a complexidade do modelo.
                    Padrão para 1.

        smooth: Se definido como true, os filtros de solução representam pontos de referência em diferentes imagens
                de entrada para reduzir o jitter, mas são ignorados se static_image_mode também estiver definido como true.
                Padrão para true.

        detectionCon:   Valor de confiança mínimo ( [0.0, 1.0]) do modelo de detecção de pessoa para que
                        a detecção seja considerada bem-sucedida.
                        Padrão para 0.5.

        trackCon:   Valor de confiança mínimo ( [0.0, 1.0]) do modelo de rastreamento de pontos de referência
                    para os pontos de referência de pose a serem considerados rastreados com sucesso, caso contrário,
                    a detecção de pessoa será chamada automaticamente na próxima imagem de entrada.
                    Configurá-lo com um valor mais alto pode aumentar a robustez da solução, às custas de uma
                    latência mais alta. Ignorado se static_image_mode for true, em que a detecção de pessoas
                    simplesmente é executada em todas as imagens.
                    Padrão para 0.5.
        '''
        self.mode = mode
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Função de desenho
        self.mpDraw = mp.solutions.drawing_utils
        # Função de detecção
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        # Converte imagem para RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Realiza a estimativa de postura na imagem
        self.results = self.pose.process(imgRGB)
        # Desenha as linhas na imagem
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img


    def findPosition(self, img, draw=True):
        # Lista os pontos detectados
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Dimenções da imagem
                h, w, c = img.shape
                # Coordenadas dos landmarks
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Adiciona à lista a identificação e as coordenadas
                self.lmList.append([id, cx, cy])
                # Desenha os pontos na imagem
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList


    def findAngle(self, img, p1, p2, p3, draw=False):

        # define as coordenadas dos landmarks recebidos
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calcula o angulo entre os três pontos de entrada
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        # Se draw=True, desenha os pontos na imagem e o resultado do calculo
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 +50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        return angle


    def comparar(self, img, x, y, w, h, classIds):

        # Se máscara:
        if classIds == 0:
            x0, y0 = self.lmList[0][1:]     # nariz
            x9, y9 = self.lmList[9][1:]     # canto esquerdo da boca
            x10, y10 = self.lmList[10][1:]  # canto direito da boca
            if (x+w) >= x0 >= x and (y+h) >= y0 >= y:
                if (x+w) >= x9 >= x and (y+h) >= y9 >= y:
                    if (x+w) >= x10 >= x and (y+h) >= y10 >= y:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0

        # Se capacete:
        elif classIds == 1:
            x0, y0 = self.lmList[0][1:]     # Nariz
            if (x+w) >= x0 >= x and 2*(y+h) >= y0 >= y:
                return 1
            else:
                return 0

        # Se óculos
        elif classIds == 2:
            x2, y2 = self.lmList[2][1:]     # Olho esquerdo
            x5, y5 = self.lmList[5][1:]     # OLho direito

            if (x+w) >= x2 >= x and (y+h) >= y2 >= y:
                if (x + w) >= x5 >= x and (y + h) >= y5 >= y:
                    return 1
                else:
                    return 0
            else:
                return 0

        # Se abafador
        elif classIds == 3:
            x7, y7 = self.lmList[7][1:]  # Orelha esquerda
            x8, y8 = self.lmList[8][1:]  # Orelha direito

            if (x + w) >= x7 >= x and (y + h) >= y7 >= y:
                if (x + w) >= x8 >= x and (y + h) >= y8 >= y:
                    return 1
                else:
                    return 0
            else:
                return 0

        # Se colete
        elif classIds == 4:
            x12, y12 = self.lmList[12][1:]  # Ombro direito
            x11, y11 = self.lmList[11][1:]  # Ombro esquerdo
            x24, y24 = self.lmList[24][1:]  # Cintura direita
            x23, y23 = self.lmList[23][1:]  # Cintura esquerda
            if (x + w) >= x12 >= x and (y + h) >= y12 >= y:
                if (x + w) >= x11 >= x and (y + h) >= y11 >= y:
                    if (x + w) >= x24 >= x and (y + h) >= y24 >= y:
                        if (x + w) >= x23 >= x and (y + h) >= y23 >= y:
                            return 1
                        else:
                            return 0
                    else:
                        return 0
                else:
                    return 0
            else:
                return 0

        # Se luva
        elif classIds == 5:
            x16, y16 = self.lmList[16][1:]  # Pulso direito
            x20, y20 = self.lmList[20][1:]  # Indicador direito
            x15, y15 = self.lmList[15][1:]  # Pulso esquerdo
            x19, y19 = self.lmList[19][1:]  # Indicador esquerda

            if (x + w) >= x16 >= x and (y + h) >= y16 >= y and (x + w) >= x20 >= x and (y + h) >= y20 >= y:
                # Retorna mão direita
                return 2
            elif (x + w) >= x15 >= x and (y + h) >= y15 >= y and (x + w) >= x19 >= x and (y + h) >= y19 >= y:
                # Retorna mão esquerda
                return 3
            else:
                return 0

        # Se bota
        elif classIds == 6:
            x30, y30 = self.lmList[28][1:]  # Calcanhar direito
            x32, y32 = self.lmList[32][1:]  # Ponta do pé direito
            x29, y29 = self.lmList[27][1:]  # Calcanhar esquerdo
            x31, y31 = self.lmList[31][1:]  # Ponta do pé esquerda
            if (x + w) >= x30 >= x and (y + h) >= y30 >= y and (x + w) >= x32 >= x and (y + h) >= y32 >= y:
                # Retorna mão direita
                return 2
            elif (x + w) >= x29 >= x and (y + h) >= y29 >= y and (x + w) >= x31 >= x and (y + h) >= y31 >= y:
                # Retorna mão esquerda
                return 3
            else:
                return 0


# Teste de classe
def main():
    cap = cv2.VideoCapture("teste.mp4") # coloque na pasta principal um vídeo de seu interesse com o nome teste.mp4
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

        cv2.imshow("Imagem", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()