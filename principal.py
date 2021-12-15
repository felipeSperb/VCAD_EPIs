'''
Nome:   Visão Computacional Aplicada na Detecção de Equipamentos de Proteção Individual (VCAD_EPIs).
Sobre:  Este programa realiza a detecção de equipamentos de proteção individual em um cena.
        O objetivo é simular um software de acesso a uma área restrita.
        Os EPIs detectavéis são: máscara, capacete, óculos, abafador, colete, luvas e botas.
        A detecção só será realizada a partir do momento que a pose de inspeção seja realiza por 3 segundos.
        A posição do EPI detectado será comparada com a região de interesse do corpo.
        A tomada de decisão dependerá que o objeto seja detectado e que coincida com a região de interesse.
        Ambas luvas e botas necessitam serem detectadas para liberação do acesso.
        É possível escolhar quais EPIs serão levados em conta na tomada de decisão.
        Também é possível acessar o histórico de detecções.
        O programa salva as imagens junto de arquivos .txt com as coordenadas da detecção.
        Esses arquivos podem ser utilizados para treinar o modelo no futuro.
Desenvolvedor: felipeSperb
'''

# ---------------- IMPORTAR BIBLIOTECAS ------------------- #

from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np
import time
import cvzone


# ------------------ IMPORTAR CLASSES ---------------------- #

import estimativa_de_postura as ep

# Ativação classe de estimativa de postura
pose = ep.poseDetector()

# ---------------------- ARQUIVOS -------------------------- #

# Endereço de salvamento padrão
myPath = 'Arquivos/'

# Endereço salvamento Imagens
myImagensPositivas = myPath + "Imagens_Registradas/Positivas/"
myImagensNegativas = myPath + "Imagens_Registradas/Negativas/"

# Endereço de Imagem Pose Menu
myPose = myPath + "/Icones/pose.png"
myMenu = myPath + "/Icones/fundoMenu.png"

# Endereço de arquivos de imagem das classes (Ícones)
myIcones = [
    "Arquivos/Icones/Mascara.png",
    "Arquivos/Icones/Capacete.png",
    "Arquivos/Icones/Oculos.png",
    "Arquivos/Icones/Abafador.png",
    "Arquivos/Icones/Colete.png",
    "Arquivos/Icones/Luva.png",
    "Arquivos/Icones/Bota.png"
]

myIconesPositivos = [
    "Arquivos/Icones/MascaraPositivo.png",
    "Arquivos/Icones/CapacetePositivo.png",
    "Arquivos/Icones/OculosPositivo.png",
    "Arquivos/Icones/AbafadorPositivo.png",
    "Arquivos/Icones/ColetePositivo.png",
    "Arquivos/Icones/LuvaPositivo.png",
    "Arquivos/Icones/BotaPositivo.png"
]

myIconesNegativos = [
    "Arquivos/Icones/MascaraNegativo.png",
    "Arquivos/Icones/CapaceteNegativo.png",
    "Arquivos/Icones/OculosNegativo.png",
    "Arquivos/Icones/AbafadorNegativo.png",
    "Arquivos/Icones/ColeteNegativo.png",
    "Arquivos/Icones/LuvaNegativo.png",
    "Arquivos/Icones/BotaNegativo.png"
]

myIconesAlerta = [
    "Arquivos/Icones/MascaraAlerta.png",
    "Arquivos/Icones/CapaceteAlerta.png",
    "Arquivos/Icones/OculosAlerta.png",
    "Arquivos/Icones/AbafadorAlerta.png",
    "Arquivos/Icones/ColeteAlerta.png",
    "Arquivos/Icones/LuvaAlerta.png",
    "Arquivos/Icones/BotaAlerta.png"
]

myIconesNeutro = [
    "Arquivos/Icones/MascaraNeutro.png",
    "Arquivos/Icones/CapaceteNeutro.png",
    "Arquivos/Icones/OculosNeutro.png",
    "Arquivos/Icones/AbafadorNeutro.png",
    "Arquivos/Icones/ColeteNeutro.png",
    "Arquivos/Icones/LuvaNeutro.png",
    "Arquivos/Icones/BotaNeutro.png"
]

# Endereço de histórico
hist_path = "Arquivos/Imagens_Registradas"

# Abrir e ler arquivos de classes de objetos
classesFile = 'YOLOv4/epi.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Arquivo com arquitetura YOLOv4 modificada
modelConfiguration = "YOLOv4/yolov4-epi.cfg"
# Arquivo de pesos treinados
modelWeights = "YOLOv4/yolov4-epi360_3200.weights"


# ------------ CONFIGURAÇÃO DE BACKEND ------------ #

# Configurar framework darknet como backend usando openCV
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# Configurar opencv como backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# Configurar cpu
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# ----------------- VARIAVEIS GLOBAIS ---------------------- #

# Proporções da imagem de entrada da CNN
whT = 416
# Confiança mínima da rede
confThreshold = 0.9
# Supressão não máxima. Limite de IOU
nmsThreshold = 0.3

# Variáveis de contagem
t = 0
espera = 0
tempoDetect = 0

# Variáveis de configuração. O objeto só será detectado quando igual a 1
chMascara = 1
chCapacete = 1
chOculos = 1
chAbafador = 1
chColete = 1
chLuva = 1
chBota = 1


# ------------------- INICIAR HARDWARES -------------------- #

# Ativar câmera padrão
cap = cv2.VideoCapture(0)


# ------------------- DECLARAR FUNÇÕES --------------------- #

'''
Função de detecção de objetos:
    Recebe o frame a ser analisado, realiza as operações previstas pela CNN, realiza as detecções levando em 
    consideração os índices de confiança mínima e de limite de IOU, salva o frame junto de arquivo .txt com as 
    coordenadas da detecção, compara as coordenadas espaciais da detecção com a zona de interesse do corpo,
    desenha as caixas delimitadoras dos objetos na imagem e realiza a tomada de decisão. 
'''
def encontrarEPI(frame):

    # Arrays de detecção
    bbox = []
    classIds = []
    confs = []
    posicao = []
    pos = [0, 0, 0, 0, 0, 0, 0]

    # Contagem de alertas
    alert = 0

    # Variável aux para colar ícones em miniatura de imagem
    deslocaIcon = 0

    # Recebe tempo real
    relogio = time.localtime()

    # Status dos objetos
    global chMascara
    global chCapacete
    global chOculos
    global chAbafador
    global chColete
    global chLuva
    global chBota

    # Variáveis auxiliares na detecção de luvas e botas
    compLuvaDir = 0
    compLuvaEsq = 0
    compBotaDir = 0
    compBotaEsq = 0

    # Converte imagem em um objeto BLOB
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    # Define blob como entrada da rede
    net.setInput(blob)
    # Estrutura da rede treinada
    layerNames = net.getLayerNames()
    # Camadas da rede não conectadas
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # Retorna lista de objetos detectados
    outputs = net.forward(outputNames)

    # Retorna as dimenções da imagem
    hT, wT, cT = frame.shape

    # Para cada detecção
    for output in outputs:
        for det in output:
            # Armazena a confiança correspondente a cada objeto
            scores = det[5:]
            # Índice correspondente a classe com maior confiança
            classId = np.argmax(scores)
            # Valor de confiança referente a classe
            confidence = scores[classId]

            # Se confiança maior que confiança minima:
            if confidence > confThreshold:
                # Adiciona as coordenadas a lista posição
                posicao.append([det[0], det[1], det[2], det[3]])
                # Converte as coordenadas para as proporções da imagem
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                # Adiciona as coordenadas convertidas à lista bbox
                bbox.append([x, y, w, h])
                # Adiciona o índice correspondente a classe na lista classIds
                # 0 = mascara, 1 = capacete, 2 = óculos, 3 = abafador, 4 = colete, 5 = luva, 6 = bota.
                classIds.append(classId)
                # Adiciona o valor de confiança da classe a lista confs
                confs.append(float(confidence))

    # Executa supressão não máxima
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    # Caso houver detecção
    if classIds:
        # Salvar cópia de imagem na pasta de positivos
        cv2.imwrite(myImagensPositivas + str(relogio.tm_year) + "-" + str(relogio.tm_mon) + "-" +
                    str(relogio.tm_mday) + "_" + str(relogio.tm_hour) + "-" +
                    str(relogio.tm_min) + "-" + str(relogio.tm_sec) + "_Positivo" + ".png", frame)
        # Cria arquivo de marcação
        arquivo = open(myImagensPositivas + str(relogio.tm_year) + "-" + str(relogio.tm_mon) + "-" +
                       str(relogio.tm_mday) + "_" + str(relogio.tm_hour) + "-" +
                       str(relogio.tm_min) + "-" + str(relogio.tm_sec) + "_Positivo" + '.txt', 'a')

        # Para cada uma das detecções
        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Comparar Objeto com a região de interesse
            comp = pose.comparar(frame, x, y, w, h, classIds[i])

            '''
            Os EPIs detectados que coincidirem com a região de interesse serão marcados com a cor verde.
            Os EPIs detectados que NÃO coincidirem serão marcados de amarelo.
            As detecções de luvas e botas serão marcadas na imagem, mas a detecção só será completa se os membros direito e esquerdo forem detectados. 
            '''
            if comp == 1:
                # Sucesso na comparação com zona de interesse
                img2 = PhotoImage(file=myIconesPositivos[classIds[i]])
                corBox = (0, 255, 0)
            elif comp == 2:
                # Sucesso na comparação de um EPI localizado em um membro direito
                img2 = PhotoImage(file=myIconesPositivos[classIds[i]])
                corBox = (0, 255, 0)
                if classIds[i] == 5:
                    compLuvaDir += 1
                elif classIds[i] == 6:
                    compBotaDir += 1
            elif comp == 3:
                # Sucesso na comparação de um EPI localizado em um membro esquerdo
                img2 = PhotoImage(file=myIconesPositivos[classIds[i]])
                corBox = (0, 255, 0)
                if classIds[i] == 5:
                    compLuvaEsq += 1
                elif classIds[i] == 6:
                    compBotaEsq += 1
            else:
                # Inssucesso na comparação com zona de interesse
                img2 = PhotoImage(file=myIconesAlerta[classIds[i]])
                corBox = (0, 255, 255)
                alert += 1

            # Desenhar caixa delimitadora na imagem
            cv2.rectangle(frame, (x, y), (x + w, y + h), corBox, 1)

            # Escrever rótulo na caixa
            cv2.putText(frame, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, corBox, 1)

            # Escrever em arquivo de coordenadas
            arquivo.write(str(classIds[i]) + " " + str(posicao[i][0]) + " " + str(posicao[i][1]) + " " + str(posicao[i][2]) + " " + str(posicao[i][3]) + "\n")

            # Incerir icone do objeto na imagem miniatura
            iconePositivo = cv2.imread(myIconesPositivos[classIds[i]], cv2.IMREAD_UNCHANGED)
            hf, wf, cf = iconePositivo.shape
            hb, wb, cb = frame.shape
            frame = cvzone.overlayPNG(frame, iconePositivo, [0 + deslocaIcon, hb - hf])
            deslocaIcon += 75

            # Substituir icone e escrever a confiança da detecção no menu
            if classIds[i] == 0 and chMascara == 1:
                lblIcone0.configure(image=img2)
                lblIcone0.image =img2
                lblPerIcone0.configure(text=f'{int(confs[i] * 100)}%')
                pos[0] = 1
            elif classIds[i] == 1 and chCapacete == 1:
                lblIcone1.configure(image=img2)
                lblIcone1.image = img2
                lblPerIcone1.configure(text=f'{int(confs[i] * 100)}%')
                pos[1] = 1
            elif classIds[i] == 2 and chOculos == 1:
                lblIcone2.configure(image=img2)
                lblIcone2.image = img2
                lblPerIcone2.configure(text=f'{int(confs[i] * 100)}%')
                pos[2] = 1
            elif classIds[i] == 3 and chAbafador == 1:
                lblIcone3.configure(image=img2)
                lblIcone3.image = img2
                lblPerIcone3.configure(text=f'{int(confs[i] * 100)}%')
                pos[3] = 1
            elif classIds[i] == 4 and chColete == 1:
                lblIcone4.configure(image=img2)
                lblIcone4.image = img2
                lblPerIcone4.configure(text=f'{int(confs[i] * 100)}%')
                pos[4] = 1
            elif classIds[i] == 5 and chLuva == 1:
                if compLuvaDir != 0 and compLuvaEsq != 0:
                    lblIcone5.configure(image=img2)
                    lblIcone5.image = img2
                    lblPerIcone5.configure(text=f'{int(confs[i] * 100)}%')
                    pos[5] = 1
            elif classIds[i]==6 and chBota == 1:
                if compBotaDir != 0 and compBotaEsq != 0:
                    lblIcone6.configure(image=img2)
                    lblIcone6.image = img2
                    lblPerIcone6.configure(text=f'{int(confs[i] * 100)}%')
                    pos[6] = 1

        # Fechar arquivo
        arquivo.close()

    # Se não houver detecção, salva imagem na pasta de negativos com arquivo txt de mesmo nome
    else:
        cv2.imwrite(myImagensNegativas + str(relogio.tm_year) + "-" + str(relogio.tm_mon) + "-" +
                    str(relogio.tm_mday) + "_" + str(relogio.tm_hour) + "-" +
                    str(relogio.tm_min) + "-" + str(relogio.tm_sec) + "_Negativo" + ".png", frame)

        arquivo = open(myImagensNegativas + str(relogio.tm_year) + "-" + str(relogio.tm_mon) + "-" +
                       str(relogio.tm_mday) + "_" + str(relogio.tm_hour) + "-" +
                       str(relogio.tm_min) + "-" + str(relogio.tm_sec) + "_Negativo" + '.txt', 'a')


    # Imprime miniatura da detecção no menu
    frame = imutils.resize(frame, width=350)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im2 = Image.fromarray(frame)
    img2 = ImageTk.PhotoImage(image=im2)
    lblDeteccao.configure(image=img2)
    lblDeteccao.image = img2

    # Substitui icones dos objetos não detectados
    if pos[0] != 1 and chMascara == 1:
        img3 = PhotoImage(file=myIconesNegativos[0])
        lblIcone0.configure(image=img3)
        lblIcone0.image = img3
    if pos[1] != 1 and chCapacete == 1:
        img3 = PhotoImage(file=myIconesNegativos[1])
        lblIcone1.configure(image=img3)
        lblIcone1.image = img3
    if pos[2] != 1 and chOculos == 1:
        img3 = PhotoImage(file=myIconesNegativos[2])
        lblIcone2.configure(image=img3)
        lblIcone2.image = img3
    if pos[3] != 1 and chAbafador == 1:
        img3 = PhotoImage(file=myIconesNegativos[3])
        lblIcone3.configure(image=img3)
        lblIcone3.image = img3
    if pos[4] != 1 and chColete == 1:
        img3 = PhotoImage(file=myIconesNegativos[4])
        lblIcone4.configure(image=img3)
        lblIcone4.image = img3
    if pos[5] != 1 and chLuva == 1:
        img3 = PhotoImage(file=myIconesNegativos[5])
        lblIcone5.configure(image=img3)
        lblIcone5.image = img3
    if pos[6] != 1 and chBota == 1:
        img3 = PhotoImage(file=myIconesNegativos[6])
        lblIcone6.configure(image=img3)
        lblIcone6.image = img3

    # Compara detecções com expectativas
    detectObj = 0
    Obj = chMascara + chCapacete + chOculos + chColete + chAbafador + chLuva + chBota
    detectObj = pos[0] + pos[1] + pos[2] + pos[3] + pos[4] + pos[5] + pos[6]

    # TOMADA DE DECISÃO
    if Obj == detectObj and alert == 0:
        btnAcesso.configure(text="ACESSO LIBERADO", bg="green")
    elif alert > 0:
        btnAcesso.configure(text="EPI MAL POSICIONADO", bg="yellow")
    else:
        btnAcesso.configure(text="ACESSO NEGADO", bg="red")


'''
Função de Estimativa de Postura Humana:
    Detecta a presença e uma pessoa e realiza a estimativa de postura.
    A detecção será realizada somente na pessoa mais bem posicionada na imagem.
    Se a pessoa permanecer na postura de inspeção por 3 segundos, o frame será enviado para CNN.
    O menu é restaurado após 30 segundos da última detecção.
'''
def detectPostura (frame):

    # Variáveis de contagem
    global t
    global espera
    global tempoDetect
    tempo = time.time()

    # Chama classe de estimativa de postura.
    # Substituindo False por True, a estimativa será desenhada na imagem.
    frame = pose.findPose(frame, False)
    # Define os pontos encontrados
    lmList = pose.findPosition(frame, False)

    # Apenas se houver detecção...
    if len(lmList) != 0:

        # Calcula ângulo dos pontos referentes ao ombro, ao cotovelo e ao pulso dos dois braços.
        bracoDireito = pose.findAngle(frame, 12, 14, 16)
        bracoEsquerdo = pose.findAngle(frame, 11, 13, 15)

        # Se os ângulos estiverem corretos iniciará a contagem.
        # Se a postura permanecer durante 3 segundos, chama a função de detecção de objetos.
        if 20 < bracoEsquerdo < 160 and -160 < bracoDireito < -20:
            if (t == 3) and (tempo - espera >= 3):
                t = 0
                encontrarEPI(frame)
                tempoDetect = tempo
            elif (t == 2) and (tempo - espera >= 2):
                t = 3
                cv2.putText(frame, str(t), (460, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 2)
                cv2.circle(frame, (490, 620), 50, (0, 255, 255), 2)
            elif (t == 1) and (tempo - espera >= 1):
                t = 2
                cv2.putText(frame, str(t), (460, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 2)
                cv2.circle(frame, (490, 620), 50, (0, 255, 255), 2)
            elif (t == 0) and (tempo - espera >= 5):
                t = 1
                cv2.putText(frame, str(t), (460, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 2)
                cv2.circle(frame, (490, 620), 50, (0, 255, 255), 2)
                espera = tempo
        else:
            t = 0

    # O Menu será restaurado após 30 segundos da última detecção
    if tempoDetect != 0 and tempo - tempoDetect >= 30:
        restauraMenu()
        tempoDetect = 0


'''
Função de visualização de imagem:
    Redimenciona a imagem, chama a função de estimativa de postura, 
    converte a imagem de BGR para RGB e atualiza o frame no menu.
'''
def visualizar():
    global cap
    global frame
    global pTime

    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            # Redimencionar imagem
            frame = imutils.resize(frame, width=920)

            # Detecção de Postura
            detectPostura(frame)

            # Conversão de imagem BGR para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Esse trecho de código imprime a taxa de FPS na tela
            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime
            # cv2.putText(frame, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

            # Atualizar frame
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizar)
        else:
            # Caso Camera não ligue
            lblVideo.image = "Não há Câmera Conectada"
            cap.release()

    return 0


'''
Restaura o menu para a configuração padrão
'''
def restauraMenu():
    global chMascara
    global chCapacete
    global chOculos
    global chAbafador
    global chColete
    global chLuva
    global chBota

    iconPose = PhotoImage(file=myPose)
    lblDeteccao.configure(image=iconPose)
    lblDeteccao.image = iconPose

    if chMascara == 1:
        icon0 = PhotoImage(file=myIcones[0])
        lblIcone0.configure(image=icon0)
        lblIcone0.image = icon0

    if chCapacete == 1:
        icon1 = PhotoImage(file=myIcones[1])
        lblIcone1.configure(image=icon1)
        lblIcone1.image = icon1

    if chOculos == 1:
        icon2 = PhotoImage(file=myIcones[2])
        lblIcone2.configure(image=icon2)
        lblIcone2.image = icon2

    if chAbafador == 1:
        icon3 = PhotoImage(file=myIcones[3])
        lblIcone3.configure(image=icon3)
        lblIcone3.image = icon3

    if chColete == 1:
        icon4 = PhotoImage(file=myIcones[4])
        lblIcone4.configure(image=icon4)
        lblIcone4.image = icon4

    if chLuva == 1:
        icon5 = PhotoImage(file=myIcones[5])
        lblIcone5.configure(image=icon5)
        lblIcone5.image = icon5

    if chBota == 1:
        icon6 = PhotoImage(file=myIcones[6])
        lblIcone6.configure(image=icon6)
        lblIcone6.image = icon6

    lblPerIcone0.configure(text=" - ")
    lblPerIcone1.configure(text=" - ")
    lblPerIcone2.configure(text=" - ")
    lblPerIcone3.configure(text=" - ")
    lblPerIcone4.configure(text=" - ")
    lblPerIcone5.configure(text=" - ")
    lblPerIcone6.configure(text=" - ")

    btnAcesso.configure(text=" * ", bg="#ededed")
''''
Janela de Histórico
'''
def openHistorico():
    # Abre pasta contendo histórico
    historico_path = filedialog.askopenfile(mode='r', initialdir='Arquivos/Imagens_Registradas')

'''
Janela de Configurações
'''
def openConfig():
    # Abre janela de configurações
    janelaConfig = Tk()
    janelaConfig.title("Configurações")

    # Escreve label
    l = Label(janelaConfig, bg='white', width=20, text='Equipamentos Detectáveis')
    l.grid(column=0, row=0, columnspan=4)

    '''
    Funções auxiliares. 
    Serão chamadas quando os check box forem acionados.
    Os EPIs que não serão levados em conta na detecção ficaram com o ícone da cor cinza no menu.
    '''

    def funcMas():
        global chMascara
        if chMascara == 1:
            chMascara = 0
            icon0 = PhotoImage(file=myIconesNeutro[0])
            lblIcone0.configure(image=icon0)
            lblIcone0.image = icon0
        else:
            chMascara = 1
            icon0 = PhotoImage(file=myIcones[0])
            lblIcone0.configure(image=icon0)
            lblIcone0.image = icon0

    def funcCap():
        global chCapacete
        if chCapacete == 1:
            chCapacete = 0
            icon1 = PhotoImage(file=myIconesNeutro[1])
            lblIcone1.configure(image=icon1)
            lblIcone1.image = icon1
        else:
            chCapacete = 1
            icon1 = PhotoImage(file=myIcones[1])
            lblIcone1.configure(image=icon1)
            lblIcone1.image = icon1

    def funcOcu():
        global chOculos
        if chOculos == 1:
            chOculos = 0
            icon2 = PhotoImage(file=myIconesNeutro[2])
            lblIcone2.configure(image=icon2)
            lblIcone2.image = icon2
        else:
            chOculos = 1
            icon2 = PhotoImage(file=myIcones[2])
            lblIcone2.configure(image=icon2)
            lblIcone2.image = icon2

    def funcAba():
        global chAbafador
        if chAbafador == 1:
            chAbafador = 0
            icon3 = PhotoImage(file=myIconesNeutro[3])
            lblIcone3.configure(image=icon3)
            lblIcone3.image = icon3
        else:
            chAbafador = 1
            icon3 = PhotoImage(file=myIcones[3])
            lblIcone3.configure(image=icon3)
            lblIcone3.image = icon3

    def funcCol():
        global chColete
        if chColete == 1:
            chColete = 0
            icon4 = PhotoImage(file=myIconesNeutro[4])
            lblIcone4.configure(image=icon4)
            lblIcone4.image = icon4
        else:
            chColete = 1
            icon4 = PhotoImage(file=myIcones[4])
            lblIcone4.configure(image=icon4)
            lblIcone4.image = icon4

    def funcLuv():
        global chLuva
        if chLuva == 1:
            chLuva = 0
            icon5 = PhotoImage(file=myIconesNeutro[5])
            lblIcone5.configure(image=icon5)
            lblIcone5.image = icon5
        else:
            chLuva = 1
            icon5 = PhotoImage(file=myIcones[5])
            lblIcone5.configure(image=icon5)
            lblIcone5.image = icon5

    def funcBot():
        global chBota
        if chBota == 1:
            chBota = 0
            icon6 = PhotoImage(file=myIconesNeutro[6])
            lblIcone6.configure(image=icon6)
            lblIcone6.image = icon6
        else:
            chBota = 1
            icon6 = PhotoImage(file=myIcones[6])
            lblIcone6.configure(image=icon6)
            lblIcone6.image = icon6

    '''
     Conjunto de CheckBoxs para determinar se um EPI será ou não levado
     em consideração no momento da analise. 
     '''

    var0 = IntVar()
    c0 = Checkbutton(janelaConfig, text='Máscara', variable=var0, onvalue=1, offvalue=0, command=funcMas)
    c0.select()
    c0.grid(column=0, row=1)
    var1 = IntVar()
    c1 = Checkbutton(janelaConfig, text='Capacete', variable=var1,onvalue=1, offvalue=0, command=funcCap)
    c1.select()
    c1.grid(column=1, row=1)
    var2 = IntVar()
    c2 = Checkbutton(janelaConfig, text='Óculos', variable=var2, onvalue=1, offvalue=0, command=funcOcu)
    c2.select()
    c2.grid(column=2, row=1)
    var3 = IntVar()
    c3 = Checkbutton(janelaConfig, text='Abafador', variable=var3, onvalue=1, offvalue=0, command=funcAba)
    c3.select()
    c3.grid(column=3, row=1)
    var4 = IntVar()
    c4 = Checkbutton(janelaConfig, text='Colete', variable=var4, onvalue=1, offvalue=0, command=funcCol)
    c4.select()
    c4.grid(column=0, row=2)
    var5 = IntVar()
    c5 = Checkbutton(janelaConfig, text='Luvas', variable=var5, onvalue=1, offvalue=0, command=funcLuv)
    c5.select()
    c5.grid(column=1, row=2)
    var6 = IntVar()
    c6 = Checkbutton(janelaConfig, text='Botas', variable=var6, onvalue=1, offvalue=0, command=funcBot)
    c6.select()
    c6.grid(column=2, row=2)



# -------------------- LOOP JANELA PRINCIPAL ---------------------- #

# Iniciar janela principal
janelaPrincipal = Tk()
janelaPrincipal.title("VCAD_EPIs (Visão Computacional Aplicada na Detecção de Equipamentos de Proteção Individual)")
janelaPrincipal.state('zoomed')

# Posição do video na janela principal
lblVideo = Label(
    janelaPrincipal,
    bd=10,
    relief="sunken"
)
lblVideo.grid(column=0, row=0, columnspan=7, rowspan=24)

# Layout Menu de Informações
lblMenu = Label(
    janelaPrincipal,
    text="EPI's Detectados",
    font="Arial 20",
    bd=10,
    relief="sunken",
    width=24,
    height=21,
    anchor=N
).grid(column=8, row=0, padx=5, pady=5, columnspan=4, rowspan=24)

# Posição Sugerida e exibição do Print de Detecção
iconPose = PhotoImage(file=myPose)
lblDeteccao = Label(janelaPrincipal,image=iconPose)
lblDeteccao.grid(column=8, row=1, columnspan=4, rowspan=12)

# Icone referente a máscara
icon0 = PhotoImage(file=myIcones[0])
lblIcone0 = Label(janelaPrincipal, image=icon0, width=70)
lblIcone0.grid(column=8, row=13, rowspan=3)

# Ícone referente ao capacete
icon1 = PhotoImage(file=myIcones[1])
lblIcone1 = Label(janelaPrincipal, image=icon1, width=70)
lblIcone1.grid(column=9, row=13, rowspan=3)

# Ícone referente aos óculos
icon2 = PhotoImage(file=myIcones[2])
lblIcone2 = Label(janelaPrincipal, image=icon2, width=70)
lblIcone2.grid(column=10, row=13, rowspan=3)

# Icone referente ao abafador
icon3 = PhotoImage(file=myIcones[3])
lblIcone3 = Label(janelaPrincipal, image=icon3, width=70)
lblIcone3.grid(column=11, row=13, rowspan=3)

# Icone referente ao colete
icon4 = PhotoImage(file=myIcones[4])
lblIcone4 = Label(janelaPrincipal, image=icon4, width=70)
lblIcone4.grid(column=8, row=17, rowspan=3)

# Icone referente a luva
icon5 = PhotoImage(file=myIcones[5])
lblIcone5 = Label(janelaPrincipal, image=icon5, width=70)
lblIcone5.grid(column=9, row=17, rowspan=3)

# Icone referente a bota
icon6 = PhotoImage(file=myIcones[6])
lblIcone6 = Label(janelaPrincipal, image=icon6, width=70)
lblIcone6.grid(column=10, row=17, rowspan=3)

# Texto referente a máscara
lblPerIcone0 = Label(janelaPrincipal, text="  -  ", font="Arial 15", bd=2, relief="solid")
lblPerIcone0.grid(column=8, row=15, rowspan=2)

# Texto referente ao capacete
lblPerIcone1 = Label(janelaPrincipal, text="  -  ", font="Arial 15", bd=2, relief="solid")
lblPerIcone1.grid(column=9, row=15, rowspan=2)

# Texto referente ao óculos
lblPerIcone2 = Label(janelaPrincipal, text="  -  ", font="Arial 15", bd=2, relief="solid")
lblPerIcone2.grid(column=10, row=15, rowspan=2)

# Texto referente ao abafador
lblPerIcone3 = Label(janelaPrincipal, text="  -  ", font="Arial 15", bd=2, relief="solid")
lblPerIcone3.grid(column=11, row=15, rowspan=2)

# Texto referente ao colete
lblPerIcone4 = Label(janelaPrincipal, text="  -  ", font="Arial 15", bd=2, relief="solid")
lblPerIcone4.grid(column=8, row=19, rowspan=2)

# Texto referente a luva
lblPerIcone5 = Label(janelaPrincipal, text="  -  ", font="Arial 15", bd=2, relief="solid")
lblPerIcone5.grid(column=9, row=19, rowspan=2)

# Texto referente a bota
lblPerIcone6 = Label(janelaPrincipal, text="  -  ", font="Arial 15", bd=2, relief="solid")
lblPerIcone6.grid(column=10, row=19, rowspan=2)

# Botão de Acesso. Restaura o menu quando precionado.
btnAcesso = Button(janelaPrincipal, text=" * ", font="Arial 22", width=20, bg="#ededed", command=restauraMenu)
btnAcesso.grid(column=8, row=21, columnspan=4)

# Abrir Janela de Histórico
btnHistorico = Button(janelaPrincipal, text="Histórico", width=22, command=openHistorico).grid(column=8, row=22, columnspan=2)

# Abrir Janela de Configurações
btnConfig = Button(janelaPrincipal, text="Configurações", width=22, command=openConfig).grid(column=10, row=22, columnspan=2)

# Chamar função de exibição de video
visualizar()

# Fim do loop janela principal
janelaPrincipal.mainloop()

# Encerra programa
cap.release()
cv2.destroyAllWindows()

# ------------------------- FIM! -------------------------- #