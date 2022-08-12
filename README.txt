Eu ainda preciso organizar melhor as coisas por aqui, então sugiro ler o "PDF - Visão Computacional Aplicada na Detecção de Equipamentos de Proteção Individual" (artigo não publicado) ou ver um vídeo dos primeiros testes aqui: https://www.youtube.com/watch?v=BBgDAaMH-2I&t=5s

Este é um projeto acadêmico com o objetivo de detectar EPIs que estão sendo vestidos por uma pessoa em uma cena. 
O programa foi desenvolvido em Python 3.
Para tratamento de imagens foi utilizado OpenCV.
A detecção de objetos é realizada com opencv-dnn e YOLOv4.
É utilizado estimativa de postura com MediaPipe para comparar as coordenadas das detecções com as regiões de interesse.
