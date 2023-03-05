import cv2

# chama o classificador xml haarcascade
cascade = cv2.CascadeClassifier('./1-cascade.xml')

# chama a imagem CPF para identificar
imagem = cv2.imread('./imgs/cpf-5.jpeg')

# cinza, porque o opencv reconhece melhor imagens em escala de cinza
reconhecimento = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#aplica o classifier na nossa imagem em cinza
cpfs = cascade.detectMultiScale(reconhecimento, scaleFactor=2.05, minNeighbors=1, minSize=(50, 50))

# loop para criar os quadrados de reconhecimento
for(x, y, w, h) in cpfs:
    cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

# final do código
cv2.imshow('DOCS', imagem)
cv2.waitKey()
cv2.destroyAllWindows()