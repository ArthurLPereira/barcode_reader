import numpy as np
import imutils
import argparse
import cv2 
import sys

class BarcodeFinder():
    
    # Criando instância da classe
    def __init__(self, path):
        self.image = cv2.imread(path)
        path_split = path.split('/') if '/' in path else path.split('\\') if '\\' in path else ['', path]
        self.image_name = path_split[-1].split('.')[0]
        self.PATH = '/'.join(path_split[:-1]) + '/'
        self.ddepth = cv2.CV_32F
    
    # Convertendo a Imagem para escala de cinza
    def grayscale(self, image=None):
        if image is None:
            image = self.image
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return grayscale

    # Aplicando Sobel e suavisando
    def sobel(self, grayscale=None):
        if grayscale is None:
            grayscale = self.grayscale()
        gradientX = cv2.Sobel(grayscale, ddepth=self.ddepth, dx=1, dy=0, ksize=-1)
        gradientY = cv2.Sobel(grayscale, ddepth=self.ddepth, dx=0, dy=1, ksize=-1)

        gradient = cv2.subtract(gradientX, gradientY)
        gradient = cv2.convertScaleAbs(gradient)
        blurred = cv2.blur(gradient, (9, 9))
        return blurred

    # Limiarizando a Imagem
    def threshold(self, blurred=None):
        if blurred is None:
            blurred = self.sobel()

        _, threshold = cv2.threshold(blurred, 220, 255, cv2.THRESH_BINARY)
        return threshold

    # Aplicando Erosão e Dilatação Várias Vezes
    def erode_dilate(self, threshold=None):
        if threshold is None:
            threshold = self.threshold()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(threshold,cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)

        return closed

    # Encontrando o Retângulo que contem um possível código de barras
    def find_rectangle(self, closed=None):
        if closed is None:
            closed = self.erode_dilate()
        
        contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    # Aplicando retângulo na imagem original    
    def final_product(self, box=None):
        if box is None:
            box = self.find_rectangle()

        copy = self.image.copy()
        cv2.drawContours(copy, [box], -1, (0, 255, 0), 3)
        cv2.imshow("Final_product", copy)
        cv2.waitKey(0)
        cv2.imwrite('{}{}_processed.jpg'.format(self.PATH, self.image_name), copy)

    def show_image(self, image=None, text='Image'):
        if image is None:
            image = self.image
        cv2.imshow(text, image)
        cv2.waitKey(0)



if __name__ == "__main__":
    path = sys.argv[1]
    try:
        finder = BarcodeFinder(path)
        finder.show_image()
        finder.final_product()
    except Exception as exp:
        print(exp)
    pass
