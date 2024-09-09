import cv2
import numpy as np

# Variável global para o background_subtractor
background_subtractor = None


def initialize():
    global background_subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2()


def process_contours(contours):
    # Lista que vai armazenar os novos contornos processados com seus respectivos nomes (rótulos)
    processed_contours = []

    for contour in contours:
        # Condição dummy: se a área do contorno for maior que 500, é considerado válido.
        # No futuro, você pode expandir essa lógica para fazer ajustes ou remover contornos.
        if cv2.contourArea(contour) > 500:
            label = "adulto"  # Placeholder - no futuro, adicione lógica real para definir "adulto", "criança", ou "animal"
            processed_contours.append((contour, label))

    return processed_contours


def process_frame(current_frame):
    global background_subtractor
    if background_subtractor is None:
        raise RuntimeError("O background_subtractor não foi inicializado. Chame initialize() primeiro.")

    # Subtração de fundo
    fg_mask = background_subtractor.apply(current_frame)

    # Limpeza de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Encontrar contornos
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Processar os contornos e obter uma nova lista de contornos com labels
    processed_contours = process_contours(contours)

    # Criar uma máscara preta do mesmo tamanho que o frame
    mask_image = np.zeros_like(current_frame)

    # Preencher as áreas dentro dos contornos processados com branco na máscara
    for contour, label in processed_contours:
        cv2.drawContours(mask_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)  # Branco

    # Converter a máscara para escala de cinza
    gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Criar uma máscara binária
    _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

    # Criar a imagem processada
    processed_frame = np.zeros_like(current_frame)  # Imagem preta

    # Aplicar a máscara à imagem original
    processed_frame[binary_mask == 255] = current_frame[binary_mask == 255]  # Manter as áreas internas dos contornos

    # Desenhar contornos verdes e os labels no frame processado
    for contour, label in processed_contours:
        if cv2.contourArea(contour) > 500:  # Filtrar pequenos contornos
            cv2.drawContours(processed_frame, [contour], -1, (0, 255, 0), 2)  # Verde

            # Determinar o ponto onde o texto será desenhado (neste caso, a coordenada do ponto mais alto do contorno)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(processed_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return processed_frame