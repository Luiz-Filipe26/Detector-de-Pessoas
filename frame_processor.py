import cv2
import numpy as np

# Variável global para o background_subtractor
background_subtractor = None

# Constantes para classificação usando aspect ratio
ADULT_ASPECT_RATIO_MIN = 20 / 200  # Largura 20, Altura 200
ADULT_ASPECT_RATIO_MAX = 20 / 100  # Largura 20, Altura 100

CHILD_ASPECT_RATIO_MIN = 20 / 100  # Largura 20, Altura 100
CHILD_ASPECT_RATIO_MAX = 20 / 50   # Largura 20, Altura 50

ANIMAL_ASPECT_RATIO_MIN = 20 / 50  # Largura 20, Altura 50
ANIMAL_ASPECT_RATIO_MAX = 50 / 20   # Largura 50, Altura 20

def initialize():
    global background_subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2()

def find_split_point(contour):
    """
    Determine the Y coordinate where the contour should be split based on significant deviation
    from the average width of neighboring regions.
    Returns 0 if no split is needed, or the Y coordinate to split.
    """
    points = contour[:, 0, :]
    y_coords = points[:, 1]

    # Ordenar os y_coords e calcular as larguras médias das vizinhanças
    sorted_y_coords = np.sort(np.unique(y_coords))
    widths = [sorted_y_coords[i + 1] - sorted_y_coords[i] for i in range(len(sorted_y_coords) - 1)]

    if len(widths) == 0:
        return 0

    # Ordenar larguras para análise
    sorted_widths = np.sort(widths)
    lower_20_percent_index = max(1, int(len(sorted_widths) * 0.2))
    lower_20_percent_widths = sorted_widths[:lower_20_percent_index]

    # Calcular a largura média dos menores 20%
    avg_lower_20_percent_width = np.mean(lower_20_percent_widths)

    # Encontrar pontos onde a largura é significativamente maior do que a média dos menores 20%
    significant_deviation_threshold = avg_lower_20_percent_width * 1.5  # Ajuste conforme necessário
    potential_splits = [sorted_y_coords[i] for i in range(len(widths)) if widths[i] > significant_deviation_threshold]

    if len(potential_splits) > 0:
        # Retorna o ponto médio dos pontos candidatos
        return int(np.mean(potential_splits))

    return 0

def cut_contour_at_y(contour, split_y):
    """
    Cut the contour into two parts at the given Y coordinate.
    """
    points = contour[:, 0, :]

    # Dividimos os pontos em dois contornos com base no split_y
    above_split = points[points[:, 1] <= split_y]
    below_split = points[points[:, 1] > split_y]

    if len(above_split) > 0 and len(below_split) > 0:
        return [above_split.reshape(-1, 1, 2), below_split.reshape(-1, 1, 2)]
    return [contour]

def classify_contour(contour):
    """
    Classify the contour as 'adult', 'child', or 'animal' based on its aspect ratio.
    """
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h

    if ADULT_ASPECT_RATIO_MIN <= aspect_ratio <= ADULT_ASPECT_RATIO_MAX:
        return 'adulto'
    elif CHILD_ASPECT_RATIO_MIN <= aspect_ratio <= CHILD_ASPECT_RATIO_MAX:
        return 'criança'
    elif ANIMAL_ASPECT_RATIO_MIN <= aspect_ratio <= ANIMAL_ASPECT_RATIO_MAX:
        return 'animal'
    else:
        return 'desconhecido'

def process_contours(contours):
    # Lista que vai armazenar os novos contornos processados com seus respectivos nomes (rótulos)
    processed_contours = []

    for contour in contours:
        # Condição dummy: se a área do contorno for maior que 500, é considerado válido.
        if cv2.contourArea(contour) > 500:
            # Determina onde deve ser feito o split
            split_y = find_split_point(contour)

            # Classificar o contorno
            label = classify_contour(contour)

            if split_y > 0:
                split_contours = cut_contour_at_y(contour, split_y)
                for split_contour in split_contours:
                    processed_contours.append((split_contour, label))  # Adiciona o rótulo classificado
            else:
                processed_contours.append((contour, label))  # Adiciona o rótulo classificado

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