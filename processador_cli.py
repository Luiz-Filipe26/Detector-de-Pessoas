import os
import cv2
from video_source import VideoSource
from frame_processor_wrapper import FrameProcessorWrapper
import threading

# Variáveis globais e condição de sincronização
current_frame = None
frame_condition = threading.Condition()

def is_valid_video_file(path):
    return os.path.isfile(path) and cv2.VideoCapture(path).isOpened()

def update_frame(new_frame):
    global current_frame
    with frame_condition:
        current_frame = new_frame
        frame_condition.notify()  # Notifica que um novo frame está disponível

if __name__ == "__main__":
    while True:
        #source = "/home/luiz/Downloads/people_walking.mp4"
        source = "/home/luiz/Downloads/man_walking.mp4"
        #source = input("Digite o caminho do arquivo de vídeo ou URL da câmera: ")
        if is_valid_video_file(source):
            print(f"Arquivo de vídeo válido: {source}")
            break
        else:
            print("Caminho inválido ou não foi possível abrir o arquivo de vídeo. Tente novamente.")

    video_source = VideoSource(source)
    video_source.start()

    frame_processor_wrapper = FrameProcessorWrapper(video_source.frame_provider, update_frame)
    frame_processor_wrapper.start()

    try:
        while True:
            with frame_condition:
                frame_condition.wait()  # Aguarda a notificação do novo frame
            if current_frame is not None:
                cv2.imshow('Processed Frame', current_frame)

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Saindo...")
                break
    except KeyboardInterrupt:
        print("Interrupção manual detectada.")

    video_source.stop()
    frame_processor_wrapper.stop()

    cv2.destroyAllWindows()