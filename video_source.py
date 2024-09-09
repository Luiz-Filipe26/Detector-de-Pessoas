import cv2
import threading
import time
from frame_provider import FrameProvider

class VideoSource:
    def __init__(self, video_source, buffer_size=10):
        self.frame_provider = FrameProvider(buffer_size)
        self.cap = cv2.VideoCapture(video_source)
        self.running = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Obtém o framerate do vídeo
        self.frame_delay = 1 / self.fps  # Calcula o atraso entre frames em segundos

        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o arquivo de vídeo: {video_source}")

    def _buffer_frames(self):
        while self.running:
            start_time = time.time()  # Marca o tempo no início do loop

            ret, frame = self.cap.read()
            if not ret:
                #print("Nenhum frame lido. Finalizando...")
                self.running = False
                break
            #print("Frame lido e adicionado ao buffer")
            self.frame_provider.add_frame(frame)

            elapsed_time = time.time() - start_time  # Calcula o tempo decorrido
            sleep_time = self.frame_delay - elapsed_time  # Tempo restante para manter o framerate
            if sleep_time > 0:
                time.sleep(sleep_time)  # Espera o tempo necessário para respeitar o framerate

    def start(self):
        print(f"Thread do VideoSource iniciando... (FPS: {self.fps:.2f})")
        self.running = True
        self.thread = threading.Thread(target=self._buffer_frames)
        self.thread.start()
        print("Thread do VideoSource iniciada!")

    def stop(self):
        print("Parando a Thread do VideoSource...")
        self.running = False
        if self.thread is not None:
            self.thread.join()
        print("Thread do VideoSource parada!")