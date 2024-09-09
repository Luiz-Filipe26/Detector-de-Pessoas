import threading
import cv2
from frame_processor import initialize, process_frame

class FrameProcessorWrapper:
    def __init__(self, frame_provider, frame_callback):
        self.frame_provider = frame_provider
        self.frame_callback = frame_callback
        self.running = False

        # Inicializar os recursos do frame_processor
        initialize()

    def _process_frames(self):
        while self.running:
            frame = self.frame_provider.get_next_frame()
            if frame is None:
                print("Nenhum frame para processar.")
                continue

            # Processar o frame
            processed_frame = process_frame(frame)
            self.frame_callback(processed_frame)  # Atualiza o frame processado

    def start(self):
        print("Thread do FrameProcessor iniciando...")
        self.running = True
        self.thread = threading.Thread(target=self._process_frames)
        self.thread.start()
        print("Thread do FrameProcessor iniciada!")

    def stop(self):
        print("Parando a Thread do FrameProcessor...")
        self.running = False
        if self.thread is not None:
            self.thread.join()
        print("Thread do FrameProcessor parada!")