from collections import deque
import threading

class FrameProvider:
    def __init__(self, buffer_size=10):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.buffer_filled_event = threading.Event()

    def add_frame(self, frame):
        #print("Tentando adicionar frame ao buffer...")
        with self.buffer_lock:
            self.buffer.append(frame)
            self.buffer_filled_event.set()  # Sinaliza que o buffer agora tem um frame disponível
        #print("Frame adicionado ao buffer.")

    def get_next_frame(self):
        while True:
            with self.buffer_lock:
                if len(self.buffer) > 0:
                    return self.buffer.popleft()

            #print("Aguardando frames no buffer...")
            self.buffer_filled_event.wait()  # Espera até que o buffer tenha pelo menos um frame
            self.buffer_filled_event.clear()  # Limpa o evento para esperar por mais frames