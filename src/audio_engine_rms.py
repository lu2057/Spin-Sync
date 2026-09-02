import threading
import asyncio
import numpy as np
import sounddevice as sd

class AudioProducer:
    def __init__(self, async_queue: asyncio.Queue, async_loop: asyncio.AbstractEventLoop):
        self.queue = async_queue
        self.loop = async_loop
        
        self.sample_rate = 16000
        self.chunk_duration = 3.0
        self.channels = 1
        self._stop_event = threading.Event()

        # --- PARAMETRI MATEMATICI (Da tarare se necessario) ---
        self.silence_threshold = 0.10  # Sotto questo valore di RMS, è considerato silenzio
        self.music_cv_threshold = 0.35   # Soglia di Varianza. Sotto = Musica, Sopra = Parlato

    def start_listening(self):
        print("[AudioEngine] Avvio ascolto microfono (Modalità Matematica/DSP)...")
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop_listening(self):
        print("[AudioEngine] Arresto thread microfono...")
        self._stop_event.set()

    def _analyze_audio_math(self, audio_array):
        """
        Analizza l'array audio usando statistiche DSP (Digital Signal Processing).
        """
        # 1. Rimuoviamo eventuale offset DC (centriamo l'onda sullo zero)
        audio_array = audio_array - np.mean(audio_array)
        
        # 2. Calcoliamo il volume globale (RMS - Root Mean Square)
        rms_total = np.sqrt(np.mean(audio_array**2))
        if rms_total < self.silence_threshold:
            return False, f"Silenzio assoluto (RMS: {rms_total:.4f})"

        # 3. Dividiamo i 3 secondi in "finestre" temporali 
        window_size = int(self.sample_rate * 1)
        num_windows = len(audio_array) // window_size
        
        energies = []
        for i in range(num_windows):
            window = audio_array[i*window_size : (i+1)*window_size]
            # Energia della singola finestra (somma dei quadrati dell'ampiezza)
            energies.append(np.sum(window**2))
            
        # 4. Calcoliamo il Coefficiente di Variazione (CV) dell'energia
        # Formula: Deviazione Standard / Media
        mean_energy = np.mean(energies)
        if mean_energy == 0:
            return False, "Silenzio"
            
        cv_energy = np.std(energies) / mean_energy
        
        # La logica:
        # - La MUSICA è un "muro di suono" continuo e compresso: l'energia varia poco tra una finestra e l'altra (Basso CV).
        # - Il PARLATO è fatto di parole esplosive seguite da pause di respiro: l'energia varia moltissimo (Alto CV).
        is_music = cv_energy < self.music_cv_threshold and cv_energy > self.silence_threshold
        
        if is_music:
            return True, f"🎵 MUSICA (Varianza Energia: {cv_energy:.2f} < {self.music_cv_threshold})"
        elif cv_energy >= self.silence_threshold:
            return False, f"🗣️ PARLATO (Varianza Energia: {cv_energy:.2f} >= {self.music_cv_threshold})"
        else:
            return False, f"Silenzio (Varianza Energia: {cv_energy:.2f})"


    def _record_loop(self):
        total_frames = int(self.sample_rate * self.chunk_duration)

        while not self._stop_event.is_set():
            # Fase 1: Registrazione diretta in RAM
            audio_data = sd.rec(
                frames=total_frames, 
                samplerate=self.sample_rate, 
                channels=self.channels, 
                dtype='float32'
            )
            sd.wait()

            # Appiattiamo l'array da 2D a 1D per darlo in pasto a numpy in modo pulito
            audio_array = audio_data.flatten()

            # Fase 2: Analisi Matematica (Avviene in pochi millisecondi, niente file su disco!)
            is_music, log_message = self._analyze_audio_math(audio_array)
            print(f"[AudioEngine] {log_message}")

            # Fase 3: Spedizione asincrona
            if is_music:
                payload = {
                    "is_music": True,
                    "audio_array": audio_data,
                    "sample_rate": self.sample_rate
                }
                self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)