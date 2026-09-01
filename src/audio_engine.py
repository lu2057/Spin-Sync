import threading
import tempfile
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import asyncio
from pyAudioAnalysis import audioTrainTest as aT

class AudioProducer:
    def __init__(self, async_queue: asyncio.Queue, async_loop: asyncio.AbstractEventLoop):
        self.queue = async_queue
        self.loop = async_loop
        
        # Parametri Audio
        self.sample_rate = 44100
        self.chunk_duration = 3.0  # Registriamo chunk di 3 secondi
        self.channels = 1          # Mono è sufficiente (e migliore) per l'analisi
        
        # Troviamo il percorso assoluto della cartella principale del progetto
        current_dir = os.path.dirname(os.path.abspath(__file__)) # cartella src/
        project_root = os.path.dirname(current_dir)              # cartella Spin-Sync/
        # Percorso del modello pre-addestrato di pyAudioAnalysis 
        # (Speech vs Music - SVM con RBF kernel)
        self.model_path = "../pyAudioAnalysis/data/models/svm_rbf_sm"
        self.model_type = "svm_rbf"
        
        self._stop_event = threading.Event()

    def start_listening(self):
        """Avvia il thread di registrazione continua in background."""
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print("[AudioEngine] Microfono in ascolto e modello ML attivato...")

    def stop_listening(self):
        """Ferma il thread in modo pulito."""
        self._stop_event.set()

    def _record_loop(self):
        """Ciclo continuo (girerà nel thread separato)"""
        frames_per_chunk = int(self.sample_rate * self.chunk_duration)

        while not self._stop_event.is_set():
            # 1. Registra dal microfono in modo bloccante (ma solo in questo thread)
            audio_data = sd.rec(frames_per_chunk, samplerate=self.sample_rate, channels=self.channels, dtype='float32')
            sd.wait() # Attende la fine dei 3 secondi

            # 2. Crea un file WAV temporaneo (pyAudioAnalysis richiede un file fisico)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_filename = temp_wav.name
            
            try:
                # Salva l'array numpy nel file wav
                sf.write(temp_filename, audio_data, self.sample_rate)

                # 3. Classifica l'audio con il modello Machine Learning
                # Risultato: class_id (int), probabilità (array), nomi_classi (list)
                class_id, probs, class_names = aT.file_classification(temp_filename, self.model_path, self.model_type)
                
                # Sostanzialmente class_names sarà ['music', 'speech']
                predicted_class = class_names[int(class_id)]
                confidence = probs[int(class_id)]
                
                print(f"[ML] Rilevato: {predicted_class} (Confidenza: {confidence*100:.1f}%)")

                # 4. Invia i dati alla coda asincrona del programma principale
                # Usiamo call_soon_threadsafe perché stiamo comunicando da un Thread all'Event Loop asincrono
                messaggio = {
                    "is_music": (predicted_class == 'music' and confidence > 0.70),
                    "audio_array": audio_data,
                    "sample_rate": self.sample_rate
                }
                self.loop.call_soon_threadsafe(self.queue.put_nowait, messaggio)

            finally:
                # Elimina sempre il file temporaneo per non intasare l'hard disk
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)