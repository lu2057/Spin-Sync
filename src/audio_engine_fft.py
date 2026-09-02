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

        # --- PARAMETRI DI CALIBRAZIONE ---
        self.silence_threshold = 0.01      # Sotto questo RMS, scarta l'audio (rumore di fondo)
        self.music_fft_threshold = 0.8     # Soglia del Rapporto Spettrale (Bassi+Alti / Medi)

    def start_listening(self):
        print("[AudioEngine] Avvio ascolto microfono (Modalità Analisi di Spettro FFT)...")
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop_listening(self):
        print("[AudioEngine] Arresto thread microfono...")
        self._stop_event.set()

    def _analyze_audio_fft(self, audio_array):
        """
        Analizza l'array audio usando la Fast Fourier Transform (FFT) per separare le frequenze.
        """
        # 1. Rimuoviamo l'offset DC e controlliamo il silenzio assoluto
        audio_array = audio_array - np.mean(audio_array)
        rms_total = np.sqrt(np.mean(audio_array**2))
        
        if rms_total < self.silence_threshold:
            return False, f"Silenzio/Rumore di fondo (RMS: {rms_total:.4f})"

        # 2. Applichiamo una finestra di Hanning per ridurre gli artefatti ai bordi dell'onda
        hanning_window = np.hanning(len(audio_array))
        windowed_audio = audio_array * hanning_window

        # 3. Eseguiamo la FFT (Usiamo rfft perché l'audio ha solo valori reali)
        fft_result = np.fft.rfft(windowed_audio)
        fft_magnitudes = np.abs(fft_result) # Otteniamo l'ampiezza delle frequenze
        fft_freqs = np.fft.rfftfreq(len(windowed_audio), 1 / self.sample_rate)

        # 4. Dividiamo lo spettro in tre "Bande" (misurate in Hz)
        # BANDA 1: Bassi (es. cassa, basso elettrico) -> 20 Hz - 300 Hz
        bass_mask = (fft_freqs >= 20) & (fft_freqs < 300)
        
        # BANDA 2: Medi/Voce umana -> 300 Hz - 3000 Hz
        mid_mask = (fft_freqs >= 300) & (fft_freqs < 3000)
        
        # BANDA 3: Alti (es. piatti, hi-hat, synth) -> 3000 Hz - 8000 Hz (il massimo per 16kHz)
        treble_mask = (fft_freqs >= 3000) & (fft_freqs <= 8000)

        # Calcoliamo l'energia totale in ciascuna banda (somma delle ampiezze)
        bass_energy = np.sum(fft_magnitudes[bass_mask])
        mid_energy = np.sum(fft_magnitudes[mid_mask])
        treble_energy = np.sum(fft_magnitudes[treble_mask])

        # Preveniamo la divisione per zero
        if mid_energy == 0:
            mid_energy = 1e-6

        # 5. Calcoliamo il "Rapporto Spettrale"
        # Rapporto = Energia ai lati (Bassi + Alti) diviso l'Energia al centro (Medi/Voce)
        spectral_ratio = (bass_energy + treble_energy) / mid_energy

        # La logica:
        # - La MUSICA satura i bassi e gli alti, alzando vertiginosamente il rapporto.
        # - Il PARLATO concentra quasi tutta l'energia nei medi, abbassando il rapporto.
        is_music = spectral_ratio > self.music_fft_threshold
        
        if is_music:
            return True, f"🎵 MUSICA (Rapporto Spettrale: {spectral_ratio:.2f} > {self.music_fft_threshold})"
        else:
            return False, f"🗣️ PARLATO (Rapporto Spettrale: {spectral_ratio:.2f} <= {self.music_fft_threshold})"

    def _record_loop(self):
        total_frames = int(self.sample_rate * self.chunk_duration)

        while not self._stop_event.is_set():
            audio_data = sd.rec(
                frames=total_frames, 
                samplerate=self.sample_rate, 
                channels=self.channels, 
                dtype='float32'
            )
            sd.wait()

            audio_array = audio_data.flatten()

            # Passiamo all'analisi frequenziale
            is_music, log_message = self._analyze_audio_fft(audio_array)
            print(f"[AudioEngine] {log_message}")

            if is_music:
                payload = {
                    "is_music": True,
                    "audio_array": audio_data,
                    "sample_rate": self.sample_rate
                }
                self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)