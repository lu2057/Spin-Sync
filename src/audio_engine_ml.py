import os
import threading
import tempfile
import asyncio
import sounddevice as sd
import soundfile as sf

# --- PERCORSO FORZATO DI FFMPEG ---
# Incolla qui sotto il percorso esatto che ti ha restituito 'where.exe ffmpeg'
percorso_ffmpeg_esatto = r"./bin/ffmpeg.exe"

# Lo iniettiamo direttamente nel PATH di sistema per Python
cartella_ffmpeg = os.path.dirname(percorso_ffmpeg_esatto)
if cartella_ffmpeg not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + cartella_ffmpeg
# ----------------------------------

import shutil

ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    raise RuntimeError("FFmpeg not found in PATH")

from inaSpeechSegmenter import Segmenter

class AudioProducer:
    def __init__(self, async_queue: asyncio.Queue, async_loop: asyncio.AbstractEventLoop):
        # Il ponte per comunicare con il main asincrono
        self.queue = async_queue
        self.loop = async_loop
        
        # Parametri di registrazione
        self.sample_rate = 16000  # inaSpeechSegmenter lavora ottimamente a 16kHz o 44100Hz (16k è il suo standard nativo)
        self.chunk_duration = 3.0  # Registriamo chunk di 3 secondi
        self.channels = 1          # Mono
        
        # Inizializziamo il Segmenter di inaSpeechSegmenter
        # vad_engine='smn' separa Speech, Music e Noise
        # detect_gender=False disattiva il riconoscimento del sesso della voce (rende tutto molto più veloce)
        print("[AudioEngine] Loading inaSpeechSegmenter model...")
        self.segmenter = Segmenter(vad_engine='smn', detect_gender=False)
        
        # Interruttore di sicurezza per il thread
        self._stop_event = threading.Event()

    def start_listening(self):
        """Avvia il thread di registrazione in background."""
        print("[AudioEngine] Starting background microphone thread...")
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop_listening(self):
        """Ferma il thread in modo pulito."""
        print("[AudioEngine] Stopping microphone thread...")
        self._stop_event.set()

    def _record_loop(self):
        """Ciclo continuo di ascolto nel thread separato."""
        total_frames_to_record = int(self.sample_rate * self.chunk_duration)

        while not self._stop_event.is_set():
            
            # --- FASE 1: REGISTRAZIONE ---
            print("[AudioEngine] Listening for 3 seconds...")
            audio_data = sd.rec(
                frames=total_frames_to_record, 
                samplerate=self.sample_rate, 
                channels=self.channels, 
                dtype='float32'
            )
            sd.wait() # Attende la fine della registrazione

            # --- FASE 2: FILE TEMPORANEO ---
            # inaSpeechSegmenter richiede un percorso file temporaneo per analizzare l'audio
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()

            try:
                # Salviamo l'array audio nel file wav usando soundfile
                sf.write(temp_wav_path, audio_data, self.sample_rate)

                # --- FASE 3: ANALISI CON INASPEECHSEGMENTER ---
                # Restituisce una lista di segmenti sotto forma di tuple: (label, start_time, end_time)
                # Le etichette (labels) possibili con 'smn' sono tipicamente: 'music', 'speech', 'noEnergy'
                segmentation_results = self.segmenter(temp_wav_path)
                
                # Stampiamo cosa ha trovato nel chunk
                print(f"[ML Raw Results]: {segmentation_results}")

                # Valutiamo se c'è presenza di musica nel chunk
                is_music_detected = False
                
                for label, start_time, end_time in segmentation_results:
                    if label == 'music':
                        is_music_detected = True
                        break

                if is_music_detected:
                    print("[AudioEngine] 🎵 Music detected in chunk!")
                else:
                    print("[AudioEngine] 🔇 No music (Speech or Noise).")

                # --- FASE 4: SPEDIZIONE ALLA CODA ---
                payload = {
                    "is_music": is_music_detected,
                    "audio_array": audio_data,
                    "sample_rate": self.sample_rate
                }
                
                self.loop.call_soon_threadsafe(self.queue.put_nowait, payload)

            # --- FASE 5: PULIZIA ---
            finally:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)