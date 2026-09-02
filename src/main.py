import sys
import os

# --- TRUCCO PER IL PATH ---
# Calcola automaticamente il percorso della cartella principale (Spin-Sync)
# e lo aggiunge ai percorsi in cui Python cerca i moduli.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# --------------------------

import asyncio
# Ho scelto di usare attualmente il metodo del rms per rilevare la musica
from audio_engine_rms import AudioProducer

async def consumer_loop(coda_audio):
    """
    Questo è il ciclo infinito che riceve i pacchetti dal microfono.
    Essendo asincrono (async def), non bloccherà mai l'intero programma.
    """
    print("✅ Sistema pronto. Prova a mettere una canzone o a parlare!\n")
    
    while True:
        # .get() mette "in pausa" questo ciclo finché non arriva qualcosa nella coda
        # NON consuma cicli di CPU mentre aspetta!
        pacchetto = await coda_audio.get()
        
        # Quando arriva il pacchetto (ogni 3 secondi), controlliamo l'etichetta
        if pacchetto["is_music"]:
            print("==================================================")
            print("🚀 MUSICA RILEVATA CON ALTA CONFIDENZA! ")
            print("In futuro, l'array audio verrà inviato a Shazam...")
            print("==================================================\n")
        else:
            print("... Silenzio o Parlato. Sto ignorando.\n")
            
        # Segnaliamo alla coda che abbiamo finito con questo pacchetto
        coda_audio.task_done()


async def main():
    print("🎵 Inizializzazione Vinyl Detector...")
    
    # 1. Recuperiamo il motore asincrono (Event Loop) e creiamo la Coda
    loop = asyncio.get_running_loop()
    coda_audio = asyncio.Queue()
    
    # 2. Inizializziamo il microfono passandogli i due strumenti di comunicazione
    motore_audio = AudioProducer(async_queue=coda_audio, async_loop=loop)
    
    # 3. Accendiamo il Thread in background
    motore_audio.start_listening()
    
    try:
        # 4. Avviamo la funzione consumer che aspetterà all'infinito
        await consumer_loop(coda_audio)
        
    except asyncio.CancelledError:
        # Se cancelliamo l'Event Loop, spegniamo il microfono
        motore_audio.stop_listening()


if __name__ == "__main__":
    try:
        # asyncio.run() è il comando fondamentale che accende l'Event Loop
        asyncio.run(main())
    except KeyboardInterrupt:
        # Cattura il classico Ctrl+C dal terminale per spegnere l'app pulita
        print("\n[Sistema] Arresto richiesto dall'utente. Spegnimento...")
        sys.exit(0)