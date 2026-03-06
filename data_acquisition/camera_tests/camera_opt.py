import cv2
import time

# Inicjalizacja z DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 1. Rozdzielczość i format
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# 2. KLUCZ: Wyłączenie wszelkiej automatyki przed ustawieniem wartości
# W DirectShow dla AUTO_EXPOSURE: 1 to Manual, 0.75 to Auto. 
# Często 0.25 (1/4) wymusza tryb całkowicie ręczny.
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 

# 3. EKSPOZYCJA - Twoje główne narzędzie
# Skoro -13 wciąż daje rozmycie, spróbujmy uderzyć w -14 lub -15 (jeśli hardware pozwoli).
# -14 to ok. 60 mikrosekund ($61 \mu s$). 
cap.set(cv2.CAP_PROP_EXPOSURE, -15) 

# 4. WZMOCNIENIE (GAIN) - Abyś widział cokolwiek przy tak krótkim czasie
# Przy -14 obraz będzie CZARNY bez ogromnej ilości światła. 
# Podbijamy Gain, by sztucznie rozjaśnić matrycę (kosztem szumu).
cap.set(cv2.CAP_PROP_GAIN, 0) # Maksymalna lub bliska maksymalnej wartość

# 5. CZYSZCZENIE OBRAZU
cap.set(cv2.CAP_PROP_BRIGHTNESS, 30)  # Lekkie podbicie bazowej jasności
cap.set(cv2.CAP_PROP_CONTRAST, 50)    # Zwiększenie kontrastu pomoże wyciągnąć krawędzie obiektu
cap.set(cv2.CAP_PROP_SHARPNESS, 0)    # WYŁĄCZ ostrość - hardware'owe ostrzenie tworzy artefakty, które wyglądają jak smużenie
cap.set(cv2.CAP_PROP_BACKLIGHT, 0)    # Wyłącz kompensację tła - to tylko miesza w algorytmach

# Sprawdzenie, czy ustawienia "weszły"
print(f"Ustawiona ekspozycja: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

# Pętla do przechwytywania 4 zdjęć na sekundę
try:
    while True:
        start_time = time.time()
        
        ret, frame = cap.read()
        if ret:
            # Tutaj robisz coś ze zdjęciem
            cv2.imshow('Global Shutter Freeze', frame)
            
        # Logika trzymania ok. 4 FPS (czekamy 250ms minus czas przetwarzania)
        #wait_time = max(1, int(250 - (time.time() - start_time) * 1000))
        #if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        #    break
finally:
    cap.release()
    cv2.destroyAllWindows()