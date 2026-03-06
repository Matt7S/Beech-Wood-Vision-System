import cv2

# Inicjalizacja kamery (DirectShow jest kluczowy dla pełnej kontroli parametrów na Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 1. Konfiguracja formatu i płynności
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 30)

# Jasność (Brightness): -64 do 64 (lub 0-255). Podnosi ogólny poziom czerni.
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)

# Kontrast (Contrast): 0 do 64. Różnica między jasnymi a ciemnymi partiami.
cap.set(cv2.CAP_PROP_CONTRAST, 32)

# Odcień (Hue): ok. -40 do 40. Zmienia barwy (np. skóra staje się zielona/fioletowa).
cap.set(cv2.CAP_PROP_HUE, 0)

# Nasycenie (Saturation): 0 do 128. Intensywność kolorów (0 = czarno-białe).
cap.set(cv2.CAP_PROP_SATURATION, 0)

# Ostrość (Sharpness): 0 do 6 (zwykle niska wartość, np. 3-10, jest najlepsza).
cap.set(cv2.CAP_PROP_SHARPNESS, 3)

# Gamma: 72 do 500 (standard to 100). Wpływa na jasność średnich tonów.
cap.set(cv2.CAP_PROP_GAMMA, 100)

# Balans bieli (White Balance): 2800 do 6500 (Kelviny). 0 wyłącza automat.
cap.set(cv2.CAP_PROP_AUTO_WB, 0) # Wyłączenie automatu
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)

# Przeciwświetlenie (Backlight Compensation): 0 lub 3. Pomaga, gdy za obiektem jest okno.
cap.set(cv2.CAP_PROP_BACKLIGHT, 1)

# Wzmocnienie (Gain): 0 do ok. 64. Cyfrowe rozjaśnienie (powoduje szumy/ziarno).
cap.set(cv2.CAP_PROP_GAIN, 15)

# Ekspozycja (Exposure): Bardzo ważne dla FPS! Skala logarytmiczna, np. -13 do -1.
# -13 to bardzo krótki czas (ciemno, ale brak smużenia), -4 to długi czas (jasno, ale wolno).
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01) # 0.25 zazwyczaj oznacza tryb manualny w DirectShow
cap.set(cv2.CAP_PROP_EXPOSURE, -13)

# Kompensacja słabego oświetlenia (Low Light Compensation): 0 (Wył) lub 1 (Wł). 
# KONIECZNIE WYŁĄCZ (0) przy 120 FPS, bo automat drastycznie ucina klatki!
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) # Czasem steruje tym flaga auto-exposure

licznik_zdjec = 0
print("Kamera gotowa. 's' - zapisz, 'q' - wyjdź.")

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Kamera - Pelna Kontrola', frame)
    
    klawisz = cv2.waitKey(1) & 0xFF
    if klawisz == ord('q'):
        break
    elif klawisz == ord('s'):
        cv2.imwrite(f"zdjecie_{licznik_zdjec}.jpg", frame)
        licznik_zdjec += 1

cap.release()
cv2.destroyAllWindows()