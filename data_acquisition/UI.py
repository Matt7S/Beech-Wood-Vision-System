import cv2
import os
import time
import csv
import numpy as np

# --- 1. KONFIGURACJA KATEGORII ---
categories = {
    '0': '0_tasmociag',
    '1': '1_zdrowe_drewno',
    '2': '2_sek',
    '3': '3_pekniecie',
    '4': '4_okrajka',
    '5': '5_zgnilizna',
    '6': '6_przebarwienie',     
    '7': '7_ubytek_mechaniczny',
    '8': '8_zabrudzenie',
    '9': '9_inne'
}

base_dir = "dataset_beech"
log_file = os.path.join(base_dir, "rejestr_zdjec.csv")

os.makedirs(base_dir, exist_ok=True)
for key, name in categories.items():
    os.makedirs(os.path.join(base_dir, name), exist_ok=True)

if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Nazwa_zdjecia", "Kategoria", "Komentarz"])

# --- 2. INICJALIZACJA KAMERY ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cap.set(cv2.CAP_PROP_FPS, 120)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)
cap.set(cv2.CAP_PROP_CONTRAST, 32)
cap.set(cv2.CAP_PROP_HUE, 0)
cap.set(cv2.CAP_PROP_SATURATION, 0)
cap.set(cv2.CAP_PROP_SHARPNESS, 3)
cap.set(cv2.CAP_PROP_GAMMA, 100)
cap.set(cv2.CAP_PROP_AUTO_WB, 0) 
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)
cap.set(cv2.CAP_PROP_BACKLIGHT, 1)
cap.set(cv2.CAP_PROP_GAIN, 15)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.001)
cap.set(cv2.CAP_PROP_EXPOSURE, -13)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 

# --- ZMIENNE GLOBALNE ---
LEFT_PANEL_WIDTH = 250

is_typing_note = False
current_note = ""
pending_filename = ""
pending_category = ""
pending_frame = None 

last_save_time = 0
last_saved_msg = ""

button_boxes = {}
save_btn_box = (0, 0, 0, 0)
cancel_btn_box = (0, 0, 0, 0)
clicked_action = None 

# --- OBSŁUGA MYSZKI ---
def mouse_click(event, x, y, flags, param):
    global clicked_action
    if event == cv2.EVENT_LBUTTONDOWN:
        if not is_typing_note:
            for key, (x1, y1, x2, y2) in button_boxes.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    clicked_action = key
        else:
            x1, y1, x2, y2 = save_btn_box
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_action = 'ENTER'
            
            cx1, cy1, cx2, cy2 = cancel_btn_box
            if cx1 <= x <= cx2 and cy1 <= y <= cy2:
                clicked_action = 'CANCEL'

okno_nazwa = 'Akwizycja Danych - Buk'
cv2.namedWindow(okno_nazwa)
cv2.setMouseCallback(okno_nazwa, mouse_click)

print("Kamera gotowa. Panel sterowania znajduje się po lewej stronie obrazu.")

# --- 3. GŁÓWNA PĘTLA ---
while(True):
    if not is_typing_note:
        ret, frame = cap.read()
        if not ret:
            print("Błąd odczytu klatki!")
            break
        display_frame = frame.copy()
    else:
        display_frame = pending_frame.copy()
        
    h, w = display_frame.shape[:2]
    
    # Utworzenie czarnego płótna z miejscem na lewy panel
    canvas = np.zeros((h, w + LEFT_PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, LEFT_PANEL_WIDTH:] = display_frame 
    
    # Rysowanie tła panelu bocznego
    cv2.rectangle(canvas, (0, 0), (LEFT_PANEL_WIDTH, h), (20, 20, 20), -1)
    cv2.line(canvas, (LEFT_PANEL_WIDTH, 0), (LEFT_PANEL_WIDTH, h), (255, 255, 255), 2)

    # --- RYSOWANIE INTERFEJSU (PANEL BOCZNY) ---
    if not is_typing_note:
        button_boxes.clear()
        
        cv2.putText(canvas, "Zapisz do:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_offset = 60
        for key in sorted(categories.keys()):
            name = categories[key]
            display_name = name.split('_', 1)[1] if '_' in name else name
            tekst = f"[{key}] {display_name}"
            
            x1, y1 = 10, y_offset
            x2, y2 = LEFT_PANEL_WIDTH - 10, y_offset + 35
            
            # Subtelniejsze przyciski ułożone w kolumnie
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 50, 50), -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cv2.putText(canvas, tekst, (x1 + 5, y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            button_boxes[key] = (x1, y1, x2, y2)
            
            y_offset += 45
            
        # Przycisk wyjścia na samym dole panelu
        q_y1 = h - 50
        cv2.rectangle(canvas, (10, q_y1), (LEFT_PANEL_WIDTH - 10, q_y1 + 35), (50, 0, 0), -1)
        cv2.rectangle(canvas, (10, q_y1), (LEFT_PANEL_WIDTH - 10, q_y1 + 35), (200, 0, 0), 1)
        cv2.putText(canvas, "[q] wyjdz", (15, q_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
            
        if time.time() - last_save_time < 1.5:
            cv2.putText(canvas, last_saved_msg, (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            
    else:
        # Tryb wpisywania notatki w bocznym panelu
        kursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        display_pending_cat = pending_category.split('_', 1)[1] if '_' in pending_category else pending_category
        
        cv2.putText(canvas, "Wybrano:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(canvas, display_pending_cat, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(canvas, "Notatka:", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Proste zawijanie tekstu dla długich notatek w wąskim panelu
        def draw_text_wrapped(img, text, x, y, max_width, font, font_scale, color, thickness):
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + word + " "
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
                if text_width > max_width and current_line != "":
                    lines.append(current_line)
                    current_line = word + " "
                else:
                    current_line = test_line
            lines.append(current_line)
            
            for i, line in enumerate(lines):
                cv2.putText(img, line, (x, y + i * 25), font, font_scale, color, thickness)
                
        draw_text_wrapped(canvas, f"{current_note}{kursor}", 10, 140, LEFT_PANEL_WIDTH - 20, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

        # Przyciski kontrolne
        cx1, cy1, cx2, cy2 = 10, h - 110, LEFT_PANEL_WIDTH - 10, h - 70
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 0, 150), -1)
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (255, 255, 255), 1)
        cv2.putText(canvas, "[ESC] WSTECZ", (cx1 + 15, cy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cancel_btn_box = (cx1, cy1, cx2, cy2)
        
        zx1, zy1, zx2, zy2 = 10, h - 55, LEFT_PANEL_WIDTH - 10, h - 15
        cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (0, 150, 0), -1)
        cv2.rectangle(canvas, (zx1, zy1), (zx2, zy2), (255, 255, 255), 1)
        cv2.putText(canvas, "[ENTER] ZAPISZ", (zx1 + 15, zy1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        save_btn_box = (zx1, zy1, zx2, zy2)

    cv2.imshow(okno_nazwa, canvas)
    
    # --- PRZECHWYTYWANIE AKCJI ---
    klawisz = cv2.waitKey(33) & 0xFF
    wybrana_kategoria = None
    zapisz_notatke = False
    anuluj_notatke = False

    if klawisz != 255:
        if is_typing_note:
            if klawisz == 13: 
                zapisz_notatke = True
            elif klawisz == 27: 
                anuluj_notatke = True
            elif klawisz == 8: 
                current_note = current_note[:-1]
            elif 32 <= klawisz <= 126:
                current_note += chr(klawisz)
        else:
            if klawisz == ord('q'):
                break
            char_klawisz = chr(klawisz) if klawisz < 256 else ""
            if char_klawisz in categories:
                wybrana_kategoria = char_klawisz

    if clicked_action:
        if clicked_action == 'ENTER' and is_typing_note:
            zapisz_notatke = True
        elif clicked_action == 'CANCEL' and is_typing_note:
            anuluj_notatke = True
        elif clicked_action in categories and not is_typing_note:
            wybrana_kategoria = clicked_action
        clicked_action = None 

    # --- WYKONYWANIE ZADAŃ ---
    if zapisz_notatke:
        pelna_sciezka = os.path.join(base_dir, pending_category, pending_filename)
        cv2.imwrite(pelna_sciezka, pending_frame) 
        
        with open(log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([pending_filename, pending_category, current_note])
            
        is_typing_note = False
        pending_frame = None
        last_save_time = time.time()
        last_saved_msg = "Zapisano!"

    if anuluj_notatke:
        is_typing_note = False
        pending_frame = None
        current_note = ""
        last_save_time = time.time()
        last_saved_msg = "Anulowano."

    if wybrana_kategoria:
        cat_name = categories[wybrana_kategoria]
        timestamp = int(time.time() * 1000)
        
        pending_filename = f"{cat_name}_{timestamp}.jpg"
        pending_category = cat_name
        pending_frame = frame.copy() 
        
        is_typing_note = True
        current_note = ""

cap.release()
cv2.destroyAllWindows()