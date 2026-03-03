from PIL import Image

def merge_to_color(red_file, green_file, blue_file, output_file):
    # 1. Wczytanie zdjęć i konwersja do czystej skali szarości (tryb 'L' - Luminance)
    img_red = Image.open(red_file).convert('L')
    img_green = Image.open(green_file).convert('L')
    img_blue = Image.open(blue_file).convert('L')

    # 2. Sprawdzenie rozmiarów (kanały R, G, B muszą mieć dokładnie ten sam wymiar)
    if img_red.size != img_green.size or img_red.size != img_blue.size:
        print("Zdjęcia mają różne rozmiary. Skaluję je do rozmiaru pierwszego zdjęcia...")
        img_green = img_green.resize(img_red.size, Image.Resampling.LANCZOS)
        img_blue = img_blue.resize(img_red.size, Image.Resampling.LANCZOS)

    # 3. Połączenie 3 obrazów w jedno zdjęcie RGB
    # Ważne: (Red, Green, Blue)
    merged_image = Image.merge('RGB', (img_red, img_green, img_blue))

    # 4. Zapisanie gotowego pliku
    merged_image.save(output_file)
    print(f"Gotowe! Połączone zdjęcie zapisano jako: {output_file}")

# Wywołanie funkcji dla Twoich plików
merge_to_color('red.jpg', 'green.jpg', 'blue.jpg', 'polaczone_rgb.jpg')