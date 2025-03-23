import cv2
import pytesseract
import numpy as np
import re

# Tesseract yolu (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image):
    """Görüntüyü gri tonlamaya, bulanıklaştırmaya ve kenar tespitine hazırlar."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    return cleaned

def find_plate_contour(edges, frame):
    """Plaka konturunu bulur ve plakayı kırpar."""
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 2 < aspect_ratio < 6:  # Plaka oranı kontrolü
                return frame[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def read_plate(plate_region):
    """Plaka alanındaki metni okur."""
    if plate_region is None:
        return None
    plate_gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    plate_enhanced = cv2.adaptiveThreshold(plate_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    text = pytesseract.image_to_string(plate_enhanced, config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    text = re.sub(r'[^A-Z0-9]', '', text)  # Sadece harf ve rakamları al
    
    if 7 <= len(text) <= 8:  # 7 veya 8 karakterli plakaları kabul et
        return text
    return None

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Kamera açılışı
    if not cap.isOpened():
        print("Hata: Kamera görüntüsü alınamadı!")
        return
    
    last_plate = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Hata: Kamera görüntüsü alınamadı!")
            break
        
        edges = preprocess_image(frame)
        plate_region, plate_coords = find_plate_contour(edges, frame)
        plate_text = read_plate(plate_region)

        if plate_text and plate_text != last_plate:  # Eğer plaka okunduysa
            last_plate = plate_text  # Son okunan plakayı sakla
            print(f"Plaka Okundu: {plate_text}")  # Konsola yazdır

        # Plakayı gerçek zamanlı ekranda göster
        if last_plate:
            cv2.putText(frame, f"Plaka: {last_plate}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if plate_coords:  # Eğer plaka bulunduysa çerçeve çiz
            x, y, w, h = plate_coords
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Plaka Okuma Sistemi', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
