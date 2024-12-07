import requests
import csv
import sys

# API isteği
response = requests.get("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Istanbul/2023-10-31/2024-10-31?unitGroup=metric&include=days&key=77Q4C3DGJMNUVLFWKSWDURQYX&contentType=csv")

# Yanıt kodunu kontrol et
if response.status_code != 200:
    print(f"Unexpected Status Code: {response.status_code}, Message: {response.text}")
    sys.exit()

# Yanıt kodlamasını Türkçe karakterler için ayarla
response.encoding = 'utf-8'

# CSV verisini işle
CSVText = csv.reader(response.text.splitlines(), delimiter=',', quotechar='"')

# Sonuçları yazdır
for row in CSVText:
    print(row)

# Veriyi dosyaya kaydet
with open('weather_data_corrected.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(csv.reader(response.text.splitlines(), delimiter=',', quotechar='"'))
    print("Düzeltilmiş veriler weather_data_corrected.csv dosyasına kaydedildi.")