# Oda Sıcaklığı Kontrolü İçin Simülasyon Uygulaması

Bu Streamlit uygulaması, oda sıcaklığını korumak amacıyla çeşitli kontrol algoritmalarının (Açma-Kapama, PID ve Q-Öğrenme) performansını karşılaştırmak için tasarlanmıştır. Uygulama, kullanıcıların farklı simülasyon parametreleri girerek bu algoritmaların sonuçlarını görselleştirmesine ve analiz etmesine olanak tanır.

## Özellikler

- **Veri Yükleme ve Doğrulama**: Kullanıcıdan dış ortam sıcaklığı verilerini içeren bir CSV dosyası yüklemesi istenir. Dosya doğrulanır ve eksik değerler veya hatalar için kullanıcıya uyarılar verilir. Alternatif olarak, kullanıcı kübik spline interpolasyonu kullanarak dış ortam sıcaklıklarını manuel olarak girebilir.
  
- **Simülasyon Parametreleri**: 
  - Başlangıç oda sıcaklığı
  - Termostat ayarı
  - Isıtıcı gücü
  - Temel ısı kaybı
  - Simülasyon süresi
  - PID ve Q-Öğrenme algoritmaları için ayrı parametreler

- **Simülasyonları Çalıştırma**:
  - **Açma-Kapama Kontrolü**: Isıtıcıyı açma-kapama mantığına dayalı olarak çalıştırır ve oda sıcaklığını kontrol eder.
  - **PID Kontrolü**: Proportional, Integral ve Derivative terimlerini hesaplayarak ısıtıcıya uygulanan gücü ayarlar.
  - **Q-Öğrenme**: Q-öğrenme algoritması kullanarak oda sıcaklığını kontrol eder. Eğitim süreci sonunda öğrenilen politika ile simülasyon gerçekleştirilir.

- **Sonuçların Görselleştirilmesi**: 
  - Oda sıcaklığı, termostat ayarı ve kontrol algoritmalarının performansını karşılaştıran grafikler
  - Konfor ve enerji metrikleri için bar grafikleri
  - Tüm sonuçlar CSV formatında indirilebilir

## Uygulama Ekran Görüntüleri

### 1. Veri Yükleme ve Doğrulama
![Veri Yükleme ve Doğrulama](https://github.com/user-attachments/assets/1b48982a-cdda-4b06-99a4-18fcb8043467)

### 2. Simülasyon Sonuçları
![Simülasyon Sonuçları](https://github.com/user-attachments/assets/fcefe19c-df64-4abd-bede-c6d041b7752c)

### 3. Performans Karşılaştırmaları
![Performans Karşılaştırmaları](https://github.com/user-attachments/assets/facb7007-dc3a-4b22-8d43-350706e9175d)

## Kurulum ve Kullanım

1. **Gereksinimler**: 
   - Python 3.x
   - Streamlit
   - Pandas, Matplotlib, vb. gibi gerekli Python kütüphaneleri

2. **Kurulum**:
   ```bash
   pip install -r requirements.txt
