# 🌡️ Termostat Simülasyonu

Bu interaktif uygulama, odadaki sıcaklığı korumak için farklı kontrol algoritmalarının (Açma-Kapama, PID, Q-Öğrenme, Karar Ağaçları) performansını karşılaştırmanızı sağlar. Simülasyon, dış ortam sıcaklığı verileri ile çalışır ve kullanıcıların belirlediği parametrelere göre oda sıcaklığını düzenlemek için kullanılan çeşitli algoritmaları test eder.

## Özellikler

- Farklı kontrol algoritmaları ile oda sıcaklığını kontrol edin
- Dış ortam sıcaklığına göre algoritmaların performansını analiz edin
- Farklı simülasyon parametreleri ile deneyler yapın
- Simülasyon sonuçlarını görsel ve CSV formatında inceleyin

## Gereksinimler

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`

### Kurulum

Uygulamayı çalıştırmak için gerekli paketleri yükleyin:

```bash
pip install streamlit pandas numpy matplotlib scipy scikit-learn
# 🌡️ Termostat Simülasyonu

Bu interaktif uygulama, odadaki sıcaklığı korumak için farklı kontrol algoritmalarının (Açma-Kapama, PID, Q-Öğrenme, Karar Ağaçları) performansını karşılaştırmanızı sağlar. Simülasyon, dış ortam sıcaklığı verileri ile çalışır ve kullanıcıların belirlediği parametrelere göre oda sıcaklığını düzenlemek için kullanılan çeşitli algoritmaları test eder.

## Veri Yükleme

Simülasyonu çalıştırmadan önce dış ortam sıcaklık verilerini içeren bir CSV dosyası yüklemeniz gerekir. Veride aşağıdaki sütunlar bulunmalıdır:

- **Date**: Tarih (günlük veri formatında)
- **Time**: Saat (zaman formatında)
- **Outdoor Temp (C)**: Dış ortam sıcaklık değerleri (Celsius)

Veri yüklendikten sonra, grafik üzerinde günlük ortalama, minimum ve maksimum sıcaklıklar gösterilir. Bu grafik, dış ortam sıcaklığındaki günlük değişimleri incelemenizi sağlar.

## Simülasyon Parametreleri

Simülasyonun nasıl çalışacağını belirlemek için bir dizi parametreyi ayarlayabilirsiniz. Bu parametreler, algoritmanın çalışma şeklini ve performansını doğrudan etkiler.

| Parametre                       | Açıklama                                                                                                                                                    |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Başlangıç Oda Sıcaklığı (°C)** | Simülasyonun başlangıcında odanın sıcaklığını belirler. Örneğin, 19°C olarak ayarlanmışsa, simülasyon başladığında oda sıcaklığı 19°C olacaktır.             |
| **Termostat Ayarı (°C)**         | Termostatın hedef sıcaklık ayarını belirler. Termostat, odayı bu sıcaklıkta tutmaya çalışacaktır. Örneğin, 20°C olarak ayarlanmışsa, algoritmalar odayı 20°C'de tutmaya çalışacaktır. |
| **Isıtıcı Gücü (°C/dakika)**     | Isıtıcının odadaki sıcaklığı dakikada ne kadar artıracağını belirler. Isıtıcı devreye girdiğinde, oda sıcaklığı bu hızla artar.                              |
| **Temel Isı Kaybı (°C/dakika)**  | Odanın dış ortam sıcaklığının etkisiyle ne kadar hızla soğuyacağını belirler. Dış ortam sıcaklığı daha düşükse oda daha hızlı soğur.                         |
| **Simülasyon Süresi (Dakika)**   | Simülasyonun ne kadar süre boyunca çalışacağını belirler. Örneğin, 60 dakika olarak ayarlanırsa, simülasyon bu süre boyunca çalışacaktır.                   |
| **Termostat Hassasiyeti (°C)**   | Termostatın sıcaklık değişimlerine ne kadar hassas olduğunu belirler. Termostat, bu değerin altına düştüğünde ısıtıcıyı açar, üstüne çıktığında ise kapatır. |
| **Minimum Çalışma Süresi (Dakika)** | Isıtıcının açıldığında en az ne kadar süre çalışması gerektiğini belirler. Bu parametre, gereksiz yere sık açılıp kapanmayı önlemek için kullanılır. Örneğin, 1 dakika olarak ayarlanırsa, ısıtıcı açıldıktan sonra en az 1 dakika boyunca çalışmak zorundadır. Bu, ısıtıcıların çok sık açılıp kapanmasını ve enerji israfını önlemeye yardımcı olur.|
| **Minimum Kapalı Kalma Süresi (Dakika)** | Isıtıcının kapandığında ne kadar süre kapalı kalması gerektiğini belirler. Isıtıcı kapandıktan sonra, belirlenen süre dolmadan tekrar çalışamaz. Bu parametre de gereksiz yere sık devreye girip çıkmayı engeller. Örneğin, 1 dakika olarak ayarlanmışsa, ısıtıcı kapandıktan sonra en az 1 dakika kapalı kalır. |

## Kontrol Algoritmaları

Simülasyon sırasında dört farklı kontrol algoritmasını seçip karşılaştırabilirsiniz:

- **Açma-Kapama (On-Off)**: En basit kontrol algoritmasıdır. Oda sıcaklığı, termostat ayarının altına düştüğünde ısıtıcı açılır, sıcaklık ayarın üzerine çıktığında ısıtıcı kapanır. Bu yöntem hızlı sonuç verir, ancak sık sık açma-kapama döngüsüne girme eğilimindedir.

- **PID (Proportional-Integral-Derivative)**: Daha gelişmiş bir algoritmadır. Sıcaklık farkını (P), zaman içinde birikmiş hatayı (I) ve sıcaklık değişim hızını (D) göz önünde bulundurarak odayı kademeli ve daha stabil bir şekilde istenen sıcaklıkta tutar. Dalgalanmayı en aza indirir.

- **Q-Öğrenme**: Makine öğrenimi tabanlı bir algoritmadır. Bu algoritma, sıcaklık kontrolünü öğrenmek için zamanla kendini optimize eder. Deneme-yanılma yöntemiyle hangi durumda hangi eylemin en iyi olduğunu öğrenir.

- **Karar Ağaçları**: Makine öğrenimine dayalı bir yöntemdir. Geçmiş sıcaklık verilerini kullanarak odadaki sıcaklığı kontrol etmek için hangi kararların alınacağını belirler. Veri tabanlı bir yaklaşımla çalışır.

## Sonuçların Analizi

Simülasyon sonuçları bir grafik ile gösterilir. Bu grafik, odadaki sıcaklık değişimlerini ve seçilen algoritmanın performansını gözler önüne serer. Simülasyon sırasında kaç defa ısıtıcının açılıp kapandığını, oda sıcaklığının ne kadar stabilleştiğini ve termostatın ne kadar verimli çalıştığını gözlemleyebilirsiniz.

Simülasyon sonuçlarını CSV dosyası olarak indirebilir ve daha ayrıntılı analizler yapabilirsiniz.

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

# Proje Hakkında
Bu proje, farklı kontrol algoritmalarının performanslarını analiz etmek isteyen araştırmacılar, mühendisler ve meraklılar için tasarlanmıştır. Uygulama, oda sıcaklığı kontrolü üzerindeki etkilerini anlamada kullanıcıya yardımcı olur. Performans değerlendirmeleri, her algoritmanın hangi koşullar altında daha iyi çalıştığını ortaya koyar.
