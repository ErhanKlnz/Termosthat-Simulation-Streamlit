Bu Streamlit uygulaması, oda sıcaklığını korumak için farklı kontrol algoritmalarının performansını karşılaştırmak amacıyla tasarlanmıştır. Kullanıcı, çeşitli simülasyon parametrelerini girerek Açma-Kapama, Q-Öğrenme ve PID kontrol algoritmalarıyla gerçekleştirilmiş simülasyonların sonuçlarını görselleştirebilir. Uygulama, kullanıcıdan dış ortam sıcaklığı verilerini yüklemesini veya kendi belirlediği interpolasyon verilerini kullanmasını sağlar.


Veri Yükleme ve Doğrulama:

Kullanıcıdan dış ortam sıcaklığı verilerini içeren bir CSV dosyası yüklemesi istenir. Dosya doğrulanır ve eksik değerler veya hatalar için uyarılar verilir. Alternatif olarak, kullanıcı interpolasyon (kübik spline) kullanarak dış ortam sıcaklıklarını girebilir.
Simülasyon Parametrelerinin Alınması:

Kullanıcı, simülasyonun başlangıç oda sıcaklığı, termostat ayarı, ısıtıcı gücü, temel ısı kaybı, simülasyon süresi ve diğer parametreleri belirler. PID ve Q-Öğrenme algoritmaları için de ayrı parametreler belirlenir.
Simülasyonları Çalıştırma:

Açma-Kapama Kontrolü: Isıtıcı açma ve kapama mantığıyla oda sıcaklığını simüle eder. Termostatın hassasiyetine göre ısıtıcıyı açar veya kapatır. Simülasyon süresince oda sıcaklığı ve ısıtıcı durumları takip edilir.
PID Kontrolü: PID algoritması kullanılarak oda sıcaklığı kontrol edilir. Proportional, Integral ve Derivative terimleri hesaplanır ve ısıtıcıya uygulanan güç ayarlanır.
Q-Öğrenme: Oda sıcaklığı Q-öğrenme algoritmasıyla kontrol edilir. Epsilon-greedy politikası ile aksiyonlar seçilir ve Q-tablosu güncellenir. Eğitim süreci sonrasında öğrenilen politika ile final simülasyonu yapılır.
Sonuçların Görselleştirilmesi:

Oda sıcaklığı, termostat ayarı ve kontrol algoritmalarının performansını karşılaştıran grafikler oluşturulur. Ayrıca, konfor ve enerji metrikleri için bar grafikleri sunulur.
Sonuçlar CSV formatında indirilebilir hale getirilir.
Sonuç
Uygulama, kullanıcıların farklı kontrol algoritmalarının oda sıcaklığı üzerindeki etkilerini karşılaştırmasını sağlar. Bu sayede, her algoritmanın konfor alanı, aşım (overshoot) ve alt geçiş (undershoot) performansı analiz edilebilir. Sonuçlar, kullanıcıya hangi kontrol yönteminin belirli koşullar altında daha etkili olduğunu anlamasında yardımcı olur. Grafikler ve CSV çıktıları, performans değerlendirmelerini kolaylaştırır ve verilerin daha detaylı analiz edilmesine olanak tanır.