# 🌡️ Thermostat Simulation
# About the Project
This project is designed for researchers, engineers and enthusiasts who want to analyze the performances of different control algorithms. The application assists the user in understanding their impact on room temperature control. Performance evaluations reveal under which conditions each algorithm works better.
This interactive application allows you to compare the performance of different control algorithms (On-Off, PID, Q-Learning, Decision Trees) to maintain the temperature in the room. The simulation works with outdoor temperature data and tests various algorithms used to regulate the room temperature according to user-specified parameters.
## Features
- Control room temperature with different control algorithms
- Analyze the performance of algorithms based on outdoor temperature
- Experiment with different simulation parameters
- Review simulation results visually and in CSV format
## Required libraries
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`
This interactive application allows you to compare the performance of different control algorithms (On-Off, PID, Q-Learning, Decision Trees) to maintain the temperature in the room. The simulation works with outdoor temperature data and tests various algorithms used to regulate the room temperature according to user-specified parameters.

## Data Loading

Before running the simulation you need to upload a CSV file containing outdoor temperature data. The data must contain the following columns:

- **Date**: Date (in daily data format)
- **Time**: Time (in time format)
- **Outdoor Temp (C)**: Outdoor temperature values (Celsius)

Once the data is loaded, the graph shows the daily average, minimum and maximum temperatures. This graph allows you to examine the daily changes in outdoor temperature.

## Simulation Parameters

You can set a number of parameters to determine how the simulation will run. These parameters directly affect the way the algorithm works and its performance.

| Parameter | Description |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Beginning Room Temperature (°C)** | Determines the temperature of the room at the start of the simulation. For example, if set to 19°C, the room temperature will be 19°C when the simulation starts. |
| **Thermostat Setting (°C)** | Determines the target temperature setting of the thermostat. The thermostat will try to keep the room at this temperature. For example, if set to 20°C, the algorithms will try to keep the room at 20°C. |
| **Heater Power (°C/minute)** | Determines how much the heater will increase the temperature in the room per minute. When the heater is activated, the room temperature will increase at this rate.
| | **Minimum Off Time (Minutes)** | Determines how long the heater must remain off when turned off. After the heater is switched off, it cannot restart until the specified time has elapsed. This parameter also prevents unnecessarily frequent switching on and off. For example, if it is set to 1 minute, the heater stays off for at least 1 minute after switching off. |
| **Basic Heat Loss (°C/minute)** | Determines how fast the room will cool down under the influence of the outdoor temperature. If the outdoor temperature is lower, the room cools faster.
| **Simulation Duration (Minutes)** | Determines how long the simulation will run. For example, if set to 60 minutes, the simulation will run for this duration. | |
| | **Thermostat Sensitivity (°C)** | Determines how sensitive the thermostat is to temperature changes. The thermostat will turn the heater on when it falls below this value and off when it rises above it.
| **Minimum Run Time (Minutes)** | Determines how long the heater should run at least when turned on. This parameter is used to prevent unnecessarily frequent switching on and off. For example, if set to 1 minute, the heater must run for at least 1 minute after switching on. This helps prevent heaters turning on and off too often and wasting energy.
| | **Minimum Off Time (Minutes)** | Determines how long the heater must remain off when turned off. After the heater is switched off, it cannot restart until the specified time has elapsed. This parameter also prevents unnecessarily frequent switching on and off. For example, if it is set to 1 minute, the heater stays off for at least 1 minute after switching off. |
## Control Algorithms

During the simulation you can select and compare four different control algorithms:

- **On-Off**: The simplest control algorithm. The heater turns on when the room temperature falls below the thermostat setting and turns off when the temperature rises above the setting. This method gives fast results, but tends to enter the on-off cycle frequently.

- **PID (Proportional-Integral-Derivative)**: This is a more advanced algorithm. It takes into account the temperature difference (P), the accumulated error over time (I) and the rate of temperature change (D) to gradually and more stably maintain the room at the desired temperature. Minimizes fluctuation.

- **Q-Learning**: It is a machine learning based algorithm. This algorithm optimizes itself over time to learn temperature control. It learns which action is best in which situation by trial and error.

- **Decision Trees**: A method based on machine learning. It uses historical temperature data to determine which decisions to make to control the temperature in the room. It works with a data-driven approach.

## Analysis of Results

The simulation results are shown in a graph. This graph shows the temperature changes in the room and the performance of the selected algorithm. You can observe how many times the heater is turned on and off during the simulation, how much the room temperature stabilizes and how efficiently the thermostat works.
## Application Screenshots

#### 1. Data Upload and Validation
![Data Upload and Verification](https://github.com/user-attachments/assets/1b48982a-cdda-4b06-99a4-18fcb8043467)

### 2. Simulation Results
![Simulation Results](https://github.com/user-attachments/assets/fcefe19c-df64-4abd-bede-c6d041b7752c)

### 3. Performance Comparisons
![Performance Benchmarks](https://github.com/user-attachments/assets/facb7007-dc3a-4b22-8d43-350706e9175d)

## Installation and Use

1. **Requirements**: 
   - Python 3.x
   - Streamlit
   - Necessary Python libraries such as Pandas, Matplotlib, etc.

2. **Installation**:
   ```bash
   pip install -r requirements.txt








# 🌡️ Termostat Simülasyonu
# Proje Hakkında
Bu proje, farklı kontrol algoritmalarının performanslarını analiz etmek isteyen araştırmacılar, mühendisler ve meraklılar için tasarlanmıştır. Uygulama, oda sıcaklığı kontrolü üzerindeki etkilerini anlamada kullanıcıya yardımcı olur. Performans değerlendirmeleri, her algoritmanın hangi koşullar altında daha iyi çalıştığını ortaya koyar.

Bu interaktif uygulama, odadaki sıcaklığı korumak için farklı kontrol algoritmalarının (Açma-Kapama, PID, Q-Öğrenme, Karar Ağaçları) performansını karşılaştırmanızı sağlar. Simülasyon, dış ortam sıcaklığı verileri ile çalışır ve kullanıcıların belirlediği parametrelere göre oda sıcaklığını düzenlemek için kullanılan çeşitli algoritmaları test eder.

## Özellikler

- Farklı kontrol algoritmaları ile oda sıcaklığını kontrol edin
- Dış ortam sıcaklığına göre algoritmaların performansını analiz edin
- Farklı simülasyon parametreleri ile deneyler yapın
- Simülasyon sonuçlarını görsel ve CSV formatında inceleyin

## Gereksi kütüphaneler

- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`

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

