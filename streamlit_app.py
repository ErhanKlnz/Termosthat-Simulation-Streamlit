import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import time

# Uygulama Başlığı ve Açıklaması

st.set_page_config(page_title="Termostat Simülasyonu", page_icon="🌡️", layout="wide")

st.title("Termostat Simülasyonu")
st.subheader("Kontrol Algoritmalarının Karşılaştırılması")
st.write("Bu interaktif simülasyon, oda sıcaklığını korumak için farklı kontrol algoritmalarının performansını karşılaştırır.")

# Dosya Yükleyici ve Hata Kontrolü

def load_data():
    """Kullanıcıdan CSV dosyası yükler, veri çerçevesine dönüştürür ve doğrular."""
    uploaded_file = st.file_uploader("Bir CSV dosyası seçin (Dış Ortam Sıcaklığı verilerini içeren)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # CSV dosyasının doğruluğunu kontrol et
            if 'Outdoor Temp (C)' not in df.columns:
                st.error("CSV dosyası 'Outdoor Temp (C)' sütununu içermiyor. Lütfen doğru dosyayı yükleyin.")
                return None
            if df.isnull().values.any():
                st.error("CSV dosyasında eksik değerler var. Lütfen düzeltin ve tekrar yükleyin.")
                return None
            outdoor_temp_values = df['Outdoor Temp (C)'].values  
            return outdoor_temp_values
        except pd.errors.EmptyDataError:
            st.error("Yüklenen CSV dosyası boş. Lütfen geçerli bir dosya yükleyin.")
            return None
        except Exception as e:
            st.error(f"CSV dosyası okunurken bir hata oluştu: {e}")
            return None
    else:
        st.warning("Lütfen devam etmek için bir CSV dosyası yükleyin.")
        return None

# Simülasyon Parametreleri 
def get_simulation_parameters():
    """Kullanıcıdan simülasyon parametrelerini alır (varsayılan değerlerle)."""
    st.sidebar.header("Simülasyon Parametreleri")
    initial_room_temperature = st.sidebar.number_input("Başlangıç Oda Sıcaklığı (°C)", min_value=10, max_value=30, value=19)
    thermostat_setting = st.sidebar.number_input("Termostat Ayarı (°C)", min_value=15, max_value=25, value=20)
    heater_power = st.sidebar.slider("Isıtıcı Gücü (°C/dakika)", min_value=0.1, max_value=0.5, value=0.3)
    base_heat_loss = st.sidebar.slider("Temel Isı Kaybı (°C/dakika)", min_value=0.05, max_value=0.2, value=0.1)
    simulation_minutes = st.sidebar.number_input("Simülasyon Süresi (Dakika)", min_value=10, max_value=5000, value=60)
    thermostat_sensitivity = st.sidebar.slider("Termostat Hassasiyeti (°C)", min_value=0.1, max_value=0.5, value=0.5, step=0.1)
    min_run_time = st.sidebar.number_input("Minimum Çalışma Süresi (Dakika)", min_value=0.2, max_value=10.0, value=2.0, step=0.1)
    return {
        'initial_room_temperature': initial_room_temperature,
        'thermostat_setting': thermostat_setting,
        'heater_power': heater_power,
        'base_heat_loss': base_heat_loss,
        'simulation_minutes': simulation_minutes,
        'thermostat_sensitivity': thermostat_sensitivity,
        'min_run_time': min_run_time  # New parameter for minimum run time
    }

# Q-Öğrenme Parametreleri 

def get_q_learning_parameters():
    """Kullanıcıdan Q-öğrenme parametrelerini alır (varsayılan değerlerle)."""
    st.sidebar.subheader("Q-Öğrenme Parametreleri")
    episodes = st.sidebar.number_input("Eğitim Bölümleri", min_value=100, max_value=5000, value=1000)
    learning_rate = st.sidebar.slider("Öğrenme Oranı", min_value=0.01, max_value=1.0, value=0.1)
    discount_factor = st.sidebar.slider("İndirim Faktörü", min_value=0.01, max_value=1.0, value=0.95)
    exploration_rate = st.sidebar.slider("Keşif Oranı", min_value=0.01, max_value=1.0, value=0.1)
    return {
        'episodes': episodes,
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'exploration_rate': exploration_rate
    }

# PID Parametreleri 

def get_pid_parameters():
    """Kullanıcıdan PID parametrelerini alır (varsayılan değerlerle)."""
    st.sidebar.subheader("PID Parametreleri")
    Kp = st.sidebar.slider("Kp (Oransal Kazanç)", min_value=0.1, max_value=20.0, value=1.0)
    Ki = st.sidebar.slider("Ki (İntegral Kazanç)", min_value=0.01, max_value=1.0, value=0.1)
    Kd = st.sidebar.slider("Kd (Türev Kazanç)", min_value=0.001, max_value=1.0, value=0.01)
    return {
        'Kp': Kp,
        'Ki': Ki,
        'Kd': Kd
    }

# Yardımcı Fonksiyonlar 

def get_state(temperature):
    """Sıcaklığı durumlara ayırır."""
    return int(min(40, max(0, (temperature - 10) / 0.5)))

def get_action(state, q_table, exploration_rate):
    """Epsilon-greedy politikasına göre bir eylem seçer."""
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.choice(num_actions)  # Exploration
    else:
        return np.argmax(q_table[state, :])  # Exploitation

def get_reward(state, action, thermostat_setting):
    """Durum ve eyleme göre ödülü hesaplar."""
    state_temp = 10 + state * 0.5

    if abs(state_temp - thermostat_setting) <= 0.5:
        return 10  # İstenen aralıkta
    elif action == 1 and state_temp > thermostat_setting + 0.5:  # Çok sıcak
        return -10
    elif action == 0 and state_temp < thermostat_setting - 0.5:  # Çok soğuk
        return -5
    else:
        return -1  # Aralıkta olmamanın cezası

# Dış Ortam Sıcaklığı Seçimi ve İnterpolasyon

def get_outdoor_temperature_data():
    """Kullanıcıdan dış ortam sıcaklığı verilerini alır (CSV veya interpolasyon)."""
    data_source = st.sidebar.radio("Dış Ortam Sıcaklığı Veri Kaynağı", ["CSV Dosyası", "İnterpolasyon (Kübik Spline)"])

    if data_source == "CSV Dosyası":
        outdoor_temp_values = load_data()
        return outdoor_temp_values, None  # İnterpolasyon fonksiyonu None olarak döndürülür

    elif data_source == "İnterpolasyon (Kübik Spline)":
        st.sidebar.subheader("İnterpolasyon Verileri")
        
        # Session state kullanarak sıcaklık değerlerini sakla
        if 'temperatures' not in st.session_state:
            st.session_state.temperatures = [20, 20, 20, 20, 20]  # Varsayılan değerler

        with st.form("interpolation_form"):
            hours = [0, 6, 12, 18, 24]
            for i, hour in enumerate(hours):
                st.session_state.temperatures[i] = st.number_input(f"{hour}:00 Sıcaklığı (°C)", value=st.session_state.temperatures[i], min_value=-20, max_value=50)

            submitted = st.form_submit_button("İnterpolasyonu Uygula")
            if submitted:
                # Güncel sıcaklık değerlerini kullanarak interpolasyon fonksiyonunu oluştur
                interpolation_func = CubicSpline(hours, st.session_state.temperatures)
                return None, interpolation_func 

        # Form gönderilmediyse veya ilk çalıştırmada, session state'deki değerlerle interpolasyon fonksiyonu döndür
        return None, CubicSpline(hours, st.session_state.temperatures)

def get_outdoor_temp(minute, outdoor_temp_values, interpolation_func):
    """Belirli bir dakika için dış ortam sıcaklığını alır (CSV veya interpolasyon)."""
    if outdoor_temp_values is not None:
        # CSV verileri kullanılıyorsa
        index = int(minute // 5) 
        return outdoor_temp_values[min(index, len(outdoor_temp_values) - 1)]
    elif interpolation_func is not None:
        # İnterpolasyon kullanılıyorsa
        hour = minute / 60  # Dakikayı saate çevir
        return float(interpolation_func(hour))  # İnterpolasyon fonksiyonundan sıcaklığı al

# Alan Hesaplama Fonksiyonları

def calculate_area_between_temp(time, room_temperatures, set_temp):
    """Mevcut sıcaklık ve ayarlanan sıcaklık arasındaki alanı hesaplar."""
    area = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        area += abs(avg_temp - set_temp) * dt
    return area

def calculate_overshoot_area(time, room_temperatures, set_temp):
    """Aşım alanını hesaplar."""
    overshoot = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        if avg_temp > set_temp:
            overshoot += (avg_temp - set_temp) * dt
    return overshoot

def calculate_undershoot_area(time, room_temperatures, set_temp):
    """Alt geçiş alanını hesaplar."""
    undershoot = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        if avg_temp < set_temp:
            undershoot += (set_temp - avg_temp) * dt
    return undershoot

# Simülasyon Mantığı (Açma-Kapama) 
def run_on_off_simulation(params, outdoor_temp_values, interpolation_func):
    """Açma-kapama kontrol algoritması ile oda sıcaklığı simülasyonunu çalıştırır."""
    time = []
    room_temperatures = []
    heater_status = False
    heater_on_duration = 0  # Track how long the heater has been on
    heater_on_off_cycles = 0
    room_temperature = params['initial_room_temperature']

    for minute in np.arange(0, params['simulation_minutes'], 0.1):
        time.append(minute)
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)

        if heater_status:
            heater_on_duration += 0.1  # Increase heater on duration

        # Check if heater should turn on
        if room_temperature < params['thermostat_setting'] - params['thermostat_sensitivity'] and not heater_status:
            heater_status = True
            heater_on_duration = 0  # Reset duration counter when heater turns on
            heater_on_off_cycles += 1  # Increment the cycle count

        # Check if heater should turn off
        elif room_temperature > params['thermostat_setting'] + params['thermostat_sensitivity'] and heater_status and heater_on_duration >= params['min_run_time']:
            heater_status = False

        heat_loss = params['base_heat_loss'] * (room_temperature - outside_temperature) / 10

        if heater_status:
            room_temperature += params['heater_power'] * 0.1
        else:
            room_temperature -= heat_loss * 0.1

        room_temperatures.append(room_temperature)

    comfort_area = calculate_area_between_temp(time, room_temperatures, params['thermostat_setting'])
    overshoot = calculate_overshoot_area(time, room_temperatures, params['thermostat_setting'])
    undershoot = calculate_undershoot_area(time, room_temperatures, params['thermostat_setting'])
    return {
        'time': time,
        'room_temperatures': room_temperatures,
        'comfort_area': comfort_area,
        'overshoot': overshoot,
        'undershoot': undershoot,
        'on_off_cycles': heater_on_off_cycles  # Return the on-off cycle count
    }


# Simülasyon Mantığı (Q-Öğrenme) 

# Global variables for Q-learning
num_states = 41
num_actions = 2
q_table = np.zeros((num_states, num_actions))
# Update Q-Learning Simulation
def run_q_learning_simulation(params, outdoor_temp_values, q_params, interpolation_func):
    """Q-öğrenme kontrol algoritması ile oda sıcaklığı simülasyonunu çalıştırır."""
    global q_table
    heater_on_off_cycles = 0

    for episode in range(q_params['episodes']):
        room_temperature = params['initial_room_temperature']
        state = get_state(room_temperature)
        heater_status = False
        heater_on_duration = 0

        for minute in np.arange(0, params['simulation_minutes'], 0.1):
            outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)
            action = get_action(state, q_table, q_params['exploration_rate'])

            if action == 1 and not heater_status:
                heater_status = True
                heater_on_duration = 0
                heater_on_off_cycles += 1

            if heater_status:
                heater_on_duration += 0.1

            if action == 0 and heater_status and heater_on_duration >= params['min_run_time']:
                heater_status = False

            if heater_status:
                room_temperature += params['heater_power'] * 0.1
            else:
                heat_loss = params['base_heat_loss'] * (room_temperature - outside_temperature) / 10
                room_temperature -= heat_loss * 0.1

            next_state = get_state(room_temperature)
            reward = get_reward(next_state, action, params['thermostat_setting'])

            q_table[state, action] += q_params['learning_rate'] * (
                reward + q_params['discount_factor'] * np.max(q_table[next_state, :]) - q_table[state, action]
            )
            state = next_state

    time = []
    room_temperatures = []
    room_temperature = params['initial_room_temperature']
    state = get_state(room_temperature)
    heater_status = False
    heater_on_duration = 0

    for minute in np.arange(0, params['simulation_minutes'], 0.1):
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)
        action = np.argmax(q_table[state, :])

        if action == 1 and not heater_status:
            heater_status = True
            heater_on_duration = 0
            heater_on_off_cycles += 1

        if heater_status:
            heater_on_duration += 0.1

        if action == 0 and heater_status and heater_on_duration >= params['min_run_time']:
            heater_status = False

        if heater_status:
            room_temperature += params['heater_power'] * 0.1
        else:
            heat_loss = params['base_heat_loss'] * (room_temperature - outside_temperature) / 10
            room_temperature -= heat_loss * 0.1

        state = get_state(room_temperature)
        time.append(minute)
        room_temperatures.append(room_temperature)

    comfort_area = calculate_area_between_temp(time, room_temperatures, params['thermostat_setting'])
    overshoot = calculate_overshoot_area(time, room_temperatures, params['thermostat_setting'])
    undershoot = calculate_undershoot_area(time, room_temperatures, params['thermostat_setting'])
    return {
        'time': time,
        'room_temperatures': room_temperatures,
        'comfort_area': comfort_area,
        'overshoot': overshoot,
        'undershoot': undershoot,
        'on_off_cycles': heater_on_off_cycles
    }


def run_pid_simulation(params, outdoor_temp_values, pid_params, interpolation_func):
    """PID kontrol algoritması ile oda sıcaklığı simülasyonunu çalıştırır."""
    time = []
    room_temperatures = []
    heater_output = []
    heater_on_off_cycles = 0

    integral_error = 0
    previous_error = 0
    room_temperature = params['initial_room_temperature']
    heater_status = False
    heater_on_duration = 0

    for minute in np.arange(0, params['simulation_minutes'], 0.1):
        time.append(minute)
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)
        error = params['thermostat_setting'] - room_temperature
        proportional_term = pid_params['Kp'] * error
        integral_error += error * 0.1
        integral_term = pid_params['Ki'] * integral_error
        derivative_term = pid_params['Kd'] * (error - previous_error) / 0.1
        previous_error = error

        pid_output = proportional_term + integral_term + derivative_term
        pid_output = max(0, min(pid_output, 1))
        heater_output.append(pid_output)

        if pid_output > 0.5 and not heater_status:
            heater_status = True
            heater_on_duration = 0
            heater_on_off_cycles += 1

        if heater_status:
            heater_on_duration += 0.1

        if pid_output < 0.1 and heater_status and heater_on_duration >= params['min_run_time']:
            heater_status = False

        heat_loss = params['base_heat_loss'] * (room_temperature - outside_temperature) / 10
        room_temperature += (params['heater_power'] * pid_output - heat_loss) * 0.1
        room_temperatures.append(room_temperature)

    comfort_area = calculate_area_between_temp(time, room_temperatures, params['thermostat_setting'])
    overshoot = calculate_overshoot_area(time, room_temperatures, params['thermostat_setting'])
    undershoot = calculate_undershoot_area(time, room_temperatures, params['thermostat_setting'])
    return {
        'time': time,
        'room_temperatures': room_temperatures,
        'comfort_area': comfort_area,
        'overshoot': overshoot,
        'undershoot': undershoot,
        'on_off_cycles': heater_on_off_cycles
    }

# Sonuçları İndirme Seçeneği

def convert_results_to_csv(results):
    """Simülasyon sonuçlarını CSV formatına dönüştürür ve sütun başlıklarına birimler ekler."""
    data = []
    for algo, result in results.items():
        for i in range(len(result['time'])):
            data.append({
                'Algoritma': algo,
                'Zaman (dakika)': result['time'][i],
                'Oda Sıcaklığı (°C)': result['room_temperatures'][i]
            })
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# Ana Çalıştırma Fonksiyonu 

def run_simulations(simulation_types, outdoor_temp_values, sim_params, q_params=None, pid_params=None, interpolation_func=None):
    """Seçilen simülasyonları çalıştırır ve sonuçları görselleştirir."""

    results = {}

    if "Açma-Kapama" in simulation_types:
        results["Açma-Kapama"] = run_on_off_simulation(sim_params, outdoor_temp_values, interpolation_func)

    if "Q-Öğrenme" in simulation_types:
        results["Q-Öğrenme"] = run_q_learning_simulation(sim_params, outdoor_temp_values, q_params, interpolation_func)

    if "PID" in simulation_types:
        results["PID"] = run_pid_simulation(sim_params, outdoor_temp_values, pid_params, interpolation_func)

    # Grafikleri Oluştur ve Görselleştir 

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    # Grafik stilini iyileştir
    plt.style.use('ggplot')  # veya başka bir stil seçin
    for algo, data in results.items():
        ax1.plot(data['time'], data['room_temperatures'], label=f"Oda Sıcaklığı ({algo})", linewidth=2)

    ax1.axhline(y=sim_params['thermostat_setting'], color='r', linestyle='--', label="Termostat Ayarı", linewidth=2)
    ax1.set_xlabel("Zaman (Dakika)", fontsize=12)
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=12)  # Birim eklendi
    ax1.legend(fontsize=10)
    ax1.grid(True)
    ax1.set_title("Oda Sıcaklığı Kontrol Simülasyonu", fontsize=14)

    st.pyplot(fig1)

    # Konfor ve Enerji Metrikleri Çubuk Grafik 

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    overshoot_values = [results[algo]['overshoot'] for algo in labels]
    undershoot_values = [results[algo]['undershoot'] for algo in labels]

    width = 0.35
    x = np.arange(len(labels))

    ax2.bar(x - width/2, overshoot_values, width, label='Aşım', color='skyblue')
    ax2.bar(x + width/2, undershoot_values, width, label='Alt Geçiş', color='lightcoral')

    ax2.set_ylabel('Alan (°C*dakika)', fontsize=12)  # Birim ve metrik eklendi
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.set_title("Konfor ve Enerji Tüketimi Metrikleri", fontsize=14)

    # Konfor ve Enerji Metriklerini Göster 

    st.write("### Konfor ve Enerji Metrikleri")
    st.write(f"**Aşım ve Alt Geçiş Değerleri:**")
    for algo in labels:
        st.write(f"{algo} - Aşım: {overshoot_values[labels.index(algo)]:.2f} °C*dakika, Alt Geçiş: {undershoot_values[labels.index(algo)]:.2f} °C*dakika")

    st.pyplot(fig2)

    # Toplam Aşım ve Alt Geçiş Karşılaştırması 

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    total_overshoot_undershoot = {algo: results[algo]['overshoot'] + results[algo]['undershoot'] for algo in labels}

    ax3.bar(total_overshoot_undershoot.keys(), total_overshoot_undershoot.values(), color=['skyblue', 'green', 'lightcoral'])
    ax3.set_title('Toplam Aşım ve Alt Geçiş Karşılaştırması', fontsize=14)
    ax3.set_ylabel('Toplam Alan (°C*dakika)', fontsize=12) 

    # Toplam Aşım ve Alt Geçiş Karşılaştırmasını Göster 

    st.write("### Toplam Aşım ve Alt Geçiş Karşılaştırması")
    st.write(f"**Toplam Alan Değerleri:**")
    for algo, total_value in total_overshoot_undershoot.items():
        st.write(f"{algo} - Toplam Alan: {total_value:.2f} °C*dakika")

    st.pyplot(fig3)

    # Dış Ortam Sıcaklığı Grafiği 

    if outdoor_temp_values is not None or interpolation_func is not None:  # Sadece veri yüklendiğinde veya interpolasyon yapıldığında grafiği göster
        st.write("### Dış Ortam Sıcaklığı Grafiği")
        outdoor_time = np.arange(0, sim_params['simulation_minutes'], 0.1)  # 0.1 dakikalık adımlarla
        outdoor_temps = [get_outdoor_temp(minute, outdoor_temp_values, interpolation_func) for minute in outdoor_time]
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(outdoor_time, outdoor_temps, label="Dış Ortam Sıcaklığı", color='purple')

        # Saat dilimlerini ekle
        hours = np.arange(0, 27, 3)  # 3 saatlik aralıklarla saat dilimleri
        hour_ticks = hours * 60  # Saatleri dakikaya çevir
        ax4.set_xticks(hour_ticks)
        ax4.set_xticklabels([f"{hour:02d}:00" for hour in hours])

        ax4.set_xlabel("Zaman (saat)", fontsize=12)
        ax4.set_ylabel("Dış Ortam Sıcaklığı (°C)", fontsize=12)
        ax4.legend()
        st.pyplot(fig4)
    st.write("### Termostat Açma-Kapama Döngü Sayısı")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    on_off_cycles = [results[algo]['on_off_cycles'] for algo in labels]    
    # Plotting On-Off Cycles
    st.write("#### Algoritmaların Açma-Kapama Döngü Sayıları:")
    for algo, cycles in zip(labels, on_off_cycles):
        st.write(f"- **{algo}**: {cycles} döngü")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    on_off_cycles = [results[algo]['on_off_cycles'] for algo in labels]

    ax6.bar(labels, on_off_cycles, color='blue')
    ax6.set_title('Termostat Açma-Kapama Döngü Sayısı', fontsize=14)
    ax6.set_ylabel('Döngü Sayısı', fontsize=12)
    st.pyplot(fig6)




        

# Main Execution 

if __name__ == "__main__":
    # Sol Sütun: Veri Yükleme ve Parametreler
    with st.sidebar:
        st.header("Veri ve Parametreler")
        outdoor_temp_values, interpolation_func = get_outdoor_temperature_data()
        
        # Simülasyon Parametreleri
        sim_params = get_simulation_parameters()

    # Orta Sütun: Algoritma Seçimi ve Simülasyonu Çalıştırma
    st.header("Simülasyon")

    # Algoritma Seçimi
    simulation_types = st.multiselect("Simülasyon Türü(lerini) Seçin:", ["Açma-Kapama", "Q-Öğrenme", "PID"])

    # Q-Öğrenme ve PID parametrelerini simulation_types tanımlandıktan sonra tanımlayın
    q_params = get_q_learning_parameters() if "Q-Öğrenme" in simulation_types else None
    pid_params = get_pid_parameters() if "PID" in simulation_types else None

    if st.button("Simülasyonları Çalıştır", key="run_simulations_button_1") and (outdoor_temp_values is not None or interpolation_func is not None):
        # Simülasyonların çalıştırılması sırasında bir progress bar göster
        progress_bar = st.progress(0)
        progress_text = st.empty()

        with st.spinner('Simülasyonlar çalıştırılıyor...'):
            results = run_simulations(simulation_types, outdoor_temp_values, sim_params, q_params, pid_params, interpolation_func)

            # Progress bar'ı güncelle
            for i in range(100):
                progress_bar.progress(i + 1)
                progress_text.text(f"İlerleme: %{i + 1}")
                time.sleep(0.05) 

        progress_bar.empty()
        progress_text.empty()

        # Sonuçları İndirme Seçeneği
        if results:
            st.download_button(
                label="Sonuçları İndir (CSV)",
                data=convert_results_to_csv(results),
                file_name='simulation_results.csv',
                mime='text/csv',
            )
    else:  # Dış ortam sıcaklığı verileri veya interpolasyon parametreleri girilmediyse hata mesajı göster
        st.error("Lütfen önce dış ortam sıcaklığı verilerini yükleyin veya interpolasyon parametrelerini girin.")