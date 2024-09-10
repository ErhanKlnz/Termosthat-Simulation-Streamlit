import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import time
from sklearn.tree import DecisionTreeRegressor  # Karar Ağaçları Modeli için gerekli kütüphane

# Uygulama Başlığı ve Açıklaması
st.set_page_config(page_title="Termostat Simülasyonu", page_icon="🌡️", layout="wide")
st.title("Termostat Simülasyonu")
st.subheader("Kontrol Algoritmalarının Karşılaştırılması")
st.write("Bu interaktif simülasyon, oda sıcaklığını korumak için farklı kontrol algoritmalarının performansını karşılaştırır.")

# Dosya Yükleyici ve Hata Kontrolü

def load_data():
    uploaded_file = st.file_uploader("Bir CSV dosyası seçin (Dış Ortam Sıcaklığı ve sutün adı 'Outdoor Temp (C)' olan verilerini içeren)", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Outdoor Temp (C)' not in df.columns:
                st.error("CSV dosyası 'Outdoor Temp (C)' sütununu içermiyor. Lütfen doğru dosyayı yükleyin.")
                return None
            if df.isnull().values.any():
                st.error("CSV dosyasında eksik değerler var. Lütfen düzeltin ve tekrar yükleyin.")
                return None

            # Tarih ve Saat Sütunlarının Kontrolü
            if 'Date' in df.columns and 'Time' in df.columns:
                # Günlük Minimum, Maksimum ve Ortalama Sıcaklıkları Hesaplama
                df['Date'] = pd.to_datetime(df['Date'])
                daily_stats = df.groupby(df['Date'].dt.date).agg(
                    daily_avg=('Outdoor Temp (C)', 'mean'),
                    daily_min=('Outdoor Temp (C)', 'min'),
                    daily_max=('Outdoor Temp (C)', 'max')
                )

                # Grafik Oluşturma: Ortalama, Min ve Max Sıcaklıkları Gösterme
                fig, ax = plt.subplots(figsize=(10, 6))

                # Ortalama sıcaklık
                ax.plot(daily_stats.index, daily_stats['daily_avg'], marker='o', linestyle='-', color='mediumseagreen', label='Ortalama Sıcaklık')

                # Min-Max aralığını çiz
                ax.fill_between(daily_stats.index, daily_stats['daily_min'], daily_stats['daily_max'], color='gray', alpha=0.3, label='Min-Max Aralığı')

                # Grafik etiketleri
                ax.set_title('Günlük Ortalama ve Min-Max Dış Sıcaklık (°C)', fontsize=16)
                ax.set_xlabel('Tarih', fontsize=14)
                ax.set_ylabel('Sıcaklık (°C)', fontsize=14)
                ax.legend()
                ax.grid(True)

                st.pyplot(fig)

                # Tarih ve saat sütunlarını birleştirip datetime formatına çeviriyoruz
                df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
                st.success("Veri başarıyla yüklendi! Şimdi bir başlangıç tarihi ve saati seçin.")

                # Başlangıç Tarih ve Saat Seçimi
                start_date = st.date_input("Başlangıç Tarihini Seçin", min_value=df['DateTime'].min().date(), max_value=df['DateTime'].max().date())
                start_time = st.time_input("Başlangıç Saatini Seçin", value=pd.Timestamp('00:00').time())

                # Seçilen Tarih ve Saatten Sonraki Verileri Filtreleme
                start_datetime = pd.to_datetime(f"{start_date} {start_time}")
                df_filtered = df[df['DateTime'] >= start_datetime]

                # Filtrelenmiş Veriyi Döndürme
                if df_filtered.empty:
                    st.error("Seçilen tarih ve saatten sonraki veri seti boş. Lütfen farklı bir tarih/saat seçin.")
                    return None
                else:
                    outdoor_temp_values = df_filtered['Outdoor Temp (C)'].values
                    return outdoor_temp_values
            else:
                st.warning("Veri setinde 'Date' ve 'Time' sütunları bulunamadı. Tüm veriler kullanılacak.")
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
    st.sidebar.header("Simülasyon Parametreleri")
    initial_room_temperature = st.sidebar.number_input("Başlangıç Oda Sıcaklığı (°C)", min_value=10, max_value=30, value=19)
    thermostat_setting = st.sidebar.number_input("Termostat Ayarı (°C)", min_value=15, max_value=25, value=20)
    heater_power = st.sidebar.slider("Isıtıcı Gücü (°C/dakika)", min_value=0.1, max_value=0.5, value=0.3)
    base_heat_loss = st.sidebar.slider("Temel Isı Kaybı (°C/dakika)", min_value=0.05, max_value=0.2, value=0.1)
    simulation_minutes = st.sidebar.number_input("Simülasyon Süresi (Dakika)", min_value=10, max_value=43200, value=60)
    thermostat_sensitivity = st.sidebar.slider("Termostat Hassasiyeti (°C)", min_value=0.1, max_value=0.5, value=0.5, step=0.1)
    min_run_time = st.sidebar.number_input("Minimum Çalışma Süresi (Dakika)", min_value=0.2, max_value=1000.0, value=1.0, step=0.1)
    min_off_time = st.sidebar.number_input("Minimum Kapalı Kalma Süresi (Dakika)", min_value=0.2, max_value=1000.0, value=1.0, step=0.1)
    return {
        'initial_room_temperature': initial_room_temperature,
        'thermostat_setting': thermostat_setting,
        'heater_power': heater_power,
        'base_heat_loss': base_heat_loss,
        'simulation_minutes': simulation_minutes,
        'thermostat_sensitivity': thermostat_sensitivity,
        'min_run_time': min_run_time,
        'min_off_time': min_off_time
    }

# Q-Öğrenme Parametreleri 
def get_q_learning_parameters():
    st.sidebar.subheader("Q-Öğrenme Parametreleri")
    episodes = st.sidebar.number_input("Eğitim Bölümleri", min_value=10, max_value=5000, value=1000)
    learning_rate = st.sidebar.slider("Öğrenme Oranı", min_value=0.01, max_value=1.0, value=0.5)
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
    st.sidebar.subheader("PID Parametreleri")
    Kp = st.sidebar.slider("Kp (Oransal Kazanç)", min_value=0.1, max_value=20.0, value=19.0)
    Ki = st.sidebar.slider("Ki (İntegral Kazanç)", min_value=0.01, max_value=1.0, value=0.15)
    Kd = st.sidebar.slider("Kd (Türev Kazanç)", min_value=0.001, max_value=1.0, value=0.01)
    return {
        'Kp': Kp,
        'Ki': Ki,
        'Kd': Kd
    }

# Karar Ağaçları Parametreleri 
def get_decision_tree_parameters():
    st.sidebar.subheader("Karar Ağaçları Parametreleri")
    max_depth = st.sidebar.slider("Maksimum Derinlik", min_value=1, max_value=25, value=5)
    min_samples_split = st.sidebar.slider("Minimum Yaprak Bölünme Sayısı", min_value=2, max_value=25, value=2)
    return {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split
    }

# Yardımcı Fonksiyonlar 
def get_state(temperature):
    return int(min(40, max(0, (temperature - 10) / 0.5)))

# Dış Ortam Sıcaklığı Seçimi ve İnterpolasyon
def get_outdoor_temperature_data():
    data_source = st.sidebar.radio("Dış Ortam Sıcaklığı Veri Kaynağı", ["CSV Dosyası", "İnterpolasyon (Kübik Spline)"])
    if data_source == "CSV Dosyası":
        outdoor_temp_values = load_data()
        return outdoor_temp_values, None
    elif data_source == "İnterpolasyon (Kübik Spline)":
        st.sidebar.subheader("İnterpolasyon Verileri")
        if 'temperatures' not in st.session_state:
            st.session_state.temperatures = [20, 20, 20, 20, 20]
        with st.form("interpolation_form"):
            hours = [0, 6, 12, 18, 24]
            for i, hour in enumerate(hours):
                st.session_state.temperatures[i] = st.number_input(f"{hour}:00 Sıcaklığı (°C)", value=st.session_state.temperatures[i], min_value=-20, max_value=50)
            submitted = st.form_submit_button("İnterpolasyonu Uygula")
            if submitted:
                interpolation_func = CubicSpline(hours, st.session_state.temperatures)
                return None, interpolation_func 
        return None, CubicSpline(hours, st.session_state.temperatures)

def get_outdoor_temp(minute, outdoor_temp_values, interpolation_func):
    if outdoor_temp_values is not None:
        index = int(minute // 5) 
        return outdoor_temp_values[min(index, len(outdoor_temp_values) - 1)]
    elif interpolation_func is not None:
        hour = minute / 60
        return float(interpolation_func(hour))

# Alan Hesaplama Fonksiyonları
def calculate_area_between_temp(time, room_temperatures, set_temp):
    area = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        area += abs(avg_temp - set_temp) * dt
    return area

def calculate_overshoot_area(time, room_temperatures, set_temp):
    overshoot = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        if avg_temp > set_temp:
            overshoot += (avg_temp - set_temp) * dt
    return overshoot

def calculate_undershoot_area(time, room_temperatures, set_temp):
    undershoot = 0
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        avg_temp = (room_temperatures[i] + room_temperatures[i - 1]) / 2
        if avg_temp < set_temp:
            undershoot += (set_temp - avg_temp) * dt
    return undershoot

def run_on_off_simulation(params, outdoor_temp_values, interpolation_func):
    time = []
    room_temperatures = []
    heater_status = False
    heater_on_duration = 0
    heater_off_duration = params['min_off_time']
    heater_on_off_cycles = 0
    room_temperature = params['initial_room_temperature']

    for minute in np.arange(0, params['simulation_minutes'], 0.1):
        time.append(minute)
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)

        if heater_status:
            heater_on_duration += 0.1
        else:
            heater_off_duration += 0.1

        if (room_temperature < params['thermostat_setting'] - params['thermostat_sensitivity'] and 
            not heater_status and heater_off_duration >= params['min_off_time']):
            heater_status = True
            heater_on_duration = 0
            heater_off_duration = 0
            heater_on_off_cycles += 1

        elif (room_temperature > params['thermostat_setting'] + params['thermostat_sensitivity'] and 
              heater_status and heater_on_duration >= params['min_run_time']):
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
        'on_off_cycles': heater_on_off_cycles
    }

    
def run_pid_simulation(params, outdoor_temp_values, pid_params, interpolation_func):
    time = []
    room_temperatures = []
    heater_output = []
    heater_on_off_cycles = 0

    integral_error = 0
    previous_error = 0
    room_temperature = params['initial_room_temperature']
    heater_status = False
    heater_on_duration = 0
    heater_off_duration = params['min_off_time']

    for minute in np.arange(0, params['simulation_minutes'], 0.1):
        time.append(minute)
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)

        # PID Control calculations
        error = params['thermostat_setting'] - room_temperature
        proportional_term = pid_params['Kp'] * error
        integral_error += error * 0.1
        integral_term = pid_params['Ki'] * integral_error
        derivative_term = pid_params['Kd'] * (error - previous_error) / 0.1
        previous_error = error

        pid_output = proportional_term + integral_term + derivative_term
        pid_output = max(0, min(pid_output, 1))
        heater_output.append(pid_output)

        # Update heater status based on PID output and min_run_time / min_off_time constraints
        if heater_status:
            heater_on_duration += 0.1
            heater_off_duration = 0  # Reset off duration when the heater is on
        else:
            heater_off_duration += 0.1
            heater_on_duration = 0  # Reset on duration when the heater is off

        if (pid_output > 0.3 and not heater_status and heater_off_duration >= params['min_off_time']):
            heater_status = True
            heater_on_duration = 0
            heater_on_off_cycles += 1

        if (pid_output < 0.3 and heater_status and heater_on_duration >= params['min_run_time']):
            heater_status = False
            heater_off_duration = 0

        # Calculate heat loss
        heat_loss = params['base_heat_loss'] * (room_temperature - outside_temperature) / 10

        # Update room temperature based on heater status and heat loss
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
        'on_off_cycles': heater_on_off_cycles
    }



# Eylem Seçim Fonksiyonu
def get_action(state, q_table, exploration_rate, num_actions):
    if np.random.rand() < exploration_rate:
        return np.random.choice(num_actions)  # Keşif yap (exploration)
    else:
        return np.argmax(q_table[state])  # En iyi eylemi seç (exploitation)

def get_reward(state, action, thermostat_setting):
    state_temp = 10 + state * 0.5
    if abs(state_temp - thermostat_setting) <= 0.3:
        return 150  # Daha yüksek ödül
    elif action == 1 and state_temp > thermostat_setting:
        return -200  # Daha yüksek ceza
    elif action == 0 and state_temp < thermostat_setting:
        return -150  # Daha yüksek ceza
    else:
        return -50  # Diğer durumlarda daha düşük ceza
    
 
num_states = 41
num_actions = 2
q_table = np.zeros((num_states, num_actions))
# Simülasyon Mantığı (Q-Öğrenme) 

def run_q_learning_simulation(params, outdoor_temp_values, q_params, interpolation_func):
    # q_table'ı uygun şekilde başlatmalısınız
    total_on_off_cycles = 0
    for episode in range(q_params['episodes']):
        room_temperature = params['initial_room_temperature']
        state = get_state(room_temperature)
        heater_status = False
        heater_on_duration = 0
        heater_off_duration = params['min_off_time']

        for minute in np.arange(0, params['simulation_minutes'], 0.1):
            outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)
            exploration_rate = q_params['exploration_rate'] * (1 - episode / q_params['episodes'])
            action = get_action(state, q_table, exploration_rate,num_actions)

            if heater_status:
                heater_on_duration += 0.1
            else:
                heater_off_duration += 0.1

            if action == 1 and not heater_status and heater_off_duration >= params['min_off_time']:
                heater_status = True
                heater_on_duration = 0
                heater_off_duration = 0
                total_on_off_cycles += 1

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
    heater_off_duration = params['min_off_time']
    total_on_off_cycles = 0

    for minute in np.arange(0, params['simulation_minutes'], 0.1):
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)
        action = np.argmax(q_table[state, :])

        if heater_status:
            heater_on_duration += 0.1
        else:
            heater_off_duration += 0.1

        if action == 1 and not heater_status and heater_off_duration >= params['min_off_time']:
            heater_status = True
            heater_on_duration = 0
            heater_off_duration = 0
            total_on_off_cycles += 1

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
        'on_off_cycles': total_on_off_cycles
    }

def run_decision_tree_simulation(params, outdoor_temp_values, decision_tree_params, interpolation_func):
    time = []
    room_temperatures = []
    heater_on_off_cycles = 0
    room_temperature = params['initial_room_temperature']
    heater_status = False
    heater_on_duration = 0
    heater_off_duration = params['min_off_time']
    model = DecisionTreeRegressor(max_depth=decision_tree_params['max_depth'], min_samples_split=decision_tree_params['min_samples_split'])

    training_data = []
    training_labels = []

    for minute in np.arange(0, params['simulation_minutes'] * 10, 0.1):
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)
        heat_loss = params['base_heat_loss'] * (room_temperature - outside_temperature) / 10

        if room_temperature < params['thermostat_setting'] - params['thermostat_sensitivity']:
            heater_status = True
        else:
            heater_status = False

        if heater_status:
            room_temperature += params['heater_power'] * 0.1
        else:
            room_temperature -= heat_loss * 0.1

        training_data.append([room_temperature, outside_temperature])
        training_labels.append(heater_status)

    if len(training_data) > 50:
        model.fit(training_data, training_labels)

    room_temperature = params['initial_room_temperature']

    for minute in np.arange(0, params['simulation_minutes'], 0.1):
        time.append(minute)
        outside_temperature = get_outdoor_temp(minute, outdoor_temp_values, interpolation_func)

        if len(training_data) > 50:
            prediction = model.predict([[room_temperature, outside_temperature]])
            predicted_heater_status = prediction[0] > 0.5

        if heater_status:
            heater_on_duration += 0.1
            heater_off_duration = 0
        else:
            heater_off_duration += 0.1
            heater_on_duration = 0

        if (predicted_heater_status and not heater_status and heater_off_duration >= params['min_off_time'] and 
            room_temperature < params['thermostat_setting'] - params['thermostat_sensitivity']):
            heater_status = True
            heater_on_duration = 0
            heater_on_off_cycles += 1

        if (not predicted_heater_status and heater_status and heater_on_duration >= params['min_run_time'] and 
            room_temperature > params['thermostat_setting'] + params['thermostat_sensitivity']):
            heater_status = False
            heater_off_duration = 0

        if room_temperature >= params['thermostat_setting']:
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
        'on_off_cycles': heater_on_off_cycles
    }


# Sonuçları İndirme Seçeneği
def convert_results_to_csv(results):
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
def run_simulations(simulation_types, outdoor_temp_values, sim_params, q_params=None, pid_params=None, decision_tree_params=None, interpolation_func=None):
    results = {}
    if "Açma-Kapama" in simulation_types:
        results["Açma-Kapama"] = run_on_off_simulation(sim_params, outdoor_temp_values, interpolation_func)
    if "Q-Öğrenme" in simulation_types:
        results["Q-Öğrenme"] = run_q_learning_simulation(sim_params, outdoor_temp_values, q_params, interpolation_func)
    if "PID" in simulation_types:
        results["PID"] = run_pid_simulation(sim_params, outdoor_temp_values, pid_params, interpolation_func)
    if "Karar Ağaçları" in simulation_types:
        results["Karar Ağaçları"] = run_decision_tree_simulation(sim_params, outdoor_temp_values, decision_tree_params, interpolation_func)

    csv = convert_results_to_csv(results)
    st.download_button("Sonuçları CSV Olarak İndir", csv, "simulasyon_sonuclari.csv", "text/csv")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plt.style.use('ggplot')
    for algo, data in results.items():
        ax1.plot(data['time'], data['room_temperatures'], label=f"Oda Sıcaklığı ({algo})", linewidth=2)
    ax1.axhline(y=sim_params['thermostat_setting'], color='r', linestyle='--', label="Termostat Ayarı", linewidth=2)
    ax1.set_xlabel("Zaman (Dakika)", fontsize=12)
    ax1.set_ylabel("Sıcaklık (°C)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True)
    st.write(f"**Oda Sıcaklığı Kontrol Simülasyonu:**")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(5, 2))
    labels = list(results.keys())
    overshoot_values = [result['overshoot'] for result in results.values()]
    undershoot_values = [result['undershoot'] for result in results.values()]
    x = np.arange(len(labels))
    width = 0.25
    bars1 = ax2.bar(x - width, overshoot_values, width, label='Aşım')
    bars2 = ax2.bar(x, undershoot_values, width, label='Alt Geçiş')
    ax2.set_xlabel('Algoritmalar')
    ax2.set_ylabel('Metrikler')
    ax2.set_title('Konfor ve Enerji Metrikleri')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    st.write("### Konfor ve Enerji Metrikleri")
    st.write(f"**Aşım ve Alt Geçiş Değerleri:**")
    for algo in labels:
        st.write(f"{algo} - Aşım: {overshoot_values[labels.index(algo)]:.2f} °C*dakika, Alt Geçiş: {undershoot_values[labels.index(algo)]:.2f} °C*dakika")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    total_overshoot_undershoot = {algo: results[algo]['overshoot'] + results[algo]['undershoot'] for algo in labels}
    ax3.bar(total_overshoot_undershoot.keys(), total_overshoot_undershoot.values(), color=['steelblue', 'lightsteelblue', 'mediumseagreen', 'palegreen'])
    ax3.set_title('Toplam Aşım ve Alt Geçiş Karşılaştırması', fontsize=14)
    ax3.set_ylabel('Toplam Alan (°C*dakika)', fontsize=12) 
    st.write("### Toplam Aşım ve Alt Geçiş Karşılaştırması")
    st.write(f"**Toplam Alan Değerleri:**")
    for algo, total_value in total_overshoot_undershoot.items():
        st.write(f"{algo} - Toplam Alan: {total_value:.2f} °C*dakika")
    st.pyplot(fig3)
   
    if outdoor_temp_values is not None or interpolation_func is not None:  # Sadece veri yüklendiğinde veya interpolasyon yapıldığında grafiği göster
        st.write("### Dış Ortam Sıcaklığı Grafiği")
        outdoor_time = np.arange(0, sim_params['simulation_minutes'], 0.1)  # 0.1 dakikalık adımlarla
        outdoor_temps = [get_outdoor_temp(minute, outdoor_temp_values, interpolation_func) for minute in outdoor_time]
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        ax4.plot(outdoor_time, outdoor_temps, label="Dış Ortam Sıcaklığı", color='purple')
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
    st.write("#### Algoritmaların Açma-Kapama Döngü Sayıları:")
    for algo, cycles in zip(labels, on_off_cycles):
        st.write(f"- **{algo}**: {cycles} döngü")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    labels = list(results.keys())
    on_off_cycles = [results[algo]['on_off_cycles'] for algo in labels]
    ax6.bar(labels, on_off_cycles, color=['steelblue', 'lightsteelblue', 'mediumseagreen', 'palegreen'])
    ax6.set_title('Termostat Açma-Kapama Döngü Sayısı', fontsize=14)
    ax6.set_ylabel('Döngü Sayısı', fontsize=12)
    st.pyplot(fig6)
    
# Main Execution 
if __name__ == "__main__":
    with st.sidebar:
        st.header("Veri ve Parametreler")
        outdoor_temp_values, interpolation_func = get_outdoor_temperature_data()
        sim_params = get_simulation_parameters()

    st.header("Simülasyon")
    simulation_types = st.multiselect("Simülasyon Türü(lerini) Seçin:", ["Açma-Kapama", "Q-Öğrenme", "PID", "Karar Ağaçları"])
    q_params = get_q_learning_parameters() if "Q-Öğrenme" in simulation_types else None
    pid_params = get_pid_parameters() if "PID" in simulation_types else None
    decision_tree_params = get_decision_tree_parameters() if "Karar Ağaçları" in simulation_types else None

    if st.button("Simülasyonları Çalıştır", key="run_simulations_button_1") and (outdoor_temp_values is not None or interpolation_func is not None):
        progress_bar = st.progress(0)
        progress_text = st.empty()
        with st.spinner('Simülasyonlar çalıştırılıyor...'):
            results = run_simulations(simulation_types, outdoor_temp_values, sim_params, q_params, pid_params, decision_tree_params, interpolation_func)
            for i in range(100):
                progress_bar.progress(i + 1)
                progress_text.text(f"İlerleme: %{i + 1}")
                time.sleep(0.05) 
        progress_bar.empty()
        progress_text.empty()
        if results:
            st.download_button(
                label="Sonuçları İndir (CSV)",
                data=convert_results_to_csv(results),
                file_name='simulation_results.csv',
                mime='text/csv',
            )
    else:  # Dış ortam sıcaklığı verileri veya interpolasyon parametreleri girilmediyse hata mesajı göster
        st.error("Lütfen önce dış ortam sıcaklığı verilerini yükleyin veya interpolasyon parametrelerini girin.")
