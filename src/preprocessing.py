import pandas as pd
import numpy as np
from datetime import datetime, time
from sklearn.ensemble import IsolationForest
from .helper import _to_minutes

def apply_anomaly_detection(df, contamination=0.01, random_state=42):
    """
    Apply Isolation Forest to detect anomalies in checkin/checkout duration.
    This should be applied BEFORE feature engineering to clean the data.
    """
    print("--- Anomaly Detection (Isolation Forest) ---")
    
    # Calculate duration (hours) for anomaly detection
    # Fill missing checkout with a heuristic (e.g., end of school day) or just use 0/negative for anomaly 
    # Here we focus on data that HAS checkout time for duration a   nomalies
    # And checkin_time anomalies (e.g. 2 AM checkin)
    
    df_iso = df.copy()

    # convert to datetime if not already
    df_iso['checkin_time'] = pd.to_datetime(df_iso['checkin_time'], errors='coerce')
    df_iso['checkout_time'] = pd.to_datetime(df_iso['checkout_time'], errors='coerce')
    df_iso['date'] = pd.to_datetime(df_iso['date'], errors='coerce')
    
    # Feature 1: Duration
    # Handle NaN checkout for duration calc (set to checkin time -> duration 0)
    # This is temporary just for anomaly scoring
    temp_checkout = df_iso['checkout_time'].fillna(df_iso['checkin_time'])
    df_iso['duration_hours'] = (temp_checkout - df_iso['checkin_time']).dt.total_seconds() / 3600
    
    # Feature 2: Arrival Hour (Decimal)
    df_iso['arrival_hour'] = df_iso['checkin_time'].dt.hour + df_iso['checkin_time'].dt.minute/60
    
    # Select features for isolation forest
    # Drop rows where checkin_time is NaT (e.g. Absent) because they are not "sensor anomalies"
    # They are valid "Absent" data.
    mask_valid = df_iso['checkin_time'].notna()
    X_iso = df_iso.loc[mask_valid, ['duration_hours', 'arrival_hour']].fillna(0)
    
    if len(X_iso) > 0:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        preds = iso.fit_predict(X_iso)
        
        # Mark anomalies (-1)
        anom_indices = X_iso[preds == -1].index
        
        print(f"Total data valid (checkin ada): {len(X_iso)}")
        print(f"Terdeteksi {len(anom_indices)} anomali teknis. Contoh:")
        print(df_iso.loc[anom_indices, ['checkin_time', 'checkout_time', 'duration_hours']].head())
        
        # Remove anomalies from original df
        df_clean = df.drop(anom_indices)
        print(f"Data setelah pembersihan anomali: {len(df_clean)}")
        return df_clean
    else:
        print("Tidak ada data valid untuk deteksi anomali.")
        return df

def add_temporal_and_lag_features(df):
    """
    Add temporal and lag features for attendance prediction.
    Uses specific column names from raw dataset:
    - rfid_tag: Student identifier
    - date: Date of attendance
    - checkin_time: Check-in time
    - note: Status note (libur, alpa, telat, etc.)
    """
    # Define labels
    late_labels = {'Telat', 'telat', 'Late', 'late'}
    absent_labels = {'Alpa', 'alpa', 'Absent', 'absent', 'ALPA'}
    holiday_labels = {'Libur', 'libur', 'Holiday'}
    
    df = df.copy()
    
    # Convert datetime columns
    df['date'] = pd.to_datetime(df['date'])
    df['checkin_time'] = pd.to_datetime(df['checkin_time'], errors='coerce')
    if 'checkout_time' in df.columns:
        df['checkout_time'] = pd.to_datetime(df['checkout_time'], errors='coerce')

    # Label 'Hadir' if has checkin_time and note is empty/null
    mask_hadir = df['checkin_time'].notna() & (df['note'].isna() | (df['note'] == ''))
    df.loc[mask_hadir, 'note'] = 'hadir'

    mask_alpa = df['checkin_time'].isna() & (df['note'].isna() | (df['note'] == ''))
    df.loc[mask_alpa, 'note'] = 'alpa'

    df = df.sort_values(['rfid_tag', 'date'])
    
    # 1. Pre-calculate Binary Flags
    df['is_late'] = df['note'].isin(late_labels).astype(int)
    df['is_absent'] = df['note'].isin(absent_labels).astype(int)
    df['_arrival_min'] = df['checkin_time'].map(_to_minutes)
    
    # 2. Group Processing
    groups_list = []
    
    for _, g in df.groupby('rfid_tag', sort=False):
        g = g.set_index('date')
        
        # --- Lag Features ---
        
        # Shift 1: Status Kemarin
        g['Lag_1_Status'] = g['note'].shift(1)
        g['Lag_1_Status'] = g['Lag_1_Status'].fillna('alpa')
        g['Lag_1_Status'] = g['Lag_1_Status'].replace('libur', 'alpa')
        
        # Rolling Count (7 hari kalender ke belakang)
        g['Count_Telat_7D'] = g['is_late'].rolling('7d', closed='left').sum().fillna(0).astype(int)
        g['Count_Alpa_30D'] = g['is_absent'].rolling('30d', closed='left').sum().fillna(0).astype(int)
        
        # Avg Arrival Time
        g['Avg_Arrival_Time_7D'] = g['_arrival_min'].rolling('7d', closed='left').mean().fillna(0)
        
        # Streak Calculation
        streak_mask = g['is_late'] == 1
        streak_group = (streak_mask == False).cumsum()
        streak_current = g.groupby(streak_group).cumcount() 
        streak_current = streak_current.where(streak_mask, 0)
        g['Streak_Telat'] = streak_current.shift(1).fillna(0).astype(int)
        
        groups_list.append(g.reset_index())
        
    # Gabung kembali
    result = pd.concat(groups_list, axis=0, ignore_index=True)
    
    # Tambah fitur Hari
    result['DayOfWeek'] = result['date'].dt.dayofweek

    # --- FILTERING AKHIR ---
    # Hapus baris yang labelnya Libur AGAR TIDAK JADI TARGET PREDIKSI
    # Tapi baris ini SUDAH DIPAKAI untuk menghitung Lag fitur baris sesudahnya
    result_final = result[~result['note'].isin(holiday_labels)].copy()
    
    # Cleanup auxiliary columns
    result_final.drop(columns=['_arrival_min', 'is_late', 'is_absent'], inplace=True)

    # sort by id
    result_final = result_final.sort_values(by='id', ascending=False).reset_index(drop=True)
    
    return result_final
