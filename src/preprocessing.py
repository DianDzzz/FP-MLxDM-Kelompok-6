import pandas as pd
import numpy as np
from datetime import datetime, time
from sklearn.ensemble import IsolationForest
from .helper import _to_minutes

def apply_anomaly_detection(df, contamination=0.05, random_state=42):
    """
    Apply Isolation Forest to detect anomalies AFTER feature engineering.
    Uses engineered features for better anomaly detection.
    
    Should be called AFTER add_temporal_and_lag_features() to preserve
    timeseries integrity during feature calculation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with engineered features (output of add_temporal_and_lag_features)
    contamination : float, default=0.05
        Expected proportion of anomalies
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with anomalies removed
    """
    print("--- Anomaly Detection (Isolation Forest) ---")
    
    df_iso = df.copy()
    
    # Ensure datetime columns
    df_iso['checkin_time'] = pd.to_datetime(df_iso['checkin_time'], errors='coerce')
    df_iso['checkout_time'] = pd.to_datetime(df_iso['checkout_time'], errors='coerce')
    
    # Feature 1: Duration (hours)
    temp_checkout = df_iso['checkout_time'].fillna(df_iso['checkin_time'])
    df_iso['_duration_hours'] = (temp_checkout - df_iso['checkin_time']).dt.total_seconds() / 3600
    
    # Feature 2: Arrival Hour (decimal)
    df_iso['_arrival_hour'] = df_iso['checkin_time'].dt.hour + df_iso['checkin_time'].dt.minute / 60
    
    # Features for anomaly detection - use engineered features too
    anomaly_features = ['_duration_hours', '_arrival_hour']
    
    # Add engineered features if available
    if 'Count_Telat_7D' in df_iso.columns:
        anomaly_features.append('Count_Telat_7D')
    if 'Count_Alpa_30D' in df_iso.columns:
        anomaly_features.append('Count_Alpa_30D')
    if 'Streak_Telat' in df_iso.columns:
        anomaly_features.append('Streak_Telat')
    if 'Avg_Arrival_Time_7D' in df_iso.columns:
        anomaly_features.append('Avg_Arrival_Time_7D')
    
    # Only process rows with valid checkin (not absent/holiday)
    mask_valid = df_iso['checkin_time'].notna()
    X_iso = df_iso.loc[mask_valid, anomaly_features].fillna(0)
    
    if len(X_iso) > 0:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        preds = iso.fit_predict(X_iso)
        
        # Get anomaly indices
        anom_indices = X_iso[preds == -1].index
        
        print(f"Features used: {anomaly_features}")
        print(f"Total data dengan checkin: {len(X_iso)}")
        print(f"Anomali terdeteksi: {len(anom_indices)} ({len(anom_indices)/len(X_iso)*100:.2f}%)")
        
        if len(anom_indices) > 0:
            print(f"\nContoh anomali:")
            display_cols = ['rfid_tag', 'date', 'checkin_time', 'note', '_duration_hours', '_arrival_hour']
            display_cols = [c for c in display_cols if c in df_iso.columns]
            print(df_iso.loc[anom_indices, display_cols].head(10).to_string())
        
        # Remove anomalies
        df_clean = df.drop(anom_indices).copy()
        
        # Drop temporary columns
        temp_cols = ['_duration_hours', '_arrival_hour']
        df_clean = df_clean.drop(columns=[c for c in temp_cols if c in df_clean.columns], errors='ignore')
        
        print(f"\nData sebelum: {len(df)} baris")
        print(f"Data setelah: {len(df_clean)} baris")
        
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


def remove_low_attendance_students(df, threshold_pct=10.0):
    """
    Remove students (rfid_tag) with attendance percentage below threshold.
    This preserves timeseries integrity by removing entire student records.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: rfid_tag, checkin_time, note
    threshold_pct : float, default=10.0
        Minimum attendance percentage to keep student. 
        Students with attendance < threshold will be removed.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with low-attendance students removed
    """
    print(f"--- Removing Low Attendance Students (< {threshold_pct}%) ---")
    
    df_temp = df.copy()
    
    # Calculate attendance per student
    df_temp['is_hadir'] = df_temp['checkin_time'].notna()
    
    student_stats = df_temp.groupby('rfid_tag').agg(
        total_records=('rfid_tag', 'count'),
        total_hadir=('is_hadir', 'sum')
    ).reset_index()
    
    student_stats['pct_hadir'] = (student_stats['total_hadir'] / student_stats['total_records'] * 100).round(2)
    
    # Identify outlier students (attendance < threshold)
    outlier_students = student_stats[student_stats['pct_hadir'] < threshold_pct]['rfid_tag'].tolist()
    normal_students = student_stats[student_stats['pct_hadir'] >= threshold_pct]['rfid_tag'].tolist()
    
    print(f"Total siswa: {len(student_stats)}")
    print(f"Siswa dengan kehadiran < {threshold_pct}%: {len(outlier_students)}")
    print(f"Siswa yang dipertahankan: {len(normal_students)}")
    
    if len(outlier_students) > 0:
        print(f"\nContoh siswa outlier yang dihapus:")
        outlier_details = student_stats[student_stats['rfid_tag'].isin(outlier_students)].head(10)
        print(outlier_details.to_string(index=False))
    
    # Remove outlier students
    df_clean = df[df['rfid_tag'].isin(normal_students)].copy()
    
    print(f"\nData sebelum: {len(df)} baris")
    print(f"Data setelah: {len(df_clean)} baris")
    print(f"Baris dihapus: {len(df) - len(df_clean)}")
    
    return df_clean
