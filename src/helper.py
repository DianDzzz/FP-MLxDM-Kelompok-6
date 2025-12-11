import pandas as pd
import numpy as np
from datetime import datetime, time

def _to_minutes(dt):
    """Konversi waktu ke menit sejak 00:00."""
    if pd.isna(dt):
        return np.nan
    if isinstance(dt, (pd.Timestamp, datetime)):
        return dt.hour * 60 + dt.minute
    if isinstance(dt, time):
        return dt.hour * 60 + dt.minute
    if isinstance(dt, str):
        try:
            parsed = pd.to_datetime(dt)
            return parsed.hour * 60 + parsed.minute
        except:
            return np.nan
    return np.nan
