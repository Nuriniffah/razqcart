import numpy as np
import pandas as pd

np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=200)
values = np.random.normal(100, 10, size=200)

# Inject anomalies
values[50] = 200   # spike
values[150] = 20   # drop

df = pd.DataFrame({"Date": dates, "Value": values})
df.to_csv("synthetic_anomaly_data.csv", index=False)


# import pandas as pd
# import numpy as np

# # Example time series: e.g., daily sales
# np.random.seed(42)
# dates = pd.date_range("2024-01-01", periods=100)
# values = np.random.normal(100, 10, size=100)
# values[30] = 200   # simulate a spike
# values[70] = 40    # simulate a drop

# df = pd.DataFrame({"Date": dates, "Value": values})

# # Rolling mean and std
# window = 10
# df["RollingMean"] = df["Value"].rolling(window).mean()
# df["RollingStd"] = df["Value"].rolling(window).std()

# # Compute z-score
# df["Zscore"] = (df["Value"] - df["RollingMean"]) / df["RollingStd"]

# # Flag anomalies: abs(z) > 3
# df["Anomaly"] = df["Zscore"].abs() > 3

# # Print anomalies
# print(df[df["Anomaly"]])

# # Optional: alert logic (pseudo-code)
# for _, row in df[df["Anomaly"]].iterrows():
#     message = f"⚠️ Anomaly detected on {row['Date'].date()}: value={row['Value']:.2f}"
#     print(message)  # Replace with email/slack/etc.
