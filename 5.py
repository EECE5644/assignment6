# Anomaly Alerting
residuals = total_actuals.flatten() - total_preds.flatten()

sigma = residuals.std()
threshold = 3 * sigma

anomalies = np.abs(residuals) > threshold
print(anomalies)

# Check if np.abs(residuals) > threshold continue for more than 2 consecutive hours and gather only the true case
alert_indices = []
count = 0
for i, flag in enumerate(anomalies):
    if flag:
        count += 1
        if count >= 2:
            alert_indices.append(i)
    else:
        count = 0

print("Residuals:", residuals)
print("Threshold:", threshold)
print("Anomalies:", anomalies)
print("Alert indices (2+ consecutive anomalies):", alert_indices)
