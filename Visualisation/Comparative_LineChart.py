import matplotlib.pyplot as plt

checkpoints = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
ped1_auc = [81.27, 82.01, 82.24, 82.32, 82.83, 82.82, 82.80, 82.73]
ped2_auc = [95.07, 95.34, 95.29, 95.41, 95.23, 95.16, 95.12, 95.08]

plt.figure(figsize=(10, 6))

# Plot PED1
plt.plot(checkpoints, ped1_auc, marker='o', linewidth=2, markersize=8, label="PED1")
plt.axhline(y=83.1, color='blue', linestyle='--', label='PED1 Baseline (83.1%)')

# Plot PED2
plt.plot(checkpoints, ped2_auc, marker='s', linewidth=2, markersize=8, label="PED2")
plt.axhline(y=95.4, color='red', linestyle='--', label='PED2 Baseline (95.4%)')

# Highlight best performances
best_ped1 = max(ped1_auc)
best_ped2 = max(ped2_auc)
plt.scatter(checkpoints[ped1_auc.index(best_ped1)], best_ped1, color='blue', s=200, zorder=5)
plt.scatter(checkpoints[ped2_auc.index(best_ped2)], best_ped2, color='red', s=200, zorder=5)

plt.xlabel('Training Iterations')
plt.ylabel('AUC Score (%)')
plt.title('PED1 vs PED2 Performance Across Iterations for FFP')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
