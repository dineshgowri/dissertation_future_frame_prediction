import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Add FFP code directory to Python path
ffp_code_path = "../Codes_tf2"  # Adjust path to your FFP code location
sys.path.append(ffp_code_path)

from evaluate import compute_auc

psnr_dir = "../Codes_tf2/psnrs/ped2_l_2_alpha_1_lp_1.0_adv_0.05_gdl_1.0_flow_2.0"
# psnr_dir = "../Codes_tf2/psnrs/ped1_l_2_alpha_1_lp_1.0_adv_0.05_gdl_1.0_flow_0.01" // For PED1 Dateset
checkpoints = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]

results_ped = []
auc_ped = []
successful_checkpoints = []

for ckpt in checkpoints:
    ckpt_file = os.path.join(psnr_dir, f"model.ckpt-{ckpt}")
    try:
        res = compute_auc(ckpt_file)
        auc_value = float(res.auc)
        auc_percent = round(auc_value * 100, 3)
        
        auc_ped.append(auc_percent)
        successful_checkpoints.append(ckpt)
        results_ped.append({"Iteration": ckpt, "AUC (%)": auc_percent})
        
        print(f"Checkpoint {ckpt}: {auc_percent}% AUC")
        
    except Exception as e:
        print(f"Error at checkpoint {ckpt}: {e}")

# Display Results Table
df = pd.DataFrame(results_ped)
print("\n" + "="*50)
print("FINAL RESULTS TABLE:")
print("="*50)
print(df.to_string(index=False))

# Create Line Chart (only if we have successful results)
if len(auc_ped) > 0:
    plt.figure(figsize=(10, 6))
    plt.plot(successful_checkpoints, auc_ped, marker='o', linewidth=2, markersize=8, color='blue', label='Our Implementation')
    
    plt.xlabel('Training Iterations')
    plt.ylabel('AUC Score (%)')
    plt.title('PED2 Performance vs Training Iterations')
    
    # Add reference line for original paper
    plt.axhline(y=95.4, color='red', linestyle='--', label='Original Paper (95.4%)')
    
    # Highlight best performance
    if auc_ped:
        best_auc = max(auc_ped)
        best_idx = auc_ped.index(best_auc)
        best_checkpoint = successful_checkpoints[best_idx]
        plt.scatter(best_checkpoint, best_auc, color='green', s=200, label=f'Best Performance ({best_auc}%)', zorder=5)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
else:
    print("\nNo successful results to plot!")