import matplotlib.pyplot as plt
import pandas as pd

# 1. Plug in the  data from runs
data = {
    "Variant": ["Otsu (auto)", "Max Fraction 0.5", "Max Fraction 0.7"],
    "field_0001 (Sparse/Spots)": [16, 16, 17],
    "field_0002 (High-Density Spots)": [83, 83, 85],
    "field_0003 (Mixed Regime)": [13, 13, 13],
    "field_0004 (Labyrinth)": [2, 2, 2],
}

# 2. Convert to a DataFrame
df = pd.DataFrame(data).set_index("Variant")

# 3. Create the clustered bar plot to visually check for the plateau
ax = df.plot(kind="bar", width=0.8, figsize=(10, 6))

# 4. Label plot
plt.title("Component Count Sensitivity Across Dynamic Threshold Variants", fontsize=14, pad=15)
plt.ylabel("Connected Component Count", fontsize=12)
plt.xlabel("Dynamic Threshold Choice", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(title="Simulation Fields", loc="upper right")
plt.tight_layout()

# 5. Save the image 
plt.savefig("threshold_plateau_plot.png", dpi=300)
print("Diagram successfully generated and saved as 'threshold_plateau_plot.png'!")