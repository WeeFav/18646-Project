import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data from your C++ output
try:
    df = pd.read_csv('results.csv')
except FileNotFoundError:
    print("Error: results.csv not found. Please run your C++ program first.")
    exit()

# 2. Calculate the average cycles for each phase per resolution
averages = df.groupby('Resolution')[['All_Transform_Cycles', 'Raster_Loop_Cycles']].mean().reset_index()

# 3. Create mapping for triangle counts for the X-axis
tri_counts = {
    16: "33,462",
    32: "133,020",
    64: "530,384",
    128: "2,117,776"
}
x_labels = [f"{res}\n({tri_counts.get(res, '0')} triangles)" for res in averages['Resolution']]

# ==========================================
# PLOT 1: VERTEX TRANSFORM
# ==========================================
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x_labels, averages['All_Transform_Cycles'], color='#f39c12', edgecolor='black')

plt.title('Average Execution Time: Vertex Transform', fontsize=15, fontweight='bold')
plt.xlabel('Resolution Scale and Triangle Count', fontsize=12)
plt.ylabel('Average CPU Cycles', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2e}', 
             va='bottom', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('vertex_transform_result.png')
print("Saved Vertex Transform plot to vertex_transform_result.png")
plt.close() # Close the figure so it doesn't overlap with the next one


# ==========================================
# PLOT 2: RASTERIZATION
# ==========================================
plt.figure(figsize=(10, 6))
bars2 = plt.bar(x_labels, averages['Raster_Loop_Cycles'], color='#3498db', edgecolor='black')

plt.title('Average Execution Time: Rasterization Loop', fontsize=15, fontweight='bold')
plt.xlabel('Resolution Scale and Triangle Count', fontsize=12)
plt.ylabel('Average CPU Cycles', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2e}', 
             va='bottom', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('rasterization_result.png')
print("Saved Rasterization plot to rasterization_result.png")
plt.close()

# If you want to display them interactively as well, you can remove the plt.close() 
# commands and put plt.show() at the very end.