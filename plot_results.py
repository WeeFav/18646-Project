import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data from your C++ output
try:
    df = pd.read_csv('results.csv')
except FileNotFoundError:
    print("Error: results.csv not found. Please run your C++ program first.")
    exit()

# 2. Calculate the average cycles for each resolution
# This groups all 20 combinations (eye/light) per resolution into one average
averages = df.groupby('Resolution')['Cycles'].mean().reset_index()

# 3. Create the plot
plt.figure(figsize=(10, 6))
bars = plt.bar(averages['Resolution'].astype(str), averages['Cycles'], color='#3498db', edgecolor='black')

# 4. Formatting for your 18-646 report
plt.title('Average Rasterization Time per Resolution', fontsize=14)
plt.xlabel('Resolution Scale', fontsize=12)
plt.ylabel('Average CPU Cycles (Linear Scale)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add the actual value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2e}', 
             va='bottom', ha='center', fontsize=10, fontweight='bold')

# If the difference between 16 and 128 is too large to see, 
# you can uncomment the line below to use a log scale:
# plt.yscale('log')

plt.tight_layout()
plt.savefig('result.png')
print("Results save to result.png")
plt.show()