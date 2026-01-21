import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import glob
import os

# Get all CSV files in the current directory
csv_files = glob.glob('*.csv')
print(f"Found {len(csv_files)} CSV files")

# Read and combine all CSV files
all_data = []
skipped_files = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        if df.empty or len(df.columns) == 0:
            skipped_files.append(csv_file)
            continue
        # Extract satellite name from filename
        sat_name = os.path.splitext(csv_file)[0]
        df['satellite_name'] = sat_name
        all_data.append(df)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        skipped_files.append(csv_file)
        continue

if skipped_files:
    print(f"Skipped {len(skipped_files)} empty or invalid CSV files")
    
if not all_data:
    print("Error: No valid CSV files found!")
    exit(1)

# Combine all dataframes
combined_df = pd.concat(all_data, ignore_index=True)

# Calculate path loss using free space path loss formula
# FSPL (dB) = 20*log10(d) + 20*log10(f) + 32.45
# where d is distance in km and f is frequency in MHz
# Using tx_freq_ghz converted to MHz
combined_df['path_loss_db'] = (20 * np.log10(combined_df['distance_km']) + 
                                20 * np.log10(combined_df['tx_freq_ghz'] * 1000) + 
                                32.45)

# Sort by time to ensure proper ordering
combined_df = combined_df.sort_values('timestamp')

# Check for large distances and filter them out
large_distance = combined_df[combined_df['distance_km'] > 7000]
if not large_distance.empty:
    print(f"\n⚠️  Filtering out {len(large_distance)} data points with distance > 7000 km")
    print("Satellites with filtered data:")
    suspicious_sats = large_distance['satellite_name'].unique()
    for sat in suspicious_sats:
        sat_data = large_distance[large_distance['satellite_name'] == sat]
        max_dist = sat_data['distance_km'].max()
        count = len(sat_data)
        print(f"  - {sat}: {count} points filtered (max distance = {max_dist:.2f} km)")
    print()
    
    # Filter out data points with distance > 7000 km
    combined_df = combined_df[combined_df['distance_km'] <= 7000]
    print(f"Remaining data points after distance filter: {len(combined_df)}\n")

# Check for unusually large relative velocities and filter them out
large_velocity = combined_df[combined_df['relative_velocity_kms'].abs() > 7.2]
if not large_velocity.empty:
    print(f"\n⚠️  Filtering out {len(large_velocity)} data points with relative velocity > 7.2 km/s")
    print("Satellites with filtered data:")
    suspicious_sats = large_velocity['satellite_name'].unique()
    for sat in suspicious_sats:
        sat_data = large_velocity[large_velocity['satellite_name'] == sat]
        max_vel = sat_data['relative_velocity_kms'].abs().max()
        count = len(sat_data)
        print(f"  - {sat}: {count} points filtered (max velocity = {max_vel:.2f} km/s)")
    print()
    
    # Filter out data points with velocity > 7.2 km/s
    combined_df = combined_df[combined_df['relative_velocity_kms'].abs() <= 7.2]
    print(f"Remaining data points: {len(combined_df)}\n")

# Convert timestamp to datetime and then to minutes from start
combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'])
start_time = combined_df['datetime'].min()
combined_df['time_from_start_minutes'] = (combined_df['datetime'] - start_time).dt.total_seconds() / 60

# Create the waterfall plot
fig, ax = plt.subplots(figsize=(12, 8))

# Create scatter plot with color representing path loss
scatter = ax.scatter(combined_df['relative_velocity_kms'], 
                     combined_df['time_from_start_minutes'],
                     c=combined_df['path_loss_db'],
                     cmap='viridis_r',
                     s=20,
                     alpha=0.6)

# Add colorbar with inverted scale (smaller values on top)
cbar = plt.colorbar(scatter, ax=ax, label='Path Loss (dB)')
cbar.ax.invert_yaxis()

# Set labels and title
ax.set_xlabel('Relative Velocity (km/s)', fontsize=12)
ax.set_ylabel('Time from Start (minutes)', fontsize=12)
ax.set_title('Satellite Waterfall Plot: Relative Velocity vs Time', fontsize=14, fontweight='bold')

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')

# Invert y-axis so time flows downward (waterfall effect)
ax.invert_yaxis()

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
plt.savefig('waterfall_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'waterfall_plot.png'")

# Save the combined data to CSV
combined_df.to_csv('waterfall_data.csv', index=False)
print("Combined data saved as 'waterfall_data.csv'")

# Show the plot
plt.show()

# Print some statistics
print(f"\nDataset Statistics:")
print(f"Total data points: {len(combined_df)}")
print(f"Time range: {combined_df['time_from_start_minutes'].min():.2f} - {combined_df['time_from_start_minutes'].max():.2f} minutes")
print(f"Relative velocity range: {combined_df['relative_velocity_kms'].min():.2f} - {combined_df['relative_velocity_kms'].max():.2f} km/s")
print(f"Path loss range: {combined_df['path_loss_db'].min():.2f} - {combined_df['path_loss_db'].max():.2f} dB")
print(f"Distance range: {combined_df['distance_km'].min():.2f} - {combined_df['distance_km'].max():.2f} km")
