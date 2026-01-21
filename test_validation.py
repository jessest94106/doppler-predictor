#!/usr/bin/env python3
"""
Clean CSV data by removing rows with unrealistic distance/velocity values
"""
import os
import sys
import csv
from pathlib import Path

# Get data directory from command line or use default
if len(sys.argv) > 1:
    data_dir = sys.argv[1]
else:
    data_dir = "/Users/jesse/Documents/doppler_predictor/satellites_20260120_184412"

if not os.path.exists(data_dir):
    print(f"Error: Directory not found: {data_dir}")
    sys.exit(1)

print("CSV Data Cleaner - Removing Invalid Data Points")
print("="*70)
print(f"Directory: {data_dir}\n")

# Validation thresholds
MAX_DISTANCE_KM = 5000  # Maximum realistic distance for LEO satellite
MAX_VELOCITY_KMS = 20   # Maximum realistic velocity

total_files = 0
total_points_before = 0
total_points_after = 0
files_cleaned = 0
files_deleted = []
satellites_with_issues = {}

# Process each CSV file
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'waterfall_data.csv'])

for filename in csv_files:
    filepath = os.path.join(data_dir, filename)
    total_files += 1
    
    # Read all rows
    clean_rows = []
    bad_rows = 0
    fieldnames = None
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_points_before += 1
            
            try:
                distance_km = float(row['distance_km'])
                velocity_kms = float(row['relative_velocity_kms'])
                
                # Check if values are realistic
                is_valid = True
                issue_reasons = []
                
                if distance_km > MAX_DISTANCE_KM or distance_km < 0:
                    is_valid = False
                    issue_reasons.append(f"distance={distance_km:.0f}km")
                    
                if abs(velocity_kms) > MAX_VELOCITY_KMS:
                    is_valid = False
                    issue_reasons.append(f"velocity={velocity_kms:.1f}km/s")
                
                if is_valid:
                    clean_rows.append(row)
                else:
                    bad_rows += 1
                    sat_name = row['satellite']
                    if sat_name not in satellites_with_issues:
                        satellites_with_issues[sat_name] = {
                            'count': 0,
                            'issues': issue_reasons[0]
                        }
                    satellites_with_issues[sat_name]['count'] += 1
                    
            except (ValueError, KeyError) as e:
                # Skip rows with parsing errors
                bad_rows += 1
                continue
    
    # Decide what to do with the file
    if bad_rows > 0:
        files_cleaned += 1
        
    if len(clean_rows) == 0:
        # Delete file if all data is bad
        print(f"❌ {filename}: All {bad_rows} rows invalid - DELETING FILE")
        os.remove(filepath)
        files_deleted.append(filename)
        
        # Also delete corresponding plot if it exists
        plot_file = filepath.replace('.csv', '_plot.png')
        if os.path.exists(plot_file):
            os.remove(plot_file)
            
    elif bad_rows > 0:
        # Rewrite file with only clean data
        print(f"🔧 {filename}: Removed {bad_rows} invalid rows, kept {len(clean_rows)} rows")
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)
        total_points_after += len(clean_rows)
    else:
        # File is clean
        total_points_after += len(clean_rows)

print("\n" + "="*70)
print("CLEANING SUMMARY")
print("="*70)
print(f"Total CSV files processed: {total_files}")
print(f"Files with invalid data: {files_cleaned}")
print(f"Files deleted (all data invalid): {len(files_deleted)}")
print(f"\nData points before: {total_points_before:,}")
print(f"Data points after:  {total_points_after:,}")
print(f"Data points removed: {total_points_before - total_points_after:,} ({100*(total_points_before - total_points_after)/total_points_before:.1f}%)")

if satellites_with_issues:
    print(f"\nProblematic satellites ({len(satellites_with_issues)} total):")
    sorted_sats = sorted(satellites_with_issues.items(), key=lambda x: x[1]['count'], reverse=True)
    for sat_name, info in sorted_sats[:15]:  # Show top 15
        print(f"  {sat_name}: {info['count']} bad rows ({info['issues']})")
    
    if len(sorted_sats) > 15:
        print(f"  ... and {len(sorted_sats) - 15} more satellites")

if files_deleted:
    print(f"\nDeleted files ({len(files_deleted)}):")
    for fname in files_deleted[:10]:
        print(f"  - {fname}")
    if len(files_deleted) > 10:
        print(f"  ... and {len(files_deleted) - 10} more")

print("\n✓ Data cleaning complete!")
print(f"  Validation thresholds:")
print(f"  - Distance: 0-{MAX_DISTANCE_KM:,} km")
print(f"  - Velocity: -{MAX_VELOCITY_KMS} to +{MAX_VELOCITY_KMS} km/s")
print("="*70)
