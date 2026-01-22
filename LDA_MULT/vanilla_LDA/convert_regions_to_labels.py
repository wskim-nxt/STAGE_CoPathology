import pandas as pd
import os

# Read the label mapping file
labels_df = pd.read_csv('C:/Users/BREIN/Desktop/copathology_visualization_temp/data/FS_Lobes.csv')

# Create a dictionary mapping region names to their label numbers
# Format: 'ctx_lh_<region>' -> lh label, 'ctx_rh_<region>' -> rh label
region_to_label = {}

for _, row in labels_df.iterrows():
    roi = row['ROI']
    lh_label = row['lh']
    rh_label = row['rh']

    # Map both left and right hemisphere versions
    region_to_label[f'ctx_lh_{roi}'] = lh_label
    region_to_label[f'ctx_rh_{roi}'] = rh_label

# Function to convert region names to labels in a CSV file
def convert_csv_regions_to_labels(input_file, output_file):
    # Read the CSV
    df = pd.read_csv(input_file, index_col=0)

    # Replace index (region names) with label numbers
    new_index = [region_to_label.get(region, region) for region in df.index]
    df.index = new_index

    # Save to the same file (overwrite)
    df.to_csv(output_file)
    print(f"Converted {input_file} -> {output_file}")
    print(f"  Converted {len(df)} regions")

# Convert both files
convert_csv_regions_to_labels(
    'lda_topic_atrophy_patterns.csv',
    'lda_topic_atrophy_patterns_labels.csv'
)
convert_csv_regions_to_labels(
    'lda_diagnosis_atrophy_maps.csv',
    'lda_diagnosis_atrophy_maps_labels.csv'

)

print("/nConversion complete!")
