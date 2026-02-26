import pandas as pd
import os

source = r'C:\Users\BALAMURUGAN\OneDrive\Desktop\Data Science\Machine Learning\zoho_dataset\Rotten_Tomatoes_Movies3.xls\Rotten_Tomatoes_Movie.csv'

if not os.path.exists(source):
    print(f"Error: {source} not found.")
    exit(1)

print(f"Loading data from {source}...")
try:
    print("Detected binary header, attempting to read as Excel (.xls)...")
    df = pd.read_excel(source, engine='xlrd')
except Exception as e:
    print(f"Error loading as Excel: {e}")
    try:
        print("Falling back to reading as CSV...")
        df = pd.read_csv(source, encoding='latin1')
    except Exception as e2:
        print(f"Error loading as CSV: {e2}")
        exit(1)

print("\n--- Data Info ---")
print(df.info())

print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\n--- Summary Statistics (Audience Rating) ---")
if 'audience_rating' in df.columns:
    print(df['audience_rating'].describe())
else:
    print("Warning: 'audience_rating' column not found.")
    print("Columns:", df.columns.tolist())

# Save stats for the implementation plan
stats_path = os.path.join(os.getcwd(), 'data_stats.txt')
with open(stats_path, 'w') as f:
    f.write(f"Shape: {df.shape}\n")
    f.write(f"Columns: {df.columns.tolist()}\n")
    f.write(f"Missing Values:\n{missing.to_string()}\n")

print(f"\nStats saved to {stats_path}")
