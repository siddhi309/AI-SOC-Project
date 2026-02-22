import pandas as pd
import numpy as np
import os


def load_and_merge_data(folder_path):
    print("Looking for CSV files in:", folder_path)

    if not os.path.exists(folder_path):
        print("❌ Folder does not exist!")
        exit()

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    print("CSV files found:", all_files)

    if len(all_files) == 0:
        print("❌ No CSV files found in this folder.")
        exit()

    df_list = []

    for file in all_files:
        file_path = os.path.join(folder_path, file)
        print(f"Loading {file}...")

        # 🔥 THIS IS THE IMPORTANT FIX
        df = pd.read_csv(file_path)

        df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df


def clean_data(df):
    df.columns = df.columns.str.strip()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if "Label" in df.columns:
        df["Label"] = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)
    else:
        print("❌ 'Label' column not found.")
        print("Available columns:", df.columns)
        exit()

    return df


if __name__ == "__main__":

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Your CSVs are inside data/raw
    data_folder = os.path.join(project_root, "data", "raw")

    print("Merging CSV files...")
    merged_data = load_and_merge_data(data_folder)

    print("Cleaning data...")
    clean_df = clean_data(merged_data)

    output_path = os.path.join(project_root, "clean_data.csv")

    print("Saving cleaned dataset...")
    clean_df.to_csv(output_path, index=False)

    print("✅ Preprocessing complete.")
    print("Saved at:", output_path)