import pandas as pd

file_path = r"C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx"

# Lire les sheet names
xl_file = pd.ExcelFile(file_path)
print("Sheets disponibles:")
for i, sheet in enumerate(xl_file.sheet_names):
    print(f"  {i+1}. {sheet}")

# Lire la première sheet pour voir la structure
df = pd.read_excel(file_path, sheet_name=0)
print(f"\nPremière sheet: {xl_file.sheet_names[0]}")
print(f"Dimensions: {df.shape}")
print("\nPremières lignes:")
print(df.head(10))
print("\nNoms des colonnes:")
print(df.columns.tolist())
print("\nTypes de données:")
print(df.dtypes)
