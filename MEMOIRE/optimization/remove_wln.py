import pandas as pd

path = r'C:\Users\TONY B\OneDrive\Eliott\Eliott_dossier\154-capital\univers.xlsx'
df = pd.read_excel(path)

print('Colonnes avant:', df.columns.tolist())
print('WLN.PA present:', 'WLN.PA' in df.columns)

if 'WLN.PA' in df.columns:
    df = df.drop(columns=['WLN.PA'])
    df.to_excel(path, index=False)
    print('WLN.PA supprime de univers.xlsx')
    print('Colonnes apres:', df.columns.tolist())
else:
    print('WLN.PA pas trouve dans le fichier.')
