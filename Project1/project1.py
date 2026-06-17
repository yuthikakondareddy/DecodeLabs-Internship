import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")

df = df.apply(pd.to_numeric, errors='coerce')

df.fillna(df.mean(numeric_only=True), inplace=True)

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

df["Total"] = df.select_dtypes(include=np.number).sum(axis=1)
df["Average"] = df.select_dtypes(include=np.number).mean(axis=1)
df["Range"] = df.select_dtypes(include=np.number).max(axis=1) - df.select_dtypes(include=np.number).min(axis=1)

df.to_csv("cleaned_data.csv", index=False)

print("Data cleaned and saved successfully!")