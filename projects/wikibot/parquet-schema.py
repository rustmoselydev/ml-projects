# Discover the schema of a large parquet file

import pandas as pd

df = pd.read_parquet("./archive/a.parquet", engine="pyarrow", columns=None)
print(df.head(0))