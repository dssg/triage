import tempfile
import random
from contextlib import contextmanager

import pandas as pd
import requests
from tqdm import tqdm
import re


@contextmanager
def download(url):
    "download `url` to a file, returning the file name"
    with tempfile.NamedTemporaryFile(mode="wb") as f:
        response = requests.get(url, stream=True)
        for data in tqdm(response.iter_content()):
            f.write(data)
        f.flush()
        yield f.name


def create_subset(src, dest, n=250):
    "Given a csv file `src`, create a subset `dest` with `n` unique entities"
    df = pd.read_csv(src)
    lics = pd.unique(df["License #"])
    sublics = lics[random.sample(range(0, len(lics)), n)]
    subset = df[df["License #"].isin(sublics)]
    # Make the column names a little more readable
    subset.columns = map(clean_column_name, subset.columns)
    subset.to_csv(dest, index=False)


def clean_column_name(col):
    col = col.lower()
    col = col.replace(" ", "_")
    col = col.replace("#", "no")
    return re.sub("[\W]+", "", col)


if __name__ == "__main__":
    # download the entire Chicago restaurant inspections CSV file
    with download(
        "https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD"
    ) as f:
        create_subset(f, "food_inspections_subset.csv")
