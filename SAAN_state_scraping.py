from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np
import requests 
import os
import re 

output_folder = "SAAH_polling_data"
os.makedirs(output_folder, exist_ok=True)

url = "https://www.wahlrecht.de/umfragen/landtage/sachsen-anhalt.htm#fn0-tab3"

response = requests.get(url)
soup = BeautifulSoup(response.content, "lxml")

table = soup.find("table", {"class": "wilko"})

headers = []
thead = table.find("thead")
spacer_indices = []
for i, cell in enumerate(thead.find_all(["th", "td"])):
    cls = cell.get("class", [])
    text = cell.get_text(strip=True)
    if "part" in cls:
        headers.append(text.replace(".",""))
    elif "dat2" in cls:
        headers.append("N Polled")
    else:
        headers.append(text)


# --- Extract data rows ---
data = []
tbody = table.find("tbody")
for row in tbody.find_all("tr"):
    cells = row.find_all(["td", "th"])
    if not cells:
        continue

    row_data = []
    for i, cell in enumerate(cells):
        if i in spacer_indices:
            continue

        colspan = int(cell.get("colspan", 1))
        if colspan > 1:
            text = cell.get_text(strip=True)
            row_data.append(text)
            row_data.extend([""] * (len(headers) - 1))
            break

        row_data.append(cell.get_text(strip=True))

    row_data = row_data[:len(headers)] + [""] * (len(headers) - len(row_data))
    data.append(row_data)


non_empty_indices = [i for i, h in enumerate(headers) if h.strip()]
headers = [headers[i] for i in non_empty_indices]
data = [[row[i] for i in non_empty_indices] for row in data]

df = pd.DataFrame(data, columns=headers)

print(df.head)


party_cols = ["CDU","SPD","GRÜNE","FDP","LINKE","AfD","FW","BSW"]

df = df.drop(columns="Auftraggeber")

df = df.rename(columns= {"Datum":"Date", "Institut":"institute"})

df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

df = df[~df['N Polled'].str.contains('Landtagswahl', na=False)]
df["N Polled"] = (
    df["N Polled"]
    .str.extract(r'(\d+\.\d{3})')[0]
    .str.replace(".", "", regex=False)
    ##.astype(int)
)

df["Sonstige"] = df["Sonstige"].apply(
    lambda x: sum(
        float(n.replace(",", "."))
        for n in re.findall(r'(\d+(?:[.,]\d+)?)\s*%', str(x))
    )
)
df[party_cols] = (
    df[party_cols]
    .replace("–", pd.NA)
    .replace("<NA>", pd.NA)            # astype(str) turns NA into the literal "<NA>"
    .apply(lambda col: col.str.replace(" %", "", regex=False))
    .apply(lambda col: col.str.replace(",", ".", regex=False))
    .apply(pd.to_numeric, errors="coerce")
)

df[party_cols] = df[party_cols].fillna(0)


df["Sonstige"] = df["Sonstige"] + df["FW"]


df = df.drop(columns="FW")

filepath = os.path.join(output_folder, "SAN.csv")
df.to_csv(filepath, index=False, encoding="utf-8-sig")