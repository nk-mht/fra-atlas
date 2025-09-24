# frontend/app.py

import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
st.title("FRA Atlas — Prototype (Using your CSV)")

# ====== CONFIGURE YOUR RAW CSV URL HERE =======
CSV_URL = "https://raw.githubusercontent.com/nk-mht/fra-atlas/data/RS_Session_265_AU_1896_B_1.csv"
# If you have a GeoJSON for Indian states, put its path here (in data/)
GEOJSON_PATH = "data/india_states.geojson"  

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/nk-mht/fra-atlas/develop/RS_Session_265_AU_1896_B_1.csv")

    except Exception as e:
        st.error(f"Could not load CSV from GitHub: {e}")
        return None
    return df

df = load_data()
if df is None:
    st.stop()

st.sidebar.write("Data preview")
st.sidebar.dataframe(df.head(10))

# Let user pick a numeric metric column to visualize
# Find numeric columns (floats/ints)
num_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()
if not num_cols:
    st.error("No numeric columns found in your CSV. AI / clustering layer skipped.")
    metric = None
else:
    metric = st.sidebar.selectbox("Numeric metric to map by state/row", options=num_cols)

# AI / clustering layer
if metric:
    # Scale and cluster
    X = df[num_cols].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # choose cluster count = min(3, number of rows)
    k = min(3, len(df))
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(Xs)
    df["cluster"] = labels
else:
    df["cluster"] = 0  # fallback

# Display table
st.write("### Data Sample")
st.dataframe(df.head(20))

# Build map
m = folium.Map(location=[22.0,78.0], zoom_start=5)

# If you have geojson loaded, draw choropleth
got_geo = False
try:
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        india_geo = json.load(f)
        got_geo = True
except FileNotFoundError:
    st.warning("GeoJSON file not found locally. Choropleth layer disabled.")
    india_geo = None

if got_geo and metric:
    folium.Choropleth(
        geo_data=india_geo,
        data=df,
        columns=["state", metric],
        key_on="feature.properties.NAME",  # or adjust property name in your geojson
        fill_color="YlOrRd",
        legend_name=f"{metric}",
        fill_opacity=0.7,
        line_opacity=0.2
    ).add_to(m)

# Markers colored by cluster or by metric value
for _, r in df.iterrows():
    # If you have lat/lon columns, use them; else skip markers
    if "latitude" in r.keys() and "longitude" in r.keys():
        try:
            lat = float(r["latitude"])
            lon = float(r["longitude"])
        except:
            continue
        col = "red"
        if "cluster" in r.keys():
            # color by cluster number
            if r["cluster"] == 0:
                col = "red"
            elif r["cluster"] == 1:
                col = "orange"
            else:
                col = "green"
        folium.CircleMarker([lat, lon],
                            radius=5,
                            color=col,
                            popup=f"State: {r.get('state', '')}<br>{metric}: {r.get(metric, '')}",
                            fill=True).add_to(m)

st.write("### Map")
st_folium(m, width=900, height=600)

st.write("### Notes")
st.write("- Clustering uses KMeans on numeric columns (if present).")
st.write("- If no geojson, the map will show only markers or fallback behavior.")
