import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt # Added for the Overview section's matplotlib plot
import seaborn as sns # Added for the Overview section's seaborn plot
import json # Added for the Overview section's geojson
import unicodedata # Added for the Overview section's normalization
import io # Added for the Overview section's export

# === CONFIG ===
st.set_page_config(page_title="📊 Bogotá Descriptive Dashboard", layout="wide")

# === LOAD DATA ===
FILE_PATH = r"C:\Study document\6611 Dataset\Final\MASTER_CLEANED1.xlsx"
df = pd.read_excel(FILE_PATH, dtype={"Dane Code": str})
# This line is crucial: it makes "Name of Establishment" -> "Name Of Establishment"
df.columns = df.columns.str.strip().str.title()

# === RENAME INCONSISTENT COLUMNS (for main df) ===
# We are *not* renaming "Name Of Establishment" here, as per your request.
df.rename(columns={
    "Enrolled Students (Last 3 Years)": "Enrollment",
    "Pass Rate": "Pass Rate",
    "Dropout Rate": "Dropout Rate",
    "Failure Rate": "Failure Rate",
    "Absorption Rate In Higher Education": "Absorption Rate"
}, inplace=True)

# === FILTER ONLY CONCESSION SCHOOLS ===
df["Concession"] = df["Concession"].astype(str).str.strip().str.lower()
df = df[df["Concession"] == "yes"]
# Keep "Name Of Establishment" here
df = df.dropna(subset=["Administrator", "Name Of Establishment", "Year"])
df["Administrator"] = df["Administrator"].str.upper()
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)

# === CLEAN NUMERIC COLUMNS ===
subject_cols = [
    "Mathematics Index", "Natural Sciences Index", "Social And Citizenship Index",
    "Critical Reading Index", "English Index", "Total Index"
]
rate_cols = ["Pass Rate", "Dropout Rate", "Failure Rate", "Absorption Rate"]
enroll_cols = ["Evaluated (Last 3 Years)"]  # Enrollment removed
rank_col = ["Ranking"]
all_indicators = subject_cols + rate_cols + enroll_cols

for col in all_indicators:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# === SIDEBAR SECTION SELECTOR ===
with st.sidebar:
    st.header("Select Section")
    section = st.selectbox("Choose Section", ["Overview", "By Locality", "By Concession Schools"])

# === DYNAMIC TITLE ===
section_titles = {
    "Overview": "📊 Overview of Bogotá Education System",
    "By Locality": "📍 Analysis by Locality",
    "By Concession Schools": "🏫 Concession School Performance"
}
st.title(section_titles.get(section, "📊 Education Dashboard"))

# === SECTION: BY CONCESSION SCHOOLS ===
if section == "By Concession Schools":
    with st.sidebar:
        st.header("🔎 Filters")
        view_mode = st.radio("Select View Mode", ["Total View", "Administrator Level", "School Level"])
        indicator = st.selectbox("Select Indicator", all_indicators)

        admin_filter = []
        school_filter = []

        if view_mode in ["Administrator Level", "School Level"]:
            admin_filter = st.multiselect("Select Administrator(s)", sorted(df["Administrator"].unique()))

        if view_mode == "School Level" and admin_filter:
            # Keep "Name Of Establishment" here
            filtered_schools = sorted(df[df["Administrator"].isin(admin_filter)]["Name Of Establishment"].unique())
            school_filter = st.multiselect("Select School(s)", filtered_schools)

    # === TOTAL VIEW ===
    if view_mode == "Total View":
        st.subheader(f"📊 Average {indicator} per Administrator (All Years)")
        avg_df = df.groupby("Administrator")[indicator].mean().reset_index().sort_values(by=indicator, ascending=False)
        fig_avg = px.bar(avg_df, x="Administrator", y=indicator, color="Administrator",
                         title=f"Average {indicator} per Administrator (All Years)", height=700)
        fig_avg.update_layout(yaxis=dict(tickfont=dict(size=13)), xaxis=dict(tickfont=dict(size=13)))
        st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader(f"📈 {indicator} Over Time by Administrator")
        chart_data = df.groupby(["Administrator", "Year"])[indicator].mean().reset_index()
        fig = px.line(chart_data, x="Year", y=indicator, color="Administrator",
                      markers=True, title=f"{indicator} Over Time by Administrator", height=600)
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏅 Ranking Contribution per Administrator")
        rank_df = df.dropna(subset=["Ranking"])
        rank_summary = rank_df.groupby(["Administrator", "Ranking"]).size().reset_index(name="Count")
        total_counts = rank_summary.groupby("Administrator")["Count"].transform("sum")
        rank_summary["Percentage"] = (rank_summary["Count"] / total_counts * 100).round(2)
        fig_rank = px.bar(
            rank_summary, x="Administrator", y="Count", color="Ranking", text="Percentage",
            title="Ranking Distribution by Administrator (%)", height=600
        )
        fig_rank.update_traces(textposition='inside')
        fig_rank.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig_rank, use_container_width=True)

    # === ADMINISTRATOR LEVEL ===
    elif view_mode == "Administrator Level" and admin_filter:
        st.subheader(f"📈 {indicator} Over Time by Administrator")
        chart_data = df[df["Administrator"].isin(admin_filter)]
        chart_grouped = chart_data.groupby(["Administrator", "Year"])[indicator].mean().reset_index()
        fig = px.line(chart_grouped, x="Year", y=indicator, color="Administrator",
                      markers=True, title=f"{indicator} Over Time by Administrator", height=600)
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Ranking Breakdown by School")
        rank_school = chart_data.dropna(subset=["Ranking"])
        # Keep "Name Of Establishment" here
        fig_rank_school = px.histogram(rank_school, x="Ranking", color="Name Of Establishment", barmode="group",
                                       title="Ranking Composition by School", height=500)
        st.plotly_chart(fig_rank_school, use_container_width=True)

        st.markdown("### 🏫 Schools under Selected Administrator(s)")
        # Keep "Name Of Establishment" here
        school_list = chart_data[["Administrator", "Name Of Establishment"]].drop_duplicates().sort_values(["Administrator", "Name Of Establishment"])
        st.dataframe(school_list, use_container_width=True)

    # === SCHOOL LEVEL ===
    elif view_mode == "School Level" and school_filter:
        for school in school_filter:
            # Keep "Name Of Establishment" here
            school_data = df[df["Name Of Establishment"] == school]
            st.subheader(f"📚 {school} — {indicator} Over Time")
            fig_school = px.line(school_data, x="Year", y=indicator, color="Administrator",
                                 markers=True, title=f"{indicator} Over Time at {school}", height=500)
            fig_school.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_school, use_container_width=True)

            st.markdown("#### 🔍 Key Details")
            recent_data = school_data.sort_values("Year", ascending=False).head(1).T
            recent_data.columns = ["Most Recent Record"]
            st.dataframe(recent_data, use_container_width=True)

# === SECTION: OVERVIEW ===
elif section == "Overview":
    with st.sidebar:
        st.header("📊 Overview Filters")

    # ============================== #
    # 📁 Load Datasets
    # ============================== #
    school_path = r"C:\Study document\6611 Dataset\Final\Copy of bogota_all_schools_ranking_saber_11.xlsx"
    map_school_path = r"C:\Study document\6611 Dataset\Final\Copy of bogota_all_schools_ranking_saber_11.xlsx"
    poverty_path = r"C:\Study document\6611 Dataset\Final\poverty_inequality_data_bogota.xlsx"
    geojson_path = r"C:\Study document\6611 Dataset\Final\bta_localidades.geojson"

    school_df = pd.read_excel(school_path)
    # Apply .str.title() immediately for consistency
    school_df.columns = school_df.columns.str.strip().str.title()

    map_school_df = pd.read_excel(map_school_path)
    # Apply .str.title() immediately for consistency
    map_school_df.columns = map_school_df.columns.str.strip().str.title()

    poverty_df = pd.read_excel(poverty_path)

    with open(geojson_path, "r", encoding="utf-8") as f:
        bogota_geo = json.load(f)

    # ============================== #
    # 🔤 Normalize Location Names
    # ============================== #
    def normalize_location(s):
        s = str(s)
        s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
        return s.strip().upper()

    location_name_map = {
        "SANTAFE": "SANTA FE",
        "LA CANDELARIA": "CANDELARIA",
        "ANTONIO NARINO": "ANTONIO NARIÑO"
    }

    # ============================== #
    # 🧹 Clean and Prepare Data
    # ============================== #
    poverty_df = poverty_df.rename(columns={
        "Año": "year",
        "Localidad": "location_name",
        "Indicador": "indicator",
        "Valor": "value"
    })
    poverty_df["location_name"] = poverty_df["location_name"].apply(normalize_location)
    poverty_df["location_name"] = poverty_df["location_name"].replace(location_name_map)
    poverty_df["value"] = pd.to_numeric(poverty_df["value"], errors="coerce")

    school_df = school_df.rename(columns={
        "Year": "year",
        "Location Name": "location_name", # After .title(), "Location name" becomes "Location Name"
        "Sector": "sector",
        "Concession": "concession",
        "Mathematics Index": "mathematics_index",
        "Natural Sciences Index": "natural_sciences_index",
        "Social And Citizenship Index": "social_and_citizenship_index", # After .title(), "Social and Citizenship Index" becomes "Social And Citizenship Index"
        "Critical Reading Index": "critical_reading_index",
        "English Index": "english_index",
        "Total Index": "total_index",
        "Ranking": "ranking"
        # Removed "Name Of Establishment": "school_name" here as per your request
    })

    school_df = school_df.dropna(subset=['location_name'])
    school_df = school_df[school_df['location_name'].str.strip() != ""]
    school_df['location_name'] = school_df['location_name'].apply(normalize_location)
    school_df['location_name'] = school_df['location_name'].replace(location_name_map)

    map_school_df = map_school_df.rename(columns={
        "Location Name": "location_name", # After .title(), "Location name" becomes "Location Name"
        # Removed "Name Of Establishment": "school_name" here as per your request
    })
    map_school_df['location_name'] = map_school_df['location_name'].apply(normalize_location)
    map_school_df['location_name'] = map_school_df['location_name'].replace(location_name_map)

    school_df["sector_type"] = school_df.apply(
        lambda row: "CONCESSION" if str(row["sector"]).upper() == "PUBLIC" and str(row["concession"]).strip().upper() == "YES"
        else ("PUBLIC" if str(row["sector"]).upper() == "PUBLIC" else "PRIVATE"),
        axis=1
    )

    # ============================== #
    # 🧭 Sidebar Filters
    # ============================== #
    school_years = sorted(school_df["year"].dropna().unique())
    selected_year = st.sidebar.selectbox("Select Year", school_years, index=len(school_years) - 1)

    localities = ['ALL'] + sorted(school_df['location_name'].unique())
    selected_locality = st.sidebar.selectbox("Select Locality", localities)

    poverty_valid_years = [2011, 2014, 2017, 2021]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"🗓️ School data: **{min(school_years)}–{max(school_years)}**\n\n📉 Poverty data: **{', '.join(map(str, poverty_valid_years))}**")

    # ============================== #
    # 🎯 Filter Data
    # ============================== #
    filtered_df = school_df[school_df["year"] == selected_year].copy()
    # Ensure "Year" in map_school_df is handled correctly if it's not already numeric
    if 'Year' in map_school_df.columns:
        map_school_df['Year'] = pd.to_numeric(map_school_df['Year'], errors='coerce').fillna(-1).astype(int) # Handle potential NaNs for conversion
    map_filtered_df = map_school_df[map_school_df["Year"] == selected_year].copy()


    if selected_locality != 'ALL':
        filtered_df = filtered_df[filtered_df['location_name'] == selected_locality]
        map_filtered_df = map_filtered_df[map_filtered_df['location_name'] == selected_locality]

    poverty_years_available = sorted(poverty_df['year'].unique())
    latest_poverty_year = max([y for y in poverty_years_available if y <= selected_year], default=None)
    poverty_year = poverty_df[(poverty_df["year"] == latest_poverty_year) & (poverty_df["indicator"].str.lower().str.contains("extrema"))]
    if selected_locality != 'ALL':
        poverty_year = poverty_year[poverty_year['location_name'] == selected_locality]

    # ============================== #
    # 📥 Export Data
    # ============================== #
    st.markdown("## 📥 Export Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Filtered Data (CSV)", data=csv, file_name=f"filtered_schools_{selected_year}.csv", mime='text/csv')

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="Filtered Data")
    st.download_button("📊 Download Filtered Data (Excel)", data=excel_buffer.getvalue(), file_name=f"filtered_schools_{selected_year}.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # ============================== #
    # 📊 KPI Cards
    # ============================== #
    st.title(":bar_chart: Bogotá Educational Index")
    total = len(filtered_df)
    private = len(filtered_df[filtered_df["sector_type"] == "PRIVATE"])
    public = len(filtered_df[filtered_df["sector_type"] == "PUBLIC"])
    concession = len(filtered_df[filtered_df["sector_type"] == "CONCESSION"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎓 Total Schools", total)
    col2.metric("🏢 Private", private)
    col3.metric("🏫 Public", public)
    col4.metric("🪡 Concession (PPP)", concession)

    # ============================== #

    # ============================== #
    # 🌍 Maps
    # ============================== #
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.subheader("School Distribution")
        # Ensure 'Name Of Establishment' exists in map_filtered_df if it's the identifier for counting
        # Assuming you want to count schools, not necessarily display their name here,
        # but if `map_school_df` has `Name Of Establishment`, it's good.
        # It's better to group by location_name as done before.
        map_df = map_filtered_df.groupby("location_name").size().reset_index(name="count")
        fig = px.choropleth_mapbox(
            map_df, geojson=bogota_geo, locations="location_name",
            featureidkey="properties.NOMBRE", color="count",
            color_continuous_scale="Viridis", mapbox_style="carto-positron",
            zoom=9.2, center={"lat": 4.65, "lon": -74.1}, opacity=0.85, height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_map2:
        st.subheader("Extreme Monetary Poverty")
        if not poverty_year.empty:
            fig_poverty = px.choropleth_mapbox(
                poverty_year, geojson=bogota_geo, locations="location_name",
                featureidkey="properties.NOMBRE", color="value",
                color_continuous_scale="Reds", mapbox_style="carto-positron",
                zoom=9.2, center={"lat": 4.65, "lon": -74.1}, opacity=0.85, height=500
            )
            st.plotly_chart(fig_poverty, use_container_width=True)
        else:
            st.info("Poverty data not available for this year.")

    # ============================== #
    # 📈 Additional Charts
    # ============================== #
    st.subheader("Poverty vs School Availability")
    # This `school_count` is from map_filtered_df, which should have 'Name Of Establishment'
    school_counts = map_filtered_df.groupby("location_name").size().reset_index(name="school_count")
    poverty_scores = poverty_year[["location_name", "value"]].rename(columns={"value": "extreme_poverty"})
    merged = pd.merge(school_counts, poverty_scores, on="location_name", how="inner")
    fig_scatter = px.scatter(
        merged, x="extreme_poverty", y="school_count", text="location_name",
        size="school_count", color="extreme_poverty", color_continuous_scale="Plasma"
    )
    fig_scatter.update_traces(textposition="top center")
    st.plotly_chart(fig_scatter)

    st.subheader("Sector Distribution by Locality")
    sector_data = filtered_df.groupby(["location_name", "sector_type"]).size().reset_index(name="count")
    fig_sector = px.bar(sector_data, x="location_name", y="count", color="sector_type", barmode="stack", height=500)
    st.plotly_chart(fig_sector)

    st.subheader("Saber 11 Index by Sector over Time")
    subject_cols = [
        "mathematics_index", "natural_sciences_index",
        "social_and_citizenship_index", "critical_reading_index",
        "english_index", "total_index"
    ]
    selected_subject = st.selectbox("Select Subject", subject_cols, index=subject_cols.index("total_index"))
    avg_by_sector = school_df.groupby(["year", "sector_type"])[selected_subject].mean().reset_index()
    fig = px.line(avg_by_sector, x="year", y=selected_subject, color="sector_type", markers=True)
    fig.update_layout(yaxis_title="Saber 11 Index", height=500)
    st.plotly_chart(fig)

    st.subheader("Ranking Distribution by Sector")
    rank_data = filtered_df.dropna(subset=['ranking'])
    rank_grouped = rank_data.groupby(['sector_type', 'ranking']).size().reset_index(name='count')
    fig_rank = px.sunburst(rank_grouped, path=['sector_type', 'ranking'], values='count')
    st.plotly_chart(fig_rank)

    st.subheader("Poverty vs Saber 11 Average")
    poverty_rate = poverty_year[["location_name", "value"]].rename(columns={"value": "poverty_rate"})
    school_year = filtered_df.copy()
    index_cols = [col for col in school_year.columns if col.endswith("_index") and col != "total_index"]
    school_year["saber_avg"] = school_year[index_cols].mean(axis=1)
    saber_avg = school_year.groupby("location_name")["saber_avg"].mean().reset_index()
    merged = pd.merge(poverty_rate, saber_avg, on="location_name", how="inner")
    merged = merged.sort_values(by="poverty_rate", ascending=False)

    fig, ax1 = plt.subplots(figsize=(16, 7))
    sns.barplot(x="location_name", y="poverty_rate", data=merged, ax=ax1, color="orange", zorder=1)
    ax2 = ax1.twinx()
    sns.lineplot(x="location_name", y="saber_avg", data=merged, ax=ax2, color="green", marker="o", linewidth=2.5, zorder=2)
    ax1.set_ylabel("Extreme Poverty Rate (%)")
    ax2.set_ylabel("Saber 11 Avg Index")
    ax1.set_xticklabels(merged["location_name"], rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    # 🧠 Average, Top 10, Bottom 10 Saber Index
    # ============================== #
    st.subheader("📌 Saber 11 Performance Summary")
    index_df = filtered_df.dropna(subset=["total_index"])
    if not index_df.empty:
        avg_score = index_df["total_index"].mean()
        max_score = index_df["total_index"].max()
        min_score = index_df["total_index"].min()

        st.markdown(f"""
            - 🔢 **Average Total Index**: {avg_score:.2f}
            - 🏆 **Highest Total Index**: {max_score:.2f}
            - 🚨 **Lowest Total Index**: {min_score:.2f}
            """)

        st.markdown("### 🥇 Top 10 Schools by Total Index")
        top10 = index_df.sort_values(by="total_index", ascending=False).head(10)[
            ["Name Of Establishment", "location_name", "total_index", "sector_type"]] # KEPT "Name Of Establishment"
        st.dataframe(top10.reset_index(drop=True))

        st.markdown("### 🛑 Bottom 10 Schools by Total Index")
        bottom10 = index_df.sort_values(by="total_index", ascending=True).head(10)[
            ["Name Of Establishment", "location_name", "total_index", "sector_type"]] # KEPT "Name Of Establishment"
        st.dataframe(bottom10.reset_index(drop=True))
    else:
        st.info("No index data available for this year and locality.")


# === SECTION: LOCALITY PLACEHOLDER ===
elif section == "By Locality":
    st.info("📍 Locality-based analysis is not yet available.")