import streamlit as st
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpInteger
import plotly.express as px

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Strategic Budget Optimizer", layout="wide")
st.title("📊 Strategic Budget Optimizer")

# ------------------ Section Selection ------------------
section = st.sidebar.selectbox(
    "Select Section",
    ["1. School Investment Prioritization", "2. Budget Allocation", "3. Funding Simulation"]
)

# ------------------ Base Data ------------------
default_data = {
    "Infrastructure": {
        "Primary Classrooms": {"cost": 4999.0, "weight": 0.12},
        "TIM Classroom": {"cost": 11253.0, "weight": 0.08},
        "Science Lab (Primary)": {"cost": 9451.0, "weight": 0.07},
        "Library": {"cost": 11976.0, "weight": 0.11},
        "Bilingualism Room": {"cost": 14790.0, "weight": 0.04},
        "Reading Room": {"cost": 1639.0, "weight": 0.03},
        "Workstations": {"cost": 600.0, "weight": 0.175},
        "Science Lab Workstations": {"cost": 700.0, "weight": 0.07},
        "Shared Spaces": {"cost": 1250.0, "weight": 0.075},
        "Storage & Lockers": {"cost": 360.0, "weight": 0.03},
        "Kitchen & Dining": {"cost": 3.0, "weight": 0.03}
    },
    "Training": {
        "Digital Pedagogy": {"cost": 160.0, "weight": 0.25},
        "STEM/STEAM Instruction": {"cost": 250.0, "weight": 0.30},
        "School Leadership": {"cost": 200.0, "weight": 0.20},
        "Inclusion & SEL": {"cost": 150.0, "weight": 0.25},
    },
    "Technology": {
        "Educational Software & Licenses": {"cost": 100.0, "weight": 0.30},
        "Network Infrastructure": {"cost": 600.0, "weight": 0.25},
        "Audio/Visual Tools": {"cost": 250.0, "weight": 0.20},
        "Cybersecurity": {"cost": 375.0, "weight": 0.15},
        "IT Maintenance & Support": {"cost": 250.0, "weight": 0.10},
    },
    "Salaries": {
        "Base Teacher": {"cost": 15000.0, "weight": 0.1887},
        "Specialized Teacher": {"cost": 24000.0, "weight": 0.2830},
        "Administrator": {"cost": 10000.0, "weight": 0.1509},
        "Manager": {"cost": 28000.0, "weight": 0.3774},
    }
}

category_data = copy.deepcopy(default_data)

# ------------------ Section 1: TOPSIS School Prioritization ------------------
if section == "1. School Investment Prioritization":
    st.header("🏢 School Investment Prioritization")

    # --- Load Sample Data ---
    st.markdown("### 🎓 Sample Prioritization Result (TOPSIS)")

    sample_df = None
    try:
        sample_df = pd.read_csv(r"C:\Study document\6611 Dataset\Final\sampled_schools_model.csv")
    except FileNotFoundError:
        st.warning("⚠️ Sample file not found. Please upload your own CSV file below to see the prioritization results.")

    if sample_df is not None:
        sample_df.columns = sample_df.columns.str.strip().str.replace(" ", "_")

        indicators = [
            "Mathematics_Index", "Natural_Sciences_Index", "Social_and_Citizenship_Index",
            "Critical_Reading_Index", "English_Index",
            "Aprobacion_(Passrate)", "Reprobacion_(Fail_rate)", "Descercion_(Dropoutrate)"
        ]

        for col in ["Aprobacion_(Passrate)", "Reprobacion_(Fail_rate)", "Descercion_(Dropoutrate)"]:
            if sample_df[col].dtype == object:
                sample_df[col] = sample_df[col].str.replace('%', '').astype(float) / 100

        weights = [0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.05]
        benefit_criteria = {
            "Mathematics_Index": True, "Natural_Sciences_Index": True, "Social_and_Citizenship_Index": True,
            "Critical_Reading_Index": True, "English_Index": True,
            "Aprobacion_(Passrate)": True, "Reprobacion_(Fail_rate)": False, "Descercion_(Dropoutrate)": False
        }

        # Normalize and Apply TOPSIS
        scaler = MinMaxScaler()
        normalized = pd.DataFrame(scaler.fit_transform(sample_df[indicators]), columns=indicators)
        for col in indicators:
            if not benefit_criteria[col]:
                normalized[col] = 1 - normalized[col]

        weighted = normalized * weights
        ideal_best = weighted.max()
        ideal_worst = weighted.min()
        distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
        sample_df["TOPSIS_Score"] = distance_worst / (distance_best + distance_worst)
        sample_df["Rank"] = sample_df["TOPSIS_Score"].rank(ascending=False)

        top_sample = sample_df.sort_values("TOPSIS_Score", ascending=False)[[
            "Name_of_Establishment", "TOPSIS_Score", "Rank"
        ]].reset_index(drop=True)

        st.dataframe(top_sample.head(10), use_container_width=True)

        # Bar chart instead of scatter to avoid label overlap
        fig = px.bar(
            top_sample.head(10).sort_values("TOPSIS_Score"),
            x="TOPSIS_Score", y="Name_of_Establishment",
            orientation="h", color="TOPSIS_Score",
            color_continuous_scale="Blues",
            title="🔍 Sample: TOPSIS Score of Top 10 Schools"
        )
        fig.update_layout(xaxis_title="TOPSIS Score", yaxis_title="School Name")
        st.plotly_chart(fig, use_container_width=True)

    # --- File Upload Option ---
    st.markdown("---")
    st.markdown("### 📂 Upload Your Own Dataset to Run TOPSIS")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.replace(" ", "_")

        for col in ["Aprobacion_(Passrate)", "Reprobacion_(Fail_rate)", "Descercion_(Dropoutrate)"]:
            if df[col].dtype == object:
                df[col] = df[col].str.replace('%', '').astype(float) / 100

        normalized = pd.DataFrame(scaler.fit_transform(df[indicators]), columns=indicators)
        for col in indicators:
            if not benefit_criteria[col]:
                normalized[col] = 1 - normalized[col]

        weighted = normalized * weights
        ideal_best = weighted.max()
        ideal_worst = weighted.min()
        distance_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
        distance_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
        df["TOPSIS_Score"] = distance_worst / (distance_best + distance_worst)
        df["Rank"] = df["TOPSIS_Score"].rank(ascending=False)

        top_user = df.sort_values("TOPSIS_Score", ascending=False)[[
            "Name_of_Establishment", "TOPSIS_Score", "Rank"
        ]].reset_index(drop=True)

        st.success("✅ TOPSIS Completed for Uploaded File!")
        st.dataframe(top_user.head(10), use_container_width=True)

        fig2 = px.bar(
            top_user.head(10).sort_values("TOPSIS_Score"),
            x="TOPSIS_Score", y="Name_of_Establishment",
            orientation="h", color="TOPSIS_Score",
            color_continuous_scale="Viridis",
            title="🏫 Uploaded: TOPSIS Score of Top 10 Schools"
        )
        fig2.update_layout(xaxis_title="TOPSIS Score", yaxis_title="School Name")
        st.plotly_chart(fig2, use_container_width=True)

# ------------------ Inputs for Budget Sections ------------------
if section in ["2. Budget Allocation", "3. Funding Simulation"]:
    total_budget = st.sidebar.number_input("💰 Total 10-Year Budget (USD)", value=1_219_399.66, step=10000.0,
                                           format="%.2f")
    num_students = st.sidebar.number_input("👩‍🏫 Number of Students in Concession Program", min_value=0, step=1)
    st.sidebar.subheader("📌 Main Category Weights")

    raw_weights = {
        cat: st.sidebar.slider(f"{cat} Weight (%)", 0, 100, int(100 * sum(sub["weight"] for sub in items.values())),
                               key=f"main_{cat}")
        for cat, items in category_data.items()
    }

    total_main_weight = sum(raw_weights.values())
    normalized_main_weights = {k: v / total_main_weight for k, v in raw_weights.items()}
    allocated_budget = {k: total_budget * normalized_main_weights[k] for k in category_data}

    st.sidebar.subheader("⚙️ Edit Sub-Items (Weight & Cost)")
    for cat, items in category_data.items():
        with st.sidebar.expander(f"🔧 {cat}"):
            for item_name in items:
                item = items[item_name]
                item["cost"] = st.number_input(f"{item_name} - Cost", min_value=0.0, value=float(item["cost"]),
                                               step=100.0, format="%.2f")
                weight_percent = st.slider(f"{item_name} - Weight (%)", 0.0, 100.0,
                                           value=round(100 * item["weight"], 2), step=1.0)
                item["weight"] = weight_percent / 100.0


# ------------------ Optimization Function ------------------
def run_optimization(data, budget, num_students):
    model = LpProblem("Optimize", LpMaximize)
    safe_keys = {k: k.replace(" ", "_").replace("(", "").replace(")", "") for k in data}
    x = {safe_keys[k]: LpVariable(safe_keys[k], lowBound=0, cat=LpInteger) for k in data}

    model += lpSum(x[safe_keys[k]] * data[k]["weight"] for k in data)
    model += lpSum(x[safe_keys[k]] * data[k]["cost"] for k in data) <= budget

    bt = st = admin = mgr = None

    if "Primary Classrooms" in data:
        pc = safe_keys["Primary Classrooms"]
        min_classrooms = int(np.ceil(num_students / 40))
        model += x[pc] >= min_classrooms
        model += x[pc] <= min_classrooms + 10

    if "TIM Classroom" in data and "Primary Classrooms" in data:
        model += x[safe_keys["TIM Classroom"]] >= 0.5 * x[safe_keys["Primary Classrooms"]]

    if all(k in data for k in ["Base Teacher", "Specialized Teacher", "Primary Classrooms"]):
        bt = safe_keys["Base Teacher"]
        st = safe_keys["Specialized Teacher"]
        pc = safe_keys["Primary Classrooms"]
        model += x[bt] >= 10
        model += x[bt] <= x[pc]
        model += x[st] >= 10
        model += x[st] < x[pc]
        model += x[bt] + x[st] >= x[pc]
        model += x[bt] + x[st] <= x[pc] + 10
        model += x[st] >= 0.5 * x[bt]

    if "Administrator" in data:
        admin = safe_keys["Administrator"]
        model += x[admin] >= 5
        model += x[admin] <= 10

    if "Manager" in data:
        mgr = safe_keys["Manager"]
        model += x[mgr] >= 3
        model += x[mgr] <= 10

    if all(v is not None for v in [bt, st, admin, mgr]):
        model += x[mgr] + x[admin] <= x[bt] + x[st] - 1

    for k in ["Digital Pedagogy", "STEM/STEAM Instruction", "School Leadership", "Inclusion & SEL"]:
        if k in data:
            key = safe_keys[k]
            model += x[key] >= 40
            model += x[key] <= 199

    if "Kitchen & Dining" in data:
        model += x[safe_keys["Kitchen & Dining"]] <= num_students / 2

    if "Storage & Lockers" in data:
        model += x[safe_keys["Storage & Lockers"]] >= num_students / 3
        model += x[safe_keys["Storage & Lockers"]] <= num_students / 2

    if "Library" in data and "Primary Classrooms" in data:
        model += x[safe_keys["Library"]] >= 5
        model += x[safe_keys["Library"]] <= x[safe_keys["Primary Classrooms"]]

    for room in [
        "Science Lab (Primary)", "Library", "Bilingualism Room", "Reading Room",
        "Workstations", "Science Lab Workstations", "Shared Spaces"
    ]:
        if room in data:
            rk = safe_keys[room]
            model += x[rk] >= 5
            model += x[rk] <= 10

    for k, (lb, ub) in {
        "Educational Software & Licenses": (0, 400),
        "Network Infrastructure": (20, 400),
        "Audio/Visual Tools": (20, 400),
        "Cybersecurity": (20, 400),
        "IT Maintenance & Support": (20, 400)
    }.items():
        if k in data:
            model += x[safe_keys[k]] >= lb
            model += x[safe_keys[k]] <= ub

    model.solve()
    result = {k: max(0, int(round(x[safe_keys[k]].varValue or 0))) for k in data}
    total_used_cost = sum(result[k] * data[k]["cost"] for k in data)
    return result, total_used_cost


# ------------------ Allocation Renderer ------------------
def render_allocation(category_data, allocated_budget, num_students, simulation=False, num_sim=100):
    for category, items in category_data.items():
        st.subheader(f"📦 {category}")
        budget = allocated_budget[category]
        costs = []

        for i in range(num_sim if simulation else 1):
            perturbed = {
                k: {
                    "cost": v["cost"] * np.random.uniform(0.9, 1.1) if simulation else v["cost"],
                    "weight": v["weight"]
                } for k, v in items.items()
            }
            result, used_cost = run_optimization(perturbed, budget, num_students)
            costs.append(used_cost)
            if not simulation:
                df = pd.DataFrame(result.items(), columns=["Item", "Units"])
                df["Cost per Unit"] = df["Item"].map({k: v["cost"] for k, v in items.items()})
                df["Weight"] = df["Item"].map({k: v["weight"] for k, v in items.items()})
                df["Total Cost"] = df["Units"] * df["Cost per Unit"]
                st.markdown(f"**Allocated Budget:** ${budget:,.2f}")
                st.markdown(f"**Used Cost:** ${used_cost:,.2f}")
                st.dataframe(df, use_container_width=True)
                fig = px.bar(df, x="Item", y="Units", text="Units", title=f"{category} - Allocated Units")
                st.plotly_chart(fig, use_container_width=True)

        if simulation:
            st.metric("Mean Cost", f"${np.mean(costs):,.2f}")
            st.metric("Std Dev", f"${np.std(costs):,.2f}")
            fig = px.histogram(costs, nbins=30, title=f"{category} - Cost Distribution")
            st.plotly_chart(fig, use_container_width=True)


# ------------------ Render Budget Sections ------------------
if section == "2. Budget Allocation":
    st.header("📊 Budget Allocation")
    render_allocation(category_data, allocated_budget, num_students)

elif section == "3. Funding Simulation":
    st.header("🌾 Monte Carlo Funding Simulation")
    num_simulations = st.number_input(
        "Number of Simulations",
        min_value=100,
        max_value=5000,
        value=100,  # ✅ Set default to 100
        step=100
    )
    render_allocation(category_data, allocated_budget, num_students, simulation=True, num_sim=num_simulations)