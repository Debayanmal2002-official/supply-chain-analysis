import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import urllib.request
import json
from pathlib import Path
from src.ui_components import kpi_card, apply_custom_css
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



# LOADING DATA
@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent
    file_path = BASE_DIR / "data" / "APL_Logistics_Clean.parquet"
    df = pd.read_parquet(file_path)
    return df

try:
    df = load_data()
    df_og = df.copy()
except FileNotFoundError:
    BASE_DIR = Path(__file__).resolve().parent
    st.error(f"Data file not found at: {BASE_DIR / 'data' / 'APL_Logistics_Clean.parquet'}")
    df = pd.DataFrame()

# MAIN PAGE CONFIG
st.set_page_config(
    page_title="Supply Chain Profitability Analytics",
    page_icon="📦",
    layout="wide"
)

# MAIN APP CONFIG

apply_custom_css()

## -- Hero Header Elements -- ##
st.markdown("""
<div class="hero-section">
    <div class="hero-title">📦 Supply Chain Profitability Analytics</div>
    <div class="hero-subtitle">
        Interactive Dashboard for Customer, Product, and Profitability Performance Analysis in Supply Chain Operations.
    </div>
</div>
""", unsafe_allow_html=True)

## --- Sidebar Filters --- ##
st.sidebar.header("🔍 Global Filters")

# 1. Market Filter
available_markets = sorted(df['Market'].unique())
selected_markets = st.sidebar.multiselect(
    "Select Market Region",
    options=available_markets,
    default=[]
)

# Filter dataframe based on market to cascade options dynamically
filtered_df = df[df['Market'].isin(selected_markets)] if selected_markets else df

# 2. Regional Filter (Cascades based on selected markets)
available_regions = sorted(filtered_df['Order Region'].unique())
selected_regions = st.sidebar.multiselect(
    "Select Order Region",
    options=available_regions,
    default=[] # Empty means "All Regions" within those markets
)
if selected_regions:
    filtered_df = filtered_df[filtered_df['Order Region'].isin(selected_regions)]

# 3. Customer Segment Filter
available_segments = sorted(filtered_df['Customer Segment'].unique())
selected_segments = st.sidebar.multiselect(
    "Select Customer Segment",
    options=available_segments,
    default=[]
)
if selected_segments:
    filtered_df = filtered_df[filtered_df['Customer Segment'].isin(selected_segments)]

# --- ANALYTICAL TABS NAVIGATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Revenue & Profit Overview",
    "📦 Product & Category Performance",
    "👥 Customer Value Dashboard",
    "💸 Discount Impact Analyzer",
    "🌏 Global Market Benchmark (Unfiltered)"
])

# TAB 1: REVENUE & PROFIT OVERVIEW
with tab1:

    ## --- KPI Analytics --- ##
    st.subheader("Key Performance Indicators")
    if not filtered_df.empty:
        total_revenue = filtered_df['Sales'].sum()
        total_profit = filtered_df['Order Profit Per Order'].sum()

        # Avoid zero division if sales are zero
        profit_margin = (total_profit / total_revenue * 100) if total_revenue != 0 else 0

        # Customer Value Index (Profit contribution per unique customer)
        unique_customers = filtered_df['Customer Id'].nunique()
        customer_value_index = (total_profit / unique_customers) if unique_customers != 0 else 0

        # Discount Impact Ratio (Margin loss due to discounts)
        total_discounts = filtered_df['Order Item Discount'].sum()
        discount_impact_ratio = (total_discounts / total_revenue * 100) if total_revenue != 0 else 0

        # Group by Category Name to find the total sales and profit per category
        cat_summary = filtered_df.groupby('Category Name').agg({
            'Sales': 'sum',
            'Order Profit Per Order': 'sum'
        }).reset_index()
        # Calculate the margin for each individual category
        # (Avoid division by zero if a category has 0 sales)
        cat_summary['Margin %'] = np.where(
            cat_summary['Sales'] != 0,
            (cat_summary['Order Profit Per Order'] / cat_summary['Sales']) * 100,
            0
        )
        # Take the average of those category margins
        avg_category_margin = cat_summary['Margin %'].mean()

    else:
        total_revenue = total_profit = profit_margin = customer_value_index = discount_impact_ratio = 0

    # Row 1: Core Financials
    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("💰 Total Revenue", f"${total_revenue:,.2f}")
    with col2:
        kpi_card("📈 Total Profit", f"${total_profit:,.2f}")
    with col3:
        kpi_card("📊 Net Profit Margin", f"{profit_margin:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Supply Chain Efficiencies
    col4, col5, col6 = st.columns(3)
    with col4:
        kpi_card("👥 Customer Value Index", f"${customer_value_index:,.2f}")
    with col5:
        kpi_card("🏷️ Avg Category Margin", f"{avg_category_margin:.1f}%")
    with col6:
        kpi_card("📉 Discount Impact Ratio", f"{discount_impact_ratio:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ADAPTIVE THRESHOLD SETUP ---
    if selected_regions:
        variance_threshold = 1.5
    else:
        variance_threshold = 0.5
    margin_variance = abs(profit_margin - avg_category_margin)

    # --- CONDITIONAL EXECUTIVE COMMENTARY ---
    if margin_variance <= variance_threshold:
        st.info(
            f"**⚖️ Operational Symmetry Detected (Variance: {margin_variance:.2f}%)**\n\n"
            f"The macro-level Net Profit Margin ({profit_margin:.1f}%) and structural Average Category Margin "
            f"({avg_category_margin:.1f}%) remain highly aligned within the established tolerance threshold of {variance_threshold}%. "
            f"This indicates balanced performance where volume-driving product lines generate profitability at a rate consistent "
            f"with the broader catalog mix."
        )

    elif profit_margin > avg_category_margin:
        st.error(
            f"**🚀 Volume-Driven Profitability (Variance: +{margin_variance:.2f}%)**\n\n"
            f"The Net Profit Margin ({profit_margin:.1f}%) outpaces the Average Category Margin ({avg_category_margin:.1f}%) "
            f"beyond the expected {variance_threshold}% variance limit. This variation reveals that regional financial health "
            f"is heavily supported by a consolidated number of high-volume, highly profitable product segments, which mask "
            f"lower margin efficiencies across smaller, niche product categories."
        )

    else:  # profit_margin < avg_category_margin
        st.success(
            f"**🏷️ Structural Efficiency / Niche Optimization Opportunity (Variance: -{margin_variance:.2f}%)**\n\n"
            f"The Average Category Margin ({avg_category_margin:.1f}%) exceeds the Net Profit Margin ({profit_margin:.1f}%) "
            f"by more than the allowed {variance_threshold}% variance limit. This uncoupling indicates the presence of highly efficient, "
            f"high-margin category configurations that are currently restricted by low transaction volumes. Expanding market demand "
            f"for these specific segments offers an optimal path to accelerating overall corporate profitability."
        )
    st.divider()

    # Row 3: Analytical Trends (Non-Temporal)
    st.subheader("Profitability & Performance Analytics")

    # 1. Define the metrics to compare against Sales
    # We calculate Margin % on the fly so we can select it as a feature
    cat_performance = filtered_df.groupby('Category Name').agg({
        'Sales': 'sum',
        'Order Profit Per Order': 'sum',
        'Order Item Quantity': 'sum'
    }).reset_index()

    cat_performance['Margin %'] = (cat_performance['Order Profit Per Order'] / cat_performance['Sales']) * 100

    # 2. Add an interactive selector
    col_select, _ = st.columns([1, 2])
    with col_select:
        column_mapping = {
            "Total Profits": "Order Profit Per Order",
            "Profit Margin for Total Revenue": "Margin %",
            "Number of Item Sold": "Order Item Quantity"
        }
        y_axis_feat = st.selectbox(
            "Select Performance Metric:",
            options=list(column_mapping.keys()),
            index=0
        )

    selected_key = y_axis_feat
    actual_column = column_mapping[selected_key]

    fig_efficiency = px.scatter(
        cat_performance,
        x="Sales",
        y=actual_column,  # Use the mapped column name
        size="Order Item Quantity",
        color=actual_column,  # Use the mapped column name
        color_continuous_scale="RdYlGn",
        hover_name="Category Name",
        title=f"Category Efficiency: Total Sales vs. {selected_key}",
        labels={
            "Sales": "Total Sales ($)",
            "Order Profit Per Order": "Total Profits ($)",
            "Margin %": "Net Profit Margin (%)",
            "Order Item Quantity": "Total Units Sold"
        },
        template="plotly_dark"
    )

    fig_efficiency.update_layout(
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font_color="#F8FAFC",
        xaxis_title="Total Sales ($)",
        yaxis_title=selected_key,  # Use the human-readable label
        xaxis=dict(showgrid=True, gridcolor="#334155"),
        yaxis=dict(showgrid=True, gridcolor="#334155")
    )

    st.plotly_chart(fig_efficiency, width='stretch')

    # --- Dynamic Insights Generation ---
    with st.expander("🔍 View Insight", expanded=True):
        # Get the top performing category based on the selected metric
        top_cat = cat_performance.loc[cat_performance[actual_column].idxmax()]

        # Identify the "Volume Trap" (High sales, but low profit/margin)
        # Logic: High sales (> mean) but low metric (< mean)
        volume_traps = cat_performance[
            (cat_performance['Sales'] > cat_performance['Sales'].mean()) &
            (cat_performance[actual_column] < cat_performance[actual_column].mean())
            ]

        st.markdown(f"**Current Trend Analysis:**")
        st.write(f"The top-performing category for **{selected_key}** is **{top_cat['Category Name']}**.")

        if not volume_traps.empty:
            trap_names = ", ".join(volume_traps['Category Name'].tolist())
            st.warning(
                f"**Potential Volume Trap Detected:** The following categories are driving high revenue but underperforming in **{selected_key}**: *{trap_names}*. Consider reviewing their cost structure.")
        else:
            st.success("No significant volume traps detected! Your profitability is scaling well with revenue.")

with tab2:
    st.subheader("Category & Product Deep Dive")

    col_1, col_2 = st.columns(2)

    with col_1:
        class_option = st.selectbox(
            "Select Feature Type:",
            options=["Category Name","Product Name"],
            index=0
        )
    with col_2:
        metric_option = st.selectbox(
            "Select Feature Type:",
            options=["Sales","Order Profit Per Order"],
            index=0
        )

    # 1. Prepare Pareto Data
    pareto_df = filtered_df.groupby(class_option)[metric_option].sum().reset_index()
    pareto_df = pareto_df.sort_values(by=metric_option, ascending=False)

    # 2. Calculate Cumulative Percentages
    pareto_df['Cumulative Revenue'] = pareto_df[metric_option].cumsum()
    total_rev = pareto_df[metric_option].sum()
    pareto_df['Cumulative %'] = (pareto_df['Cumulative Revenue'] / total_rev) * 100


    # 3. Categorize Products
    def categorize_pareto(pct):
        if pct <= 80: return "Short-Tail (Top 80%)"
        return "Long-tail (Remaining 20%)"


    pareto_df['Classification'] = pareto_df['Cumulative %'].apply(categorize_pareto)

    # 4. Create the Treemap
    pareto_df['Abs_Value'] = pareto_df[metric_option].abs()

    fig_tree = px.treemap(
        pareto_df,
        path=['Classification', class_option],
        values='Abs_Value',
        color=metric_option,  # Now color shows the actual gain/loss
        color_continuous_scale="RdYlGn",
        # Use a diverging scale if you have both positive and negative profit
        color_continuous_midpoint=0
    )

    # 2. Update hovertemplate to show the REAL value, not the absolute size
    fig_tree.update_traces(
        textinfo="label+value",
        texttemplate="<b>%{label}</b><br>$%{customdata:,.0f}",
        customdata=pareto_df[metric_option],  # Pass the real value here
        hovertemplate=f"<b>%{{label}}</b><br>{metric_option}: $%{{customdata:,.2f}}<br>Contribution: %{{percentParent:.1%}}",
        marker_line_width=2,
        marker_line_color="#0F172A"
    )

    fig_tree.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        font_color="#F8FAFC",
        height=600
    )

    st.plotly_chart(fig_tree, width='stretch')

    st.markdown("---")

    # --- Visualizing Concentration ---
    # Calculate metrics for the summary
    anchors_count = len(pareto_df[pareto_df['Classification'] == "Short-Tail (Top 80%)"])
    total_products = len(pareto_df)
    anchor_ratio = (anchors_count / total_products) * 100

    col1, col2 = st.columns(2)

    col1.metric("Short-Tail Anchors", f"{anchors_count} Products")
    col2.metric("Concentration Ratio", f"{anchor_ratio:.1f}%")

    st.markdown("---")

    # Risk Assessment
    if anchor_ratio < 15:
        st.error("⚠️ **High Risk**: Revenue is too concentrated in very few products.")
    elif anchor_ratio > 25:
        st.success("✅ **Balanced**: Revenue is healthy and spread across the menu.")
    else:
        st.warning("⚖️ **Moderate**: Standard 80/20 distribution.")

    st.markdown("---")

    # 4. Long-tail Detail
    with st.expander("🔍 Identify Long-tail Products (Candidates for Re-evaluation)"):
        long_tail = pareto_df[pareto_df['Classification'] == "Long-tail (Remaining 20%)"]
        st.write(f"These {len(long_tail)} products contribute to only 20% of your total revenue.")

        # Define columns based on your condition
        if class_option == "Category Name":
            cols_to_show = [class_option, metric_option, 'Cumulative %']
        else:
            # Assuming 'Category Name' exists in your filtered_df,
            # we merge it back to show it alongside Product Name
            long_tail = long_tail.merge(
                filtered_df[['Product Name', 'Category Name']].drop_duplicates(),
                left_on='Product Name',
                right_on='Product Name',
                how='left'
            )
            cols_to_show = [class_option, 'Category Name', metric_option, 'Cumulative %']

        st.dataframe(long_tail[cols_to_show], width='stretch')

    st.markdown("---")

    st.subheader("Category Performance Matrix")

    # Let the user choose the granularity of the heatmap
    axis_choice = st.radio(
        "Analyze Matrix by:",
        ["Market", "Order Region"],
        horizontal=True
    )

    # Build the pivot based on the user's choice
    matrix = filtered_df.groupby([axis_choice, 'Category Name'])['Order Profit Per Order'].sum() / \
             filtered_df.groupby([axis_choice, 'Category Name'])['Sales'].sum() * 100
    matrix = matrix.reset_index(name='Margin %').pivot(index=axis_choice, columns='Category Name', values='Margin %')

    # Plot the matrix
    fig_heatmap = px.imshow(
        matrix,
        labels=dict(x="Category Name", y=axis_choice, color="Margin %"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        template="plotly_dark",
        zmin=-20,
        zmax=20
    )

    # 4. Refine layout
    fig_heatmap.update_layout(
        title="Category Margin Matrix",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        coloraxis_colorbar=dict(title="Margin %"),
        xaxis={'side': 'bottom'}  # Ensure x-axis labels are at the bottom
    )

    st.plotly_chart(fig_heatmap, width='stretch')

with tab3:
    st.subheader("Customer Value Dashboard")

    # 0. Toggle between Hard Logic and ML
    analysis_type = st.radio(
        "Analysis Methodology:",
        ["Hard Logic (Rule-Based)", "ML-Powered (K-Means Clustering)"],
        horizontal=True,
        key="analysis_method_selector"
    )

    # 1. Aggregate Customer Data (Shared step)
    cust_performance = filtered_df.groupby('Customer Id').agg({
        'Sales': 'sum',
        'Order Profit Per Order': 'sum',
        'Order Item Quantity': 'sum'
    }).reset_index()

    # 2. APPLY SEGMENTATION LOGIC
    if analysis_type == "Hard Logic (Rule-Based)":
        q_sales = cust_performance['Sales'].quantile(0.8)


        def segment_customer(row):
            if row['Sales'] >= q_sales and row['Order Profit Per Order'] > 0:
                return "High-Value (Whales)"
            elif row['Order Profit Per Order'] <= 0:
                return "Loss-Making (Maintenance Drain)"
            else:
                return "Standard Growth"


        cust_performance['Segment'] = cust_performance.apply(segment_customer, axis=1)
        title_suffix = "(Rule-Based)"

    else:  # ML-Powered Logic
        # Feature Engineering
        cust_performance['Is_Profitable'] = (cust_performance['Order Profit Per Order'] > 0).astype(int)
        scaler = StandardScaler()
        features = ['Sales', 'Order Profit Per Order', 'Order Item Quantity', 'Is_Profitable']
        scaled_data = scaler.fit_transform(cust_performance[features])

        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cust_performance['Cluster_ID'] = kmeans.fit_predict(scaled_data)


        # Mapping
        def get_segment_name(row):
            if row['Order Profit Per Order'] <= 0:
                return "Loss-Making (Maintenance Drain)"
            elif row['Sales'] > cust_performance['Sales'].median():
                return "High-Value (Whales)"
            else:
                return "Standard Growth"


        cust_performance['Segment'] = cust_performance.apply(get_segment_name, axis=1)
        title_suffix = "(ML-Powered)"

    # 3. View Mode Toggle
    view_mode = st.radio(
        "Choose Visualization View:",
        ["Executive Summary (Segmented)", "Granular Analysis (All Customers)"],
        horizontal=True,
        key="view_mode_selector"
    )

    # 4. Preparing data for plotting
    if view_mode == "Executive Summary (Segmented)":
        data_to_plot = cust_performance.groupby('Segment').agg({
            'Sales': 'sum', 'Order Profit Per Order': 'sum', 'Customer Id': 'count'
        }).reset_index()
        size_col = 'Customer Id'
    else:
        data_to_plot = cust_performance
        size_col = 'Order Item Quantity'

    # 5. Plotting
    fig_cust = px.scatter(
        data_to_plot,
        x="Sales",
        y="Order Profit Per Order",
        size=size_col,
        color="Segment",
        size_max=70 if view_mode == "Executive Summary (Segmented)" else 15,
        color_discrete_map={
            "High-Value (Whales)": "#22c55e",
            "Standard Growth": "#3b82f6",
            "Loss-Making (Maintenance Drain)": "#ef4444"
        },
        title=f"Customer Performance {title_suffix}",
        template="plotly_dark"
    )

    fig_cust.update_layout(plot_bgcolor="#0F172A", paper_bgcolor="#0F172A", font_color="#F8FAFC")
    st.plotly_chart(fig_cust, width='stretch')

    # 6. Detailed Action List
    st.markdown("---")
    st.subheader("Detailed Customer Action List")
    all_segs = sorted(cust_performance['Segment'].unique())
    selected_segments = st.multiselect("Filter by Segment:", options=all_segs, default=all_segs, key="seg_filter_final")

    filtered_list = cust_performance[cust_performance['Segment'].isin(selected_segments)]
    st.dataframe(filtered_list.sort_values(by='Sales', ascending=False), width='stretch', hide_index=True)

with tab4:
    st.subheader("Discount Impact Diagnostics")

    # 1. Calculate Volume-Weighted Profit Ratio
    discount_analysis = filtered_df.groupby('Order Item Discount Rate').agg({
        'Order Profit Per Order': 'sum',
        'Sales': 'sum',
        'Order Item Quantity': 'sum'
    }).reset_index()

    # Calculate the weighted ratio: Total Profit / Total Sales
    discount_analysis['Weighted Profit Ratio'] = (
            discount_analysis['Order Profit Per Order'] / discount_analysis['Sales']
    )

    # 2. Filter out "Noise": Only show discount rates with significant volume
    # Adjust '20' if you want to be more or less strict
    min_volume = 20
    significant_analysis = discount_analysis[discount_analysis['Order Item Quantity'] >= min_volume]

    efficiency_df = filtered_df.groupby('Order Item Discount Rate').agg({
        'Order Profit Per Order': 'sum',
        'Order Item Quantity': 'sum'
    }).reset_index()

    # 1. Create the dual-axis figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    bar_colors = ['#10B981' if val >= 0 else '#EF4444' for val in efficiency_df['Order Profit Per Order']]

    # 2. Add Bar trace for Absolute Profit (Volume/Growth)
    fig.add_trace(
        go.Bar(
            x=efficiency_df['Order Item Discount Rate'],
            y=efficiency_df['Order Profit Per Order'],
            name="Total Profit",
            marker_color=bar_colors,  # Using the dynamic color list here
            opacity=0.6
        ),
        secondary_y=False,
    )

    # 3. Add Line trace for Profit Ratio (Margin/Efficiency)
    fig.add_trace(
        go.Scatter(
            x=significant_analysis['Order Item Discount Rate'],
            y=significant_analysis['Weighted Profit Ratio'],
            name="Profit Margin Ratio",
            line=dict(color='#3B82F6', width=3),
            mode='lines+markers'
        ),
        secondary_y=True,
    )

    # 4. Add the Break-Even line (on the secondary y-axis)
    fig.add_hline(y=0, line_dash="dash", line_color="red", secondary_y=True)

    # 5. Layout and Styling
    fig.update_layout(
        title="Discount Diagnostic: Efficiency vs. Total Profitability",
        template="plotly_dark",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font_color="#F8FAFC",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(title_text="Discount Rate")
    fig.update_yaxes(title_text="Total Profit ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit Ratio", secondary_y=True)

    st.plotly_chart(fig, width='stretch')

    # 4. Diagnostic Summary
    erosion_points = significant_analysis[significant_analysis['Weighted Profit Ratio'] < 0]

    if not erosion_points.empty:
        critical_rate = erosion_points['Order Item Discount Rate'].min()
        st.error(
            f"⚠️ **Critical Threshold Detected**: Profit margins turn negative starting at a **{critical_rate * 100:.0f}%** discount rate (based on significant transaction volume).")
    else:
        st.success("✅ **Healthy Margins**: No major discount levels are driving negative profit at current volumes.")

    # 3. Decision Logic
    max_profit_rate = efficiency_df.loc[efficiency_df['Order Profit Per Order'].idxmax(), 'Order Item Discount Rate']

    st.markdown(
        f"**Insight:** The discount rate that currently generates the maximum absolute profit is **{max_profit_rate * 100:.0f}%**.")

    st.markdown("---")
    st.subheader("Profitability Impact Summary")
    st.caption("💡 **Tip:** Use side bar drill down filter here for more accurate market and customer specific predictions.")

    # Calculate Theoretical Revenue (Price without discount)
    # Price = Sales / (1 - Discount_Rate)
    filtered_df['Theoretical Sales'] = filtered_df['Sales'] / (1 - filtered_df['Order Item Discount Rate'])

    # Assuming Profit = (Actual Sales * Profit Ratio)
    # We can estimate "Theoretical Profit" by assuming costs stay constant
    # Theoretical Profit = Sales_without_discount - Cost_of_Goods
    # Profit_with_Discount = Sales_with_discount - Cost_of_Goods
    # COGS = Sales_with_Discount * (1 - Profit_Ratio)

    filtered_df['COGS'] = filtered_df['Sales'] * (1 - filtered_df['Order Item Profit Ratio'])
    filtered_df['Theoretical Profit'] = filtered_df['Theoretical Sales'] - filtered_df['COGS']

    # Aggregate for summary
    total_actual_profit = filtered_df['Order Profit Per Order'].sum()
    total_theoretical_profit = filtered_df['Theoretical Profit'].sum()
    margin_loss = total_theoretical_profit - total_actual_profit

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual Profit", f"${total_actual_profit:,.0f}")
    col2.metric("Theoretical Profit (No Disc.)", f"${total_theoretical_profit:,.0f}")
    col3.metric("Profit Leakage", f"${margin_loss:,.0f}", delta_color="inverse")

    st.markdown("---")

    st.subheader("Discount Impact Simulator (Random Forest Model)")

    # 1. Feature Selection & Preprocessing
    cat_features = ['Shipping Mode', 'Payment Type']
    num_features = ['Order Item Discount Rate', 'Order Item Product Price']
    features = num_features + cat_features

    # Convert text categories to dummy/one-hot variables
    X = pd.get_dummies(filtered_df[features], columns=cat_features)
    y = filtered_df['Order Item Quantity']

    # 2. Train the Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 3. Simulator UI
    sim_discount = st.slider(
        "Hypothetical Discount Rate (%)",
        min_value=0.0, max_value=50.0, value=10.0, step=0.1,
        key="sim_discount_rf"
    ) / 100

    # 4. Predict
    X_sim = X.copy()
    X_sim['Order Item Discount Rate'] = sim_discount
    predicted_volume = model.predict(X_sim).sum()

    # 5. Metrics Calculation
    avg_price = filtered_df['Sales'].sum() / filtered_df['Order Item Quantity'].sum()
    projected_sales = predicted_volume * (avg_price * (1 - sim_discount))
    # Assume simple margin calculation for simulation
    projected_profit = projected_sales * 0.2

    historical_revenue = filtered_df['Sales'].sum()
    revenue_delta = projected_sales - historical_revenue

    col1, col2 = st.columns(2)
    col1.metric(
        "Predicted Total Volume",
        f"{predicted_volume:,.0f} units"
    )

    col2.metric(
        label="Projected Revenue",
        value=f"${projected_sales:,.0f}",
        delta=f"{revenue_delta:,.0f}",
        delta_color="normal"  
    )

    # 6. Feature Importance
    st.write("---")
    st.subheader("What drives demand for this region?")
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).head(5)

    fig_importance = px.bar(
        importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Key Demand Drivers",
        template="plotly_dark"
    )
    fig_importance.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="#F8FAFC"
    )

    st.plotly_chart(fig_importance, width='stretch')

with tab5:
    st.subheader("Global Profitability Heatmap")

    # 1. Calculate Benchmarks (Global vs Filtered)
    global_avg_margin = (df['Order Profit Per Order'].sum() / df['Sales'].sum()) * 100
    filtered_avg_margin = (filtered_df['Order Profit Per Order'].sum() / filtered_df['Sales'].sum()) * 100

    # 2. Aggregate at the Country level based on ACTIVE FILTERS
    geo_df = filtered_df.groupby(['Order Country']).agg({
        'Sales': 'sum',
        'Order Profit Per Order': 'sum'
    }).reset_index()
    geo_df['Profit Margin %'] = (geo_df['Order Profit Per Order'] / geo_df['Sales']) * 100

    # 3. Create Map (using Filtered Benchmark as the center)
    fig_map = px.choropleth(
        geo_df,
        locations="Order Country",
        locationmode='country names',
        color="Profit Margin %",
        color_continuous_midpoint=filtered_avg_margin,
        color_continuous_scale=px.colors.diverging.RdYlGn,
        range_color=[-10, 25],
        template="plotly_dark",
        projection="natural earth"
    )
    fig_map.update_layout(
        plot_bgcolor="#0F172A",  # Dark slate background
        paper_bgcolor="#0F172A",  # Matches the container
        font_color="#F8FAFC",  # Off-white text for readability
        template="plotly_dark"  # The base dark theme
    )
    st.plotly_chart(fig_map, width='stretch')

    # 4. Performance vs Both Benchmarks
    st.subheader("Performance vs. Benchmarks")

    c1, c2 = st.columns(2)
    c1.metric("Global Constant Avg", f"{global_avg_margin:.1f}%")
    c2.metric("Filtered Selection Avg", f"{filtered_avg_margin:.1f}%")

    # Create a ranked bar chart
    fig_bench = px.bar(
        geo_df.sort_values('Profit Margin %', ascending=True),
        x='Profit Margin %',
        y='Order Country',
        orientation='h',
        color='Profit Margin %',
        color_continuous_scale="RdYlGn",
        range_color=[-10, 25],
        template="plotly_dark"
    )

    fig_bench.update_layout(
        plot_bgcolor="#0F172A",  # Dark slate background
        paper_bgcolor="#0F172A",  # Matches the container
        font_color="#F8FAFC",  # Off-white text for readability
        template="plotly_dark"  # The base dark theme
    )

    # Draw the TWO benchmark lines
    fig_bench.add_vline(x=global_avg_margin, line_dash="dash", line_color="white",
                        annotation_text=f"Global: {global_avg_margin:.1f}%")
    fig_bench.add_vline(x=filtered_avg_margin, line_dash="solid", line_color="#3B82F6",
                        annotation_text=f"Selection: {filtered_avg_margin:.1f}%")

    st.plotly_chart(fig_bench, width='stretch')

    st.markdown("---")
    st.subheader("Profitability Breakdown by Country")

    # 4. Display the table
    table_df = geo_df[['Order Country', 'Sales', 'Order Profit Per Order', 'Profit Margin %']].sort_values(
        by='Profit Margin %', ascending=False)

    styled_table = table_df.style.format({
        'Sales': '${:,.0f}',
        'Order Profit Per Order': '${:,.0f}',
        'Profit Margin %': '{:.1f}%'
    }).map(lambda x: 'color: red;' if x < 0 else 'color: white;', subset=['Order Profit Per Order', 'Profit Margin %'])

    st.dataframe(styled_table, width='stretch')






