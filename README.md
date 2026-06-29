# 📊 Supply Chain Profitability & Margin Intelligence Framework

> *Turning supply chain complexity into actionable financial intelligence.*

## 📝 Overview
This repository hosts the documentation and analytical framework for optimizing supply chain profitability. The project shifts the analytical focus from superficial volume metrics to granular profitability diagnostics, empowering leadership with a high-performance, interactive dashboard.

## 🚀 Key Features
* **High-Performance Data Pipeline:** Leverages **Apache Parquet** columnar storage to process 280k+ records with sub-second latency.
* **Interactive Dashboard:** Built with **Streamlit** to facilitate real-time "what-if" scenario modeling.
* **Advanced Diagnostics:**
    * **Profitability Mapping:** Identifies "Volume Traps" (high revenue, low margin).
    * **Customer Value Index (CVI):** Segments "Whale" accounts vs. "Maintenance Drain" customers.
    * **Discount Impact Simulator:** Quantifies margin erosion caused by aggressive pricing strategies.
* **Privacy-First Design:** Strict PII decoupling and tokenization for secure, compliant analysis.

## 🛠 Tech Stack
* **Language:** Python
* **Dashboarding:** Streamlit
* **Data Engineering:** Pandas, Apache Parquet
* **Visualization:** Plotly (for interactive Heatmaps and TreeMaps)
* **Documentation:** LaTeX

## 📂 Project Structure
```text
.
├── main.tex                 # Full research paper/report
├── app/                     # Streamlit application source
│   ├── dashboard.py         # Main dashboard interface
│   └── data_pipeline.py     # ETL, cleaning, and Parquet conversion
├── plots/                   # Visual outputs (Heatmaps, TreeMaps)
├── data/                    # Processed & anonymized datasets
└── README.md                # You are here!
``` 

### 📈 Methodology
*   **Data Sanitization:** Rigorous auditing of transactional integrity, including removal of nulls and duplicate order records.
*   **Feature Engineering:** Calculation of specific KPIs such as `Profit Margin Percentage` and `Discount-Driven Margin Erosion`.
*   **Storage Optimization:** Columnar serialization to ensure the dashboard remains fast even as datasets scale.

### 💡 Strategic Playbook
The dashboard provides actionable insights into:
*   **Cost Structure Audits:** Targeting underperforming product categories.
*   **Surgical Pricing:** Replacing broad discounts with data-driven, margin-focused pricing.
*   **Retention Programs:** Implementing tiered outreach for top-tier "Whale" customers.

***
**Developed by:** Debayan Mal | Unified Mentor Intern | May 2026
