# 🚢 Health Monitoring and Fault Detection in Naval Vessel Systems

> A machine learning–based framework for integrated health monitoring of naval vessel propulsion systems and IGBT power electronics, culminating in a unified **Ship Health Index (SHI)** via Fuzzy Logic.

---

## 📌 Project Overview

Naval vessels operate under extreme mechanical, thermal, and electrical stresses. This project presents an end-to-end predictive maintenance pipeline that:

- Predicts **propulsion system degradation** (decay coefficients) from operational sensor data
- Estimates the **Remaining Useful Life (RUL)** of IGBT power modules
- Fuses both subsystem health indicators into a single **Ship Health Index** using a Fuzzy Inference System (FIS)

This work was developed as part of the **23ELC212 Machine Learning** course at Amrita School of Engineering, Coimbatore.

---

## 👥 Team

| Name | Roll No. | Contribution |
|---|---|---|
| Dharineesh MV | CB.EN.U4ELC24105 | Mechanical system modelling & Fuzzy Logic (Python) |
| Harshitaa JA | CB.EN.U4ELC24112 | IGBT modelling, Mechanical system & Documentation |
| Haritha SK | CB.EN.U4ELC24151 | IGBT modelling & MATLAB Fuzzy Logic |

---

## 🗂️ Repository Structure

```
├── dataset/                    # Raw and processed datasets (UCI Naval + NASA IGBT)
├── models/                     # Trained model files and experiment outputs
├── presentation/               # Project presentation slides
└── README.md
```

---

## 🧠 Methodology

### 1. Mechanical Subsystem — Propulsion Decay Prediction

**Dataset:** UCI Naval Propulsion Plant Dataset (OpenML #41012)

**Targets:** Four decay coefficients — `Kkt`, `Khull`, `KMcompr`, `KMturb`

**Pipeline:**
- Dropped 8 redundant/duplicate columns
- Extracted statistical features from degradation signals
- Normalized data for multi-output regression
- Evaluated **12 regression algorithms** with grid-search hyperparameter tuning

**Best Model:** SVR (RBF kernel, C=10, ε=0.001)
- Test R² = **0.9897**
- RMSE = **0.004615**

---

### 2. IGBT Subsystem — RUL Prediction

**Dataset:** NASA IGBT Accelerated Aging Dataset

**Features:** 12 physical precursor features (VCE-ON, VGE, IC, thermal resistance, power loss, etc.)
- Time proxy columns excluded (r > 0.99 with RUL) to prevent leakage
- `Transient_Vce_max` identified as dominant signal (r = 0.51)
- Group-based train/test split on `Transient_Block_ID`

**Best Model:** Gradient Boosting
- Test R² = **0.9561**
- RMSE = **6.87**

---

### 3. Fuzzy Logic Health Fusion

**Inputs:**
- **WDA** (Weighted Decay Aggregate) — computed from the 4 decay coefficients:

$$WDA = 0.25 \cdot K_{kt} + 0.10 \cdot (2 - K_{hull}) + 0.35 \cdot KM_{compr} + 0.30 \cdot KM_{turb}$$

- **RUL** — IGBT Remaining Useful Life (0–100%)

**Membership Functions:**

| Variable | Poor | Average | Good |
|---|---|---|---|
| WDA | 0.930–0.955 | 0.948–0.978 | 0.972–1.000 |
| RUL | 0–35% | 20–80% | 65–100% |
| Health | 0.0–0.40 | 0.30–0.70 | 0.60–1.00 |

**Rule Base (3×3):**

| WDA \ RUL | Poor | Average | Good |
|---|---|---|---|
| **Poor** | Poor | Poor | Average |
| **Average** | Poor | Average | Average |
| **Good** | Average | Good | Good |

**Defuzzification:** Centroid method → crisp health score ∈ [0, 1]

---

## 📊 Results Summary

### Propulsion Decay — 12-Model Benchmark

| Rank | Model | Test R² | Test RMSE |
|---|---|---|---|
| 1 | **SVR (RBF)** | **0.9897** | 0.004615 |
| 2 | KNN Regressor | 0.9868 | 0.004210 |
| 3 | Gradient Boosting | 0.9791 | 0.003771 |
| 4 | Polynomial Regression | 0.9616 | 0.010740 |
| 5 | Random Forest | 0.9586 | 0.005449 |
| 6–8 | Linear / MLR / Ridge | ~0.80 | ~0.023 |

### IGBT RUL — 12-Model Benchmark

| Rank | Model | Test R² | RMSE |
|---|---|---|---|
| 1 | **Gradient Boosting** | **0.9561** | 6.87 |
| 2 | Random Forest | 0.9155 | 9.53 |
| 3 | AdaBoost | 0.9063 | 10.04 |
| 4 | Decision Tree | 0.8951 | 10.62 |
| 5 | SVR (RBF Kernel) | 0.8692 | 11.81 |

> Linear models yielded negative R² across both tasks, confirming inherently **nonlinear** degradation behavior in naval systems.

---

## 🔑 Key Findings

- **SVR** excels at smooth, continuous propulsion degradation prediction
- **Gradient Boosting** excels at nonlinear IGBT RUL estimation
- **IGBT health dominates** overall system health — even a healthy propulsion system cannot compensate for a failing IGBT
- The **Fuzzy Inference System** successfully fuses two independent ML pipelines into a single, interpretable Ship Health Index
- The complete framework enables **sensor-only, inspection-free** predictive maintenance decisions

---

## 📦 Dependencies

```bash
pip install numpy pandas scikit-learn scikit-fuzzy matplotlib seaborn
```

| Library | Purpose |
|---|---|
| `scikit-learn` | Regression models, preprocessing, evaluation |
| `scikit-fuzzy` | Fuzzy inference system (FIS) |
| `numpy` / `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization and heatmaps |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run notebooks in order:
# 1. propulsion-system.ipynb  → train SVR, get decay coefficients
# 2. final-igbt.ipynb         → train Gradient Boosting, get RUL
# 3. fuzzy_logic.ipynb        → fuse outputs into Ship Health Index
```

---

## 📚 Datasets

| Dataset | Source |
|---|---|
| UCI Naval Propulsion Plant | [OpenML #41012](https://www.openml.org/search?type=data&sort=runs&id=41012&status=active) |
| NASA IGBT Accelerated Aging | [NASA Data Portal](https://data.nasa.gov/dataset/insulated-gate-bipolar-transistor-igbt-accelerated-aging) |

---

## 📄 References

1. Cipollini et al. — *Condition-Based Maintenance of Naval Propulsion Systems with Supervised Data Analysis*, Ocean Engineering, 2018
2. Sood et al. — *ML-based Prediction of the Decay Coefficient of a Naval Propulsion System's Turbine*, ARIIA, 2024
3. Patil et al. — *Precursor Parameter Identification for IGBT Prognostics*, IEEE Transactions on Reliability, 2009
4. Dimitriou et al. — *Supervised Machine Learning Algorithms for the Analysis of Ship Engine Data*, 2024
5. Sonnenfeld et al. — *An Agile Accelerated Aging, Characterization and Scenario Simulation System for Gate Controlled Power Transistors*, IEEE, 2008
6. Prathiba et al. — *Machine Learning-Based Direct Estimation of RUL of IGBT Using Multi-Precursor Prognostics*, Vol 107, 2025

---

## 📃 License

This project was developed for academic purposes under the 23ELC212 Machine Learning course. Please cite appropriately if you build on this work.
