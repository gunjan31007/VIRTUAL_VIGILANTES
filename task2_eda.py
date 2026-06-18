# Task II - Exploratory Data Analysis (EDA) on a Dataset
# Virtual Vigilantes Data Science Internship
# Dataset: Titanic (classic beginner-friendly dataset)
# Libraries: Pandas, Matplotlib, Seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. CREATE / LOAD DATASET
#    Using a built-in Titanic-style dataset
#    (manually constructed for offline use)
# ─────────────────────────────────────────

# We recreate a realistic Titanic-like dataset
np.random.seed(42)
n = 200

data = {
    'PassengerId': range(1, n+1),
    'Survived':    np.random.choice([0, 1], n, p=[0.61, 0.39]),
    'Pclass':      np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
    'Sex':         np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
    'Age':         np.round(np.random.normal(29.7, 14.5, n).clip(1, 80), 1),
    'SibSp':       np.random.choice([0,1,2,3], n, p=[0.68,0.23,0.06,0.03]),
    'Parch':       np.random.choice([0,1,2,3], n, p=[0.76,0.13,0.09,0.02]),
    'Fare':        np.round(np.random.exponential(32, n).clip(3, 500), 2),
    'Embarked':    np.random.choice(['S','C','Q'], n, p=[0.72,0.19,0.09]),
}

df = pd.DataFrame(data)

# Make survival more realistic: women & 1st class survive more
mask_female   = df['Sex'] == 'female'
mask_1st      = df['Pclass'] == 1
df.loc[mask_female, 'Survived'] = np.random.choice([0,1],
    mask_female.sum(), p=[0.26, 0.74])
df.loc[mask_1st,    'Survived'] = np.random.choice([0,1],
    mask_1st.sum(), p=[0.37, 0.63])

# Introduce a few missing values (as in real data)
missing_idx = np.random.choice(df.index, 20, replace=False)
df.loc[missing_idx, 'Age'] = np.nan

print("=" * 55)
print("   TASK II — EXPLORATORY DATA ANALYSIS (EDA)")
print("   Dataset: Titanic Passenger Data")
print("   Virtual Vigilantes Data Science Internship")
print("=" * 55)

# ─────────────────────────────────────────
# 2. BASIC OVERVIEW
# ─────────────────────────────────────────
print("\n[1] DATASET SHAPE:", df.shape)
print("\n[2] FIRST 5 ROWS:")
print(df.head().to_string(index=False))

print("\n[3] DATA TYPES & NULL VALUES:")
info = pd.DataFrame({
    'dtype':    df.dtypes,
    'non_null': df.count(),
    'nulls':    df.isnull().sum(),
    'null_%':   (df.isnull().sum() / len(df) * 100).round(1)
})
print(info.to_string())

# ─────────────────────────────────────────
# 3. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────
print("\n[4] DESCRIPTIVE STATISTICS (Numerical Columns):")
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────
# 4. SURVIVAL ANALYSIS
# ─────────────────────────────────────────
print("\n[5] SURVIVAL RATE:")
print(f"  Overall survival rate : {df['Survived'].mean()*100:.1f}%")
print(f"  Survived              : {df['Survived'].sum()}")
print(f"  Did not survive       : {(df['Survived']==0).sum()}")

print("\n  By Gender:")
print(df.groupby('Sex')['Survived'].mean().mul(100).round(1)
        .rename('Survival %').to_string())

print("\n  By Passenger Class:")
print(df.groupby('Pclass')['Survived'].mean().mul(100).round(1)
        .rename('Survival %').to_string())

# ─────────────────────────────────────────
# 5. CORRELATIONS
# ─────────────────────────────────────────
print("\n[6] CORRELATION WITH SURVIVAL:")
num_df = df[['Survived','Pclass','Age','SibSp','Parch','Fare']].dropna()
corr = num_df.corr()['Survived'].drop('Survived').sort_values()
print(corr.round(3).to_string())

# ─────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────
print("\n[7] GENERATING VISUALIZATIONS...")

sns.set_style('whitegrid')
palette = {'Survived': '#2ecc71', 'Did Not Survive': '#e74c3c'}

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Task II — Exploratory Data Analysis: Titanic Dataset\n"
             "Virtual Vigilantes Data Science Internship",
             fontsize=15, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

# ── Plot 1: Survival Count ──
ax1 = fig.add_subplot(gs[0, 0])
counts = df['Survived'].value_counts()
ax1.bar(['Did Not Survive', 'Survived'], counts.values,
        color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.5)
for i, v in enumerate(counts.values):
    ax1.text(i, v + 1, str(v), ha='center', fontweight='bold')
ax1.set_title("Survival Count", fontweight='bold')
ax1.set_ylabel("Number of Passengers")

# ── Plot 2: Survival by Gender ──
ax2 = fig.add_subplot(gs[0, 1])
gender_surv = df.groupby(['Sex', 'Survived']).size().unstack(fill_value=0)
gender_surv.columns = ['Did Not Survive', 'Survived']
gender_surv.plot(kind='bar', ax=ax2,
                 color=['#e74c3c', '#2ecc71'],
                 edgecolor='white', rot=0)
ax2.set_title("Survival by Gender", fontweight='bold')
ax2.set_ylabel("Count")
ax2.legend(fontsize=8)

# ── Plot 3: Survival by Pclass ──
ax3 = fig.add_subplot(gs[0, 2])
class_surv = df.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
class_surv.columns = ['Did Not Survive', 'Survived']
class_surv.plot(kind='bar', ax=ax3,
                color=['#e74c3c', '#2ecc71'],
                edgecolor='white', rot=0)
ax3.set_title("Survival by Passenger Class", fontweight='bold')
ax3.set_ylabel("Count")
ax3.set_xlabel("Class")
ax3.legend(fontsize=8)

# ── Plot 4: Age Distribution ──
ax4 = fig.add_subplot(gs[1, 0])
df_clean = df.dropna(subset=['Age'])
ax4.hist(df_clean[df_clean['Survived']==0]['Age'], bins=20,
         alpha=0.7, color='#e74c3c', label='Did Not Survive')
ax4.hist(df_clean[df_clean['Survived']==1]['Age'], bins=20,
         alpha=0.7, color='#2ecc71', label='Survived')
ax4.set_title("Age Distribution by Survival", fontweight='bold')
ax4.set_xlabel("Age")
ax4.set_ylabel("Count")
ax4.legend(fontsize=8)

# ── Plot 5: Fare Distribution ──
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df[df['Survived']==0]['Fare'], bins=25,
         alpha=0.7, color='#e74c3c', label='Did Not Survive')
ax5.hist(df[df['Survived']==1]['Fare'], bins=25,
         alpha=0.7, color='#2ecc71', label='Survived')
ax5.set_title("Fare Distribution by Survival", fontweight='bold')
ax5.set_xlabel("Fare (£)")
ax5.set_ylabel("Count")
ax5.legend(fontsize=8)

# ── Plot 6: Survival Rate by Class & Gender ──
ax6 = fig.add_subplot(gs[1, 2])
pivot = df.groupby(['Pclass', 'Sex'])['Survived'].mean().mul(100).unstack()
pivot.plot(kind='bar', ax=ax6, color=['#3498db', '#e91e8c'],
           edgecolor='white', rot=0)
ax6.set_title("Survival Rate (%) by Class & Gender", fontweight='bold')
ax6.set_ylabel("Survival Rate (%)")
ax6.set_xlabel("Passenger Class")
ax6.legend(title='Gender', fontsize=8)
ax6.set_ylim(0, 110)

# ── Plot 7: Correlation Heatmap ──
ax7 = fig.add_subplot(gs[2, 0:2])
corr_matrix = num_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, ax=ax7, mask=mask,
            linewidths=0.5, annot_kws={'size': 9})
ax7.set_title("Correlation Heatmap", fontweight='bold')

# ── Plot 8: Embarkation vs Survival ──
ax8 = fig.add_subplot(gs[2, 2])
emb_labels = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
df['Embarked_label'] = df['Embarked'].map(emb_labels)
emb_surv = df.groupby('Embarked_label')['Survived'].mean().mul(100).sort_values()
bars = ax8.barh(emb_surv.index, emb_surv.values,
                color='steelblue', edgecolor='white')
for bar, val in zip(bars, emb_surv.values):
    ax8.text(val + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
ax8.set_xlim(0, 100)
ax8.set_title("Survival Rate by Port of Embarkation", fontweight='bold')
ax8.set_xlabel("Survival Rate (%)")

plt.savefig('/mnt/user-data/outputs/task2_eda_output.png',
            dpi=150, bbox_inches='tight')
print("Saved: task2_eda_output.png")
plt.close()

print("\n[DONE] Task II complete.")
