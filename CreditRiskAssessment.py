# Import libraries
import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, chi2_contingency
from sklearn.preprocessing import (
    PowerTransformer, FunctionTransformer,
    RobustScaler, OneHotEncoder, OrdinalEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
sns.set_theme(style="whitegrid", palette="Blues")
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "4"

########################################
# LOAD DATASETS
########################################

column_names_ger = [
    'Status_of_existing_checking_account', 'Duration_in_month', 'Credit_history', 'Purpose', 'Credit_amount',
    'Savings_account_bonds', 'Present_employment_since', 'Installment_rate_in_percentage_of_disposable_income',
    'Personal_status_and_sex', 'Other_debtors_guarantors', 'Present_residence_since', 'Property',
    'Age_in_years', 'Other_installment_plans', 'Housing', 'Number_of_existing_credits_at_this_bank',
    'Job', 'Number_of_people_being_liable_to_provide_maintenance_for', 'Telephone', 'Foreign_worker',
    'Target']
df_german = pd.read_csv("german.data", sep=' ', names=column_names_ger)
df_german['Target'] = df_german['Target'].map({1: 1, 2: 0}) # Maps Target Variable so it's the same as the Australian dataset

column_names_aus = [
    'Gender_Status', 'Age', 'Debt', 'Marital_Status', 'Bank_Customer', 'Education_Level', 'Ethnicity', 'Years_Employed',
    'Prior_Default', 'Employed', 'Credit_Score', 'Drivers_License', 'Citizen', 'Income', 'Target']
df_aus = pd.read_csv("australian.dat", sep=' ', header=None, names=column_names_aus)

#########################################
# SPLIT INTO STRATIFIED 80-20 TRAIN/TEST 
#########################################

# German Dataset
X_german = df_german.drop("Target", axis=1) # Features
y_german = df_german["Target"] # Target variable
X_train_german, X_test_german, y_train_german, y_test_german = train_test_split(
    X_german, y_german, test_size=0.2, random_state=42, stratify=y_german) # Train-Test Split

# Australian Dataset
X_aus = df_aus.drop("Target", axis=1) # Features
y_aus = df_aus["Target"] # Target variable
X_train_aus, X_test_aus, y_train_aus, y_test_aus = train_test_split(
    X_aus, y_aus, test_size=0.2, random_state=42, stratify=y_aus) # Train-Test Split

#####################################
# COLUMN SPECIFICATIONS
#####################################

# Continuous Features
continuous_german = ['Duration_in_month', 'Credit_amount', 'Age_in_years']
continuous_aus = ['Age', 'Debt', 'Income', 'Employed']

# Categorical Features
categorical_aus = [
    'Gender_Status', 'Marital_Status', 'Education_Level', 'Bank_Customer',
    'Ethnicity', 'Years_Employed', 'Prior_Default', 'Drivers_License',
    'Citizen', 'Credit_Score']
categorical_german = [
    'Status_of_existing_checking_account', 'Credit_history', 'Purpose', 
    'Savings_account_bonds', 'Present_employment_since', 
    'Installment_rate_in_percentage_of_disposable_income', 'Personal_status_and_sex',
    'Other_debtors_guarantors', 'Present_residence_since', 'Property',
    'Other_installment_plans', 'Housing', 'Number_of_existing_credits_at_this_bank',
    'Job', 'Number_of_people_being_liable_to_provide_maintenance_for', 'Telephone',
    'Foreign_worker']

#------------------------------------------------------------------------------------------------------------
################################################
# EXPLORATORY DATA ANALYSIS SECTION
################################################

# Detects outliers using IQR and plots boxplots for continuous features
def detect_and_plot_outliers(df, continuous_cols, dataset_name):

    print(f"\n{dataset_name} - Outlier Detection on {len(continuous_cols)} numeric features")
    for col in continuous_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        outliers = df[(df[col] < Q1 - 1.5*(Q3-Q1)) | (df[col] > Q3 + 1.5*(Q3-Q1))]
        
        if not outliers.empty:
            print(f" Feature '{col}': {outliers.shape[0]} outliers")
            plt.figure(figsize=(6, 2))
            sns.boxplot(x=df[col], color='skyblue', fliersize=5, flierprops=dict(marker='o', color='red', alpha=1))
            plt.title(f"{dataset_name} - {col} (Outliers)", fontsize=14)
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

# Plots distributions of continuous features and calculates Skew value
def plot_distributions_and_skew(df, continuous_cols, dataset_name, skew_threshold=1.0):
    
    print(f"\n {dataset_name} - Numerical Features: {len(continuous_cols)} continuous columns")
    skewed_features = {}
    for col in continuous_cols:
        col_skew = skew(df[col].dropna())
        skewed_features[col] = col_skew

    top_skewed_cols = list(skewed_features.keys())[:4]
    for col in top_skewed_cols:
        skew_value = skewed_features[col]
        print(f"Feature: {col} — Skew: {skew_value:.4f}")  # Print skew value
        plt.figure(figsize=(6, 4)) # Plot KDE
        sns.kdeplot(df[col], fill=True, color='#1f77b4', linewidth=2)
        plt.title(f"{dataset_name} - KDE of '{col}' (Skew = {skew_value:.2f})", fontsize=14)
        plt.xlabel(col, fontsize=14)
        plt.tight_layout()
        plt.show()

# Plots correlation heatmap showing relationship between continuous features
def plot_correlation_heatmap(df, continuous_cols, dataset_name):
    numeric_df = df[continuous_cols]
    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title(f"{dataset_name} - Correlation Heatmap (Continuous Features)", fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.tight_layout()
    plt.show()

# Plots and shows the Class imbalance in each dataset
def visualise_class_imbalance_train(y_train, dataset_name):
    print(f"\n {dataset_name} - Target Distribution (Train):")
    print(y_train.value_counts())

    plt.figure(figsize=(5, 4))
    sns.countplot(x=y_train, palette='pastel')
    plt.title(f"{dataset_name} - Class Distribution (Train)")
    plt.xlabel("Target Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Checks if the datasets have missing values
def check_missing_values(df, dataset_name, continuous_cols):
    print(f"\n Checking missing values and feature ranges for {dataset_name}:\n")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print(" No missing values found.")
    else:
        print(f" Missing values found:\n{missing[missing > 0]}")

# Plots categorical features by Target
def plot_cat_features_with_target_train(X_train, y_train, dataset_name, cat_cols):
    print(f"\n Plotting {len(cat_cols)} categorical features for {dataset_name} (Train Set):")
    df_train = X_train.copy()
    df_train['Target'] = y_train.values
    
    # Define colours for target classes
    pastel_red = '#FF9999'
    pastel_blue = '#99CCFF'
    hue_order = sorted(df_train['Target'].unique())
    colors = [pastel_red, pastel_blue] 
    palette = dict(zip(hue_order, colors))

    # Loop over categorical features and plot countplots by target
    for col in cat_cols:
        if col in df_train.columns:
            plt.figure(figsize=(7, 4))
            sns.countplot(data=df_train, y=col, hue='Target',  palette=palette,
                          order=df_train[col].value_counts().index)
            plt.title(f"{dataset_name} - '{col}' Distribution by Target (Train Set)")
            plt.xlabel("Count", fontsize=14)
            plt.ylabel(col, fontsize=14)
            plt.legend(title='Target', fontsize=16)
            plt.tight_layout()
            plt.show()

# Computes Cramér's V for inter-feature association.
def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))  # Correction for bias   
    rcorr = r - ((r-1)**2) / (n-1)
    kcorr = k - ((k-1)**2) / (n-1)
    return np.sqrt(phi2corr / max((kcorr - 1), (rcorr - 1)))

# Applies Cramér's V to all categorical features vs Target
def compute_cramers_v_train(X_train, y_train, cat_cols, dataset_name):
    print(f"\n Cramér's V with 'Target' for {dataset_name} (Train Set):")
    df_train = X_train.copy()
    df_train['Target'] = y_train.values
    results = pd.Series({col: cramers_v(pd.crosstab(df_train[col], df_train['Target'])) for col in cat_cols}).sort_values(ascending=False)
    print(results)

# Running EDA functions for the German and Australian datasets
detect_and_plot_outliers(X_train_german, continuous_german, "German Credit Train Set")
detect_and_plot_outliers(X_train_aus, continuous_aus, "Australian Credit Train Set")

plot_distributions_and_skew(X_train_german, continuous_german, "German Credit Train Set")
plot_distributions_and_skew(X_train_aus, continuous_aus, "Australian Credit Train Set")

plot_correlation_heatmap(X_train_german, continuous_german, "German Credit Train Set")
plot_correlation_heatmap(X_train_aus, continuous_aus, "Australian Credit Train Set")

visualise_class_imbalance_train(y_train_german, "German Credit")
visualise_class_imbalance_train(y_train_aus, "Australian Credit")

check_missing_values(X_train_german, "German Credit Train Set", continuous_german)
check_missing_values(X_train_aus, "Australian Credit Train Set", continuous_aus)

plot_cat_features_with_target_train(X_train_german, y_train_german, "German Credit Dataset", categorical_german)
plot_cat_features_with_target_train(X_train_aus, y_train_aus, "Australian Credit Dataset", categorical_aus)

compute_cramers_v_train(X_train_german, y_train_german, categorical_german, "German Credit Dataset")
compute_cramers_v_train(X_train_aus, y_train_aus, categorical_aus, "Australian Credit Dataset")

#------------------------------------------------------------------------------------------------------------------------------------
####################################################
# PREPROCESSING SECTION
####################################################

####################################################
# SELECT AND APPLY BEST SKEW TO CONTINUOUS FEATURES
####################################################

# Selects the best transformation (Log, Box-Cox, Yeo-Johnson)
# for continuous training features by minimising average skewness
def select_best_transformation_on_train(df_train, continuous_cols, name):
    print(f"\nSelecting best transformation for {name} (train only)...")

    transformers = {
        'Log': FunctionTransformer(np.log1p, validate=True),
        'Box-Cox': PowerTransformer(method='box-cox', standardize=False),
        'Yeo-Johnson': PowerTransformer(method='yeo-johnson', standardize=False)}

    scores, skew_details = {}, {}
    for key, trans in transformers.items():
        vals = []
        for c in continuous_cols:
            data = df_train[c].dropna()
            if key in ['Log','Box-Cox'] and (data <= 0).any(): 
                skew_details.setdefault(key,{})[c] = None
                continue
            try:
                sk = abs(skew(trans.fit_transform(data.values.reshape(-1,1)).ravel()))
                vals.append(sk); skew_details.setdefault(key,{})[c] = sk
            except: skew_details.setdefault(key,{})[c] = None
        scores[key] = np.mean(vals) if vals else np.inf

    best = min(scores, key=scores.get)
    print(f"\nBest for {name}: {best} (avg |skew|={scores[best]:.3f})")
    for k,v in skew_details.items():
        print(f"\n  {k}: " + " ".join([f"{c}:{s:.4f}" if s else f"{c}:skipped" for c,s in v.items()]))
    return best

# Applies the chosen transformation to continuous 
# features in both train and test sets
def apply_transformation(train_df, test_df, cols, method):
    mapping = {'Log': FunctionTransformer(np.log1p, validate=True),
               'Box-Cox': PowerTransformer(method='box-cox', standardize=False),
               'Yeo-Johnson': PowerTransformer(method='yeo-johnson', standardize=False)}
    trans = mapping[method]
    Xtr, Xte = train_df.copy(), test_df.copy()

    for c in cols:
        if method in ['Log','Box-Cox'] and (train_df[c] <= 0).any():
            print(f" Skipping {c} for {method}"); continue
        trans.fit(Xtr[[c]])
        Xtr[c], Xte[c] = trans.transform(Xtr[[c]]), trans.transform(Xte[[c]])
    return Xtr, Xte

# Finds best skew reduction method for each dataset
best_ger = select_best_transformation_on_train(X_train_german, continuous_german, "German")
best_aus = select_best_transformation_on_train(X_train_aus,    continuous_aus,    "Australian")

# If Box-Cox selected but data not strictly positive, fallback to Yeo-Johnson
if best_aus == 'Box-Cox':
    for c in continuous_aus:
        if (X_train_aus[c] <= 0).any():
            print(f" Box-Cox invalid for Australian ('{c}' contains non-positive). Falling back to Yeo-Johnson.")
            best_aus = 'Yeo-Johnson'
            break
# Similarly for German
if best_ger == 'Box-Cox':
    for c in continuous_german:
        if (X_train_german[c] <= 0).any():
            print(f" Box-Cox invalid for German ('{c}' contains non-positive). Falling back to Yeo-Johnson.")
            best_ger = 'Yeo-Johnson'
            break

#########################################
# ENCODING COLUMN LISTS
#########################################

# German encoding lists(One-Hot and Ordinal)
ger_ohe = [
    'Status_of_existing_checking_account','Credit_history','Purpose',
    'Savings_account_bonds','Personal_status_and_sex','Other_debtors_guarantors',
    'Property','Other_installment_plans','Housing','Telephone','Foreign_worker']
ger_ord = ['Present_employment_since','Job']
ger_pass = [
    'Installment_rate_in_percentage_of_disposable_income',
    'Present_residence_since',
    'Number_of_existing_credits_at_this_bank',
    'Number_of_people_being_liable_to_provide_maintenance_for']

# Australian encoding lists(One-Hot, Ordinal and Frequency)
aus_ohe = ['Marital_Status','Bank_Customer','Drivers_License']
aus_ord = ['Education_Level']
freq_cols = ['Ethnicity','Citizen']
aus_pass = ['Gender_Status','Years_Employed','Prior_Default','Credit_Score']

######################################
# BUILD PREPROCESSING PIPELINES
######################################

# Frequency encoding for Australian dataset
def freq_encode(df):
    df = df.copy()
    for c in freq_cols:
        f = df[c].value_counts(normalize=True)
        df[f'{c}_freq'] = df[c].map(f).fillna(0)
    return df

# Wrap in FunctionTransformer for pipeline compatibility
freq_tf_aus = FunctionTransformer(func=freq_encode, validate=False)

# Continuous Feature Pipelines
method_map = {'Box-Cox': 'box-cox', 'Yeo-Johnson': 'yeo-johnson'}

# German continuous features: apply best skew transform + scaling
cont_pipe_ger = Pipeline([
    ('power', PowerTransformer(method=method_map.get(best_ger,best_ger), standardize=False) if best_ger!='Log' else 
     FunctionTransformer(np.log1p, validate=True)),
    ('scale', RobustScaler())])
# Australian continuous features: apply best skew transform + scaling
cont_pipe_aus = Pipeline([
    ('power', PowerTransformer(method=method_map.get(best_aus,best_aus), standardize=False) if best_aus!='Log' else 
     FunctionTransformer(np.log1p, validate=True)),
    ('scale', RobustScaler())])

# Column Transformers
# German dataset: continuous + one-hot + ordinal, remaining columns passthrough
preproc_ger = ColumnTransformer([
    ('cont', cont_pipe_ger, continuous_german),
    ('ohe',  OneHotEncoder(handle_unknown='ignore', sparse_output=False), ger_ohe),
    ('ord',  OrdinalEncoder(), ger_ord),
], remainder='passthrough')
# Australian dataset: continuous + one-hot + ordinal, remaining columns passthrough
preproc_aus = ColumnTransformer([
    ('cont', cont_pipe_aus, continuous_aus),
    ('ohe',  OneHotEncoder(handle_unknown='ignore', sparse_output=False), aus_ohe),
    ('ord',  OrdinalEncoder(), aus_ord),
], remainder='passthrough')

# Full preprocessing pipelines
# German: directly apply column transformer
pipeline_german = Pipeline([('preproc', preproc_ger)])

# Australian: first apply frequency encoding, then column transformer
pipeline_aus = Pipeline([('freq', freq_tf_aus),('preproc', preproc_aus)])

# Apply Preprocessing Pipeline to German and Australian datasets
X_train_german_pre = pipeline_german.fit_transform(X_train_german)
X_test_german_pre = pipeline_german.transform(X_test_german)

X_train_aus_pre = pipeline_aus.fit_transform(X_train_aus)
X_test_aus_pre = pipeline_aus.transform(X_test_aus)

########################################################################
# PLOT DISTRIBUTIONS ORIGINALLY, AFTER TRANSFORMATION AND AFTER SCALING
########################################################################

numerical_cols = ['Duration_in_month', 'Credit_amount', 'Age_in_years']
# Extract transformation steps for numeric features
yeo = pipeline_german.named_steps['preproc'].named_transformers_['cont'].named_steps['power']
scaler = pipeline_german.named_steps['preproc'].named_transformers_['cont'].named_steps['scale']

# Apply transformations step-by-step
X_yeo = yeo.transform(X_train_german[numerical_cols])   # After Yeo-Johnson
X_scaled = scaler.transform(X_yeo)                      # After RobustScaler

# Plot original, Yeo-Johnson, and scaled distributions side by side
for i, feature in enumerate(numerical_cols):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Original distribution
    sns.kdeplot(X_train_german[feature], fill=True, ax=axes[0], color='#1f77b4', linewidth=2)
    axes[0].set_title(f"{feature} - Original")

    # After Yeo-Johnson
    sns.kdeplot(X_yeo[:, i], fill=True, ax=axes[1], color='#1f77b4', linewidth=2)
    axes[1].set_title(f"{feature} - After Yeo-Johnson")

    # After Standard Scaling
    sns.kdeplot(X_scaled[:, i], fill=True, ax=axes[2], color='#1f77b4', linewidth=2)
    axes[2].set_title(f"{feature} - After Scaling")

    plt.tight_layout()
    plt.show()

###################################
# CHECK CLASS WEIGHTS
###################################

# German Dataset class weights
classes_ger = np.unique(y_train_german)
weights_ger = compute_class_weight(class_weight='balanced', classes=classes_ger, y=y_train_german)
class_weights_ger = dict(zip(classes_ger, weights_ger))
print("German class weights:", class_weights_ger)

# Australian Dataset class weights
classes_aus = np.unique(y_train_aus)
weights_aus = compute_class_weight(class_weight='balanced', classes=classes_aus, y=y_train_aus)
class_weights_aus = dict(zip(classes_aus, weights_aus))
print("Australian class weights:", class_weights_aus)

#------------------------------------------------------------------------
##################################
# CLUSTERING SECTION
##################################

# Finding Optimal K by plotting metrics like Inertia, Silhouette Score 
# and Davies-Bouldin Index for different K values for K-Means Clustering
def evaluate_clustering_scores(X, k_range=range(2, 11), dataset_name="Dataset"):
    inertia_list = []
    silhouette_list = []
    davies_list = []

    print(f"\nEvaluating clustering quality for {dataset_name}:")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(X)

        inertia = kmeans.inertia_
        silhouette = silhouette_score(X, labels)
        davies = davies_bouldin_score(X, labels)

        inertia_list.append(inertia)
        silhouette_list.append(silhouette)
        davies_list.append(davies)
        print(f"k={k}: Inertia={inertia:.2f}, Silhouette={silhouette:.3f}, Davies-Bouldin={davies:.3f}")

    # Plot Elbow (Inertia)
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.plot(k_range, inertia_list, 'bo-')
    plt.xlabel("Number of Clusters (k)", fontsize=14)
    plt.ylabel("Inertia", fontsize=14)
    plt.title(f"{dataset_name} - Elbow Method", fontsize=14)

    # Plot Silhouette Score
    plt.subplot(1, 3, 2)
    plt.plot(k_range, silhouette_list, 'go-')
    plt.xlabel("Number of Clusters (k)", fontsize=14)
    plt.ylabel("Silhouette Score", fontsize=14)
    plt.title(f"{dataset_name} - Silhouette Score", fontsize=14)
    
    # Plot Davies-Bouldin Index
    plt.subplot(1, 3, 3)
    plt.plot(k_range, davies_list, 'ro-')
    plt.xlabel("Number of Clusters (k)", fontsize=14)
    plt.ylabel("Davies-Bouldin Index", fontsize=14)
    plt.title(f"{dataset_name} - Davies-Bouldin Index", fontsize=14)
    plt.tight_layout()
    plt.show()

# Plot figures to find optimal K
evaluate_clustering_scores(X_train_german_pre, dataset_name="German Credit Train Set")
evaluate_clustering_scores(X_train_aus_pre, dataset_name="Australian Credit Train Set")


def cluster_and_visualise(X, y_true=None, dataset_name="Dataset", k=4):
    print(f"\n Running KMeans for {dataset_name} with k={k}:")

    # Run K-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X)
    
    # Print Optimal Metric Values for chosen K
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    print(f" Inertia: {inertia:.2f}")
    print(f" Silhouette Score: {silhouette:.4f}")
    print(f" Davies-Bouldin Index: {db_index:.4f}")

    # UMAP Visualisation
    print("Reducing dimensions using UMAP:")
    umap_embedded = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # KMeans cluster labels on the left
    scatter_kmeans = axs[0].scatter(umap_embedded[:, 0], umap_embedded[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.8)
    axs[0].set_title(f"{dataset_name} - UMAP (KMeans Clusters)", fontsize=14)
    axs[0].set_xlabel("UMAP Dim 1", fontsize=12)
    axs[0].set_ylabel("UMAP Dim 2", fontsize=12)
    axs[0].grid(True)
    fig.colorbar(scatter_kmeans, ax=axs[0], label="Cluster")
    
    # True labels on the right
    scatter_true = axs[1].scatter(umap_embedded[:, 0], umap_embedded[:, 1], c=y_true, cmap='coolwarm', alpha=0.8)
    axs[1].set_title(f"{dataset_name} - UMAP (True Labels)", fontsize=14)
    axs[1].set_xlabel("UMAP Dim 1", fontsize=12)
    axs[1].set_ylabel("UMAP Dim 2", fontsize=12)
    axs[1].grid(True)
    fig.colorbar(scatter_true, ax=axs[1], label="True Class")
    
    fig.suptitle(f"{dataset_name} - UMAP Cluster Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
    # t-SNE Visualisation 
    print("Reducing dimensions using t-SNE...")
    tsne_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # KMeans cluster labels on the left
    scatter_kmeans = axs[0].scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.8)
    axs[0].set_title(f"{dataset_name} - t-SNE (KMeans Clusters)", fontsize=14)
    axs[0].set_xlabel("t-SNE Dim 1", fontsize=12)
    axs[0].set_ylabel("t-SNE Dim 2", fontsize=12)
    axs[0].grid(True)
    fig.colorbar(scatter_kmeans, ax=axs[0], label="Cluster")
    
    # True labels on the right
    scatter_true = axs[1].scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], c=y_true, cmap='coolwarm', alpha=0.8)
    axs[1].set_title(f"{dataset_name} - t-SNE (True Labels)", fontsize=14)
    axs[1].set_xlabel("t-SNE Dim 1", fontsize=12)
    axs[1].set_ylabel("t-SNE Dim 2", fontsize=12)
    axs[1].grid(True)
    fig.colorbar(scatter_true, ax=axs[1], label="True Class")
    
    fig.suptitle(f"{dataset_name} - t-SNE Cluster Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()

    return labels, kmeans

# Compute Feature means for each cluster
def analyse_cluster_profiles(X, labels, feature_names=None, dataset_name="Dataset"):
    print(f"\n Analyzing cluster profiles for {dataset_name}:")
    
    # Makes sure X is a Dataframe
    if isinstance(X, np.ndarray):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X.copy()
    
    X_df["Cluster"] = labels #Add cluster labels
    
    # Compute cluster-wise mean of each feature and print
    cluster_profiles = X_df.groupby("Cluster").mean().T
    print("\n Cluster Feature Means:")
    with pd.option_context('display.max_columns', None):
        print(cluster_profiles.round(2))
    
    return cluster_profiles

# Chi-Squared Test for association between Clusters and Target
def evaluate_cluster_label_association(cluster_labels, true_labels, dataset_name="Dataset"):
    
    # Contingency Table and perform Chi-Squared test
    contingency_table = pd.crosstab(cluster_labels, true_labels)
    print("\nContingency Table (Clusters vs True Labels):")
    print(contingency_table)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nChi-squared Statistic = {chi2:.4f}")
    print(f"Degrees of Freedom = {dof}")
    print(f"P-value = {p:.4e}")

    if p < 0.05:
        print(" Clusters are significantly associated with the true target labels.")
    else:
        print(" No significant association between clusters and target labels.")

    # Visualize residuals (observed - expected) for each cluster
    residuals = contingency_table - expected
    plt.figure(figsize=(8, 6))
    sns.heatmap(residuals, annot=True, cmap="coolwarm", center=0)
    plt.title(f"{dataset_name} - Chi-Squared Residuals (Observed - Expected)")
    plt.xlabel("True Labels")
    plt.ylabel("Cluster")
    plt.show()

# Run KMeans, visualise and prints feature means in each German Cluster
# Performs Chi-Squared Test of association between Clusters and Target
labels_german, kmeans_german = cluster_and_visualise(
    X_train_german_pre, y_train_german, dataset_name="German Credit", k=4)

feature_names_german = pipeline_german.named_steps['preproc'].get_feature_names_out()
analyse_cluster_profiles(X_train_german_pre, labels_german, feature_names=feature_names_german, dataset_name="German Credit")
evaluate_cluster_label_association(labels_german, y_train_german, dataset_name="German Credit")

# Run KMeans, visualise and prints feature means in each Australian Cluster
# Performs Chi-Squared Test of association between Clusters and Target
labels_aus, kmeans_aus = cluster_and_visualise(
    X_train_aus_pre, y_train_aus, dataset_name="Australian Credit", k=9)

feature_names_aus = pipeline_aus.named_steps['preproc'].get_feature_names_out()
analyse_cluster_profiles(X_train_aus_pre, labels_aus, feature_names=feature_names_aus, dataset_name="Australian Credit")
evaluate_cluster_label_association(labels_aus, y_train_aus, dataset_name="Australian Credit")

# Perform Principal Component Analysis while keeping 95% variance
def run_pca(X, n_components=0.95, dataset_name="Dataset"):
    print(f"\nRunning PCA on {dataset_name}:")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained Variance Ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    return X_pca, pca

# Perform all the clustering analysis again after PCA to compare
# New optimal K values of 3 and 9 have been used
X_pca_german, pca_german = run_pca(X_train_german_pre, dataset_name="German Credit")
X_pca_aus, pca_aus = run_pca(X_train_aus_pre, dataset_name="Australian Credit")

evaluate_clustering_scores(X_pca_german, dataset_name="German Credit (PCA)")
evaluate_clustering_scores(X_pca_aus, dataset_name="Australian Credit (PCA)")

labels_german_pca, kmeans_german_pca = cluster_and_visualise(
    X_pca_german, y_train_german, dataset_name="German Credit (PCA)", k=3)
labels_aus_pca, kmeans_aus_pca = cluster_and_visualise(
    X_pca_aus, y_train_aus, dataset_name="Australian Credit (PCA)", k=9)

analyse_cluster_profiles(X_pca_german, labels_german_pca, feature_names=[f"PC{i+1}" for i in range(X_pca_german.shape[1])], dataset_name="German Credit (PCA)")
analyse_cluster_profiles(X_pca_aus, labels_aus_pca, feature_names=[f"PC{i+1}" for i in range(X_pca_aus.shape[1])], dataset_name="Australian Credit (PCA)")

evaluate_cluster_label_association(labels_german_pca, y_train_german, dataset_name="German Credit (PCA)")
evaluate_cluster_label_association(labels_aus_pca, y_train_aus, dataset_name="Australian Credit (PCA)")

#------------------------------------------------------------------------
##################################
# CLASSIFICATION SECTION
##################################

# Classification Metric Visualisation
# Plots Confusion Matrix, ROC curve, Precision-Recall curve and Calibration plot
def plot_all_classification_curves(y_test, y_pred, y_prob, title_prefix):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # ensure order
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 14}, linewidths=0.5, linecolor='black')
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.title(f'{title_prefix} - Confusion Matrix', fontsize=14)
    plt.xticks(ticks=[0.5, 1.5], labels=['Bad', 'Good'], fontsize=12)
    plt.yticks(ticks=[0.5, 1.5], labels=['Bad', 'Good'], fontsize=12)
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title(f"{title_prefix} - ROC Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

    # Precision–Recall Curve
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec, prec)
    plt.figure()
    plt.plot(rec, prec, color='blue', lw=2, label=f"AUC = {pr_auc:.3f}")
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title(f"{title_prefix} - Precision–Recall Curve", fontsize=14)
    plt.legend(loc="lower left", fontsize=12)
    plt.show()

    # Calibration Plot
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=15)
    plt.figure()
    plt.plot(prob_pred, prob_true, color='blue', marker='o', lw=2, markersize=4)  # smaller markers
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("Mean Predicted Probability", fontsize=14)
    plt.ylabel("Fraction of Positives", fontsize=14)
    plt.title(f"{title_prefix} - Calibration Plot", fontsize=14)
    plt.show()

# Convert Target Labels to 0/1 if they havent been already
def convert_labels_binary(y_train, y_test):
    unique_labels = np.unique(y_train)
    if set(unique_labels) != {0, 1}:
        y_train_bin = np.where(y_train == unique_labels.min(), 0, 1)
        y_test_bin = np.where(y_test == unique_labels.min(), 0, 1)
        return y_train_bin, y_test_bin
    return y_train, y_test

# Add Cluster Features to the Datasets
def add_cluster_features(X_train, X_test, kmeans):
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    train_clusters = enc.fit_transform(kmeans.predict(X_train).reshape(-1, 1))
    test_clusters = enc.transform(kmeans.predict(X_test).reshape(-1, 1))
    return np.hstack((X_train, train_clusters)), np.hstack((X_test, test_clusters))

# Tunes the model with the best hyperparameters using GridSearch Cross Validation
def tune_model(model, param_grid, X_train, y_train):
    grid = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    return grid.best_estimator_

# Evaluate models and plot curves 
def evaluate_models(X_train, y_train, X_test, y_test, dataset_name="Dataset", balanced=False, weighted=False):

    # Make sure labels are binary (0/1)
    y_train_bin, y_test_bin = convert_labels_binary(y_train, y_test)
    
    # Define models to evaluate
    print(f"\n=== {dataset_name} ===")
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, 
                            class_weight='balanced' if balanced else None, 
                            random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, 
                            class_weight='balanced' if balanced else None, 
                            random_state=42))]
    avg = 'weighted' if weighted else 'binary'
    
    # Train each model, predict, compute probabilities, evaluate, and plot
    for name, model in models:
        model.fit(X_train, y_train_bin)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        # Calculate AUC
        auc_score = roc_auc_score(y_test_bin, y_prob)

        # Print metrics
        print(f"\n{name}:")
        print(f"Accuracy: {accuracy_score(y_test_bin, y_pred):.3f}")
        print(f"AUC: {auc_score:.3f}")
        print(f"F1 Score: {f1_score(y_test_bin, y_pred, average=avg):.3f}")
        print(classification_report(y_test_bin, y_pred, digits=3))
        
        # Plot all evaluation curves
        plot_all_classification_curves(y_test_bin, y_pred, y_prob, f"{dataset_name} - {name}")

# Finds best parameters, fits models, prints metrics, and visualises curves
def evaluate_models_with_tuning(X_train, y_train, X_test, y_test, dataset_name="Dataset", balanced=False, weighted=False):
    
    # Make sure labels are binary (0/1)
    y_train_bin, y_test_bin = convert_labels_binary(y_train, y_test)
    
    print(f"\n=== {dataset_name} ===")
    avg = 'weighted' if weighted else 'binary'

    # Logistic Regression parameter grid
    LR_params = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2'],
        'class_weight': [None, 'balanced'] if not balanced else ['balanced']}

    # Random Forest parameter grid
    RF_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced'] if not balanced else ['balanced']}

    # Tune Logistic Regression
    print("\nTuning Logistic Regression:")
    best_lr = tune_model(LogisticRegression(max_iter=1000, random_state=42), LR_params, X_train, y_train_bin)

    # Tune Random Forest
    print("\nTuning Random Forest:")
    best_rf = tune_model(RandomForestClassifier(random_state=42), RF_params, X_train, y_train_bin)

    # Evaluate tuned models
    tuned_models = [("Logistic Regression (Tuned)", best_lr), ("Random Forest (Tuned)", best_rf)]

    for name, model in tuned_models:
        model.fit(X_train, y_train_bin)
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        acc = accuracy_score(y_test_bin, y_pred)
        f1 = f1_score(y_test_bin, y_pred, average=avg)
        auc = roc_auc_score(y_test_bin, y_prob)

        print(f"\n{name}:")
        print(f"Accuracy: {acc:.3f}")
        print(f"F1 Score: {f1:.3f}")
        print(f"AUC: {auc:.3f}")
        print(classification_report(y_test_bin, y_pred, digits=3))
        plot_all_classification_curves(y_test_bin, y_pred, y_prob, f"{dataset_name} - {name}")

    return best_rf

# Train & evaluate models on original features
print("\n=== Step 1: Baseline ===")
evaluate_models(X_train_german_pre, y_train_german, X_test_german_pre, y_test_german, "German Credit - Baseline", balanced=True)
evaluate_models(X_train_aus_pre, y_train_aus, X_test_aus_pre, y_test_aus, "Australian Credit - Baseline", balanced=True)

# Add cluster Features
print("\n=== Step 2: Add Cluster Features ===")
X_train_german_c, X_test_german_c = add_cluster_features(X_train_german_pre, X_test_german_pre, kmeans_german)
X_train_aus_c, X_test_aus_c = add_cluster_features(X_train_aus_pre, X_test_aus_pre, kmeans_aus)

# Train & evaluate models with added cluster features
evaluate_models(X_train_german_c, y_train_german, X_test_german_c, y_test_german, "German Credit - Baseline + Clusters", balanced=True)
evaluate_models(X_train_aus_c, y_train_aus, X_test_aus_c, y_test_aus, "Australian Credit - Baseline + Clusters", balanced=True)

# Train & evaluate models with tuned parameters and added cluster features
print("\n=== Step 3: Hyperparameter Tuning on Cluster Features ===")
evaluate_models_with_tuning(X_train_german_c, y_train_german, X_test_german_c, y_test_german, "German Credit - Tuned + Clusters", balanced=True)
evaluate_models_with_tuning(X_train_aus_c, y_train_aus, X_test_aus_c, y_test_aus, "Australian Credit - Tuned + Clusters", balanced=True)

# Train & evaluate models with original features and tuned parameters
print("\n=== Step 4: Hyperparameter Tuning (Original Features) ===")
evaluate_models_with_tuning(X_train_german_pre, y_train_german, X_test_german_pre, y_test_german, "German Credit", balanced=True)
evaluate_models_with_tuning(X_train_aus_pre, y_train_aus, X_test_aus_pre, y_test_aus, "Australian Credit", balanced=True)


# Australian Dataset Feature Name Mapping
# Get feature names from ColumnTransformer 
encoded_feature_names_aus = pipeline_aus.named_steps['preproc'].get_feature_names_out()
freq_cols_aus = ['col1', 'col2']
freq_cols_aus_names = [f"{c}_freq" for c in freq_cols_aus]
all_feature_names_aus = list(encoded_feature_names_aus) + freq_cols_aus_names
# Check if cluster features exist and add
n_extra = X_train_aus_c.shape[1] - len(all_feature_names_aus)
cluster_names_aus = [f"Cluster_{i}" for i in range(n_extra)] if n_extra > 0 else []
# Final feature names for Australian dataset
feat_names_aus = all_feature_names_aus + cluster_names_aus
print("Australian feature names:")
print(feat_names_aus)

# German Dataset Feature Name Mapping
# Get feature names from ColumnTransformer 
encoded_feature_names_ger = pipeline_german.named_steps['preproc'].get_feature_names_out()
freq_cols_ger = ['colA', 'colB']
freq_cols_ger_names = [f"{c}_freq" for c in freq_cols_ger]
all_feature_names_ger = list(encoded_feature_names_ger) + freq_cols_ger_names
# Check if cluster features exist and add
n_extra = X_train_german_c.shape[1] - len(all_feature_names_ger)
cluster_names_ger = [f"Cluster_{i}" for i in range(n_extra)] if n_extra > 0 else []
# Final feature names for German dataset
feat_names_german = all_feature_names_ger + cluster_names_ger
print("\nGerman feature names:")
print(feat_names_german)

# German Credit - Random Forest Feature Importance analysis
best_rf_german = evaluate_models_with_tuning(
    X_train_german_c, y_train_german, X_test_german_c, y_test_german,
    "German Credit - Tuned + All Features", balanced=True)

feat_imp_german = pd.DataFrame({
    'Feature': feat_names_german,
    'Importance': best_rf_german.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Australian Credit - Random Forest Feature Importance analysis
best_rf_aus = evaluate_models_with_tuning(
    X_train_aus_c, y_train_aus, X_test_aus_c, y_test_aus,
    "Australian Credit - Tuned + All Features", balanced=True)

feat_imp_aus = pd.DataFrame({
    'Feature': feat_names_aus,
    'Importance': best_rf_aus.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print feature importances 
print("German Credit Feature Importances:")
print(feat_imp_german.to_string(index=False))

print("\nAustralian Credit Feature Importances:")
print(feat_imp_aus.to_string(index=False))
