# src/utils/perform_univariate_analysis.py

from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def perform_univariate_analysis(
    df, 
    column_name, 
    show_graphs=True, 
    show_distribution=True,
    bins=30
):
    """
    Perform comprehensive analysis on a specific column in the DataFrame.

    Non-Graphical Analysis:
    - Data type and number of unique values.
    - For categorical columns: unique categories, value counts, and proportions.
    - For numerical columns: summary statistics, skewness, and kurtosis.
    - Missing values (optional).

    Graphical Analysis (optional):
    - Categorical columns: countplot, pie chart (with "Others" for low-frequency categories).
    - Numerical columns: histogram with KDE, boxplot, violin plot, and QQ plot.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to analyze.
        show_graphs (bool): Whether to display graphical analysis. Default is True.
        bins (int): Number of bins for the histogram (default: 30).

    Returns:
        None: Results are printed and displayed interactively.
    """
    # Non-Graphical Analysis
    print(f"🔢 Data Type: {df[column_name].dtype}")
    print(f"💎 Number of Unique Values: {df[column_name].nunique()}")
    print(f"📋 List of Unique Categories: {df[column_name].unique().tolist()}")
    if show_distribution:
        # Categorical column analysis
        value_counts = df[column_name].value_counts(normalize=True).to_frame(name='Proportion')
        value_counts['Count'] = df[column_name].value_counts()
        print("📊 Value Distribution:")
        display(value_counts.style.format({
            'Count': '{:,}',  # Add thousand separators
            'Proportion': '{:.2%}'  # Format as percentage with 2 decimal places
        }))
    if df[column_name].dtype not in ['object', 'category']:
        summary_stats = df[column_name].describe().to_frame().T
        skewness = df[column_name].skew()
        kurtosis = df[column_name].kurtosis()
        print("📊 Summary Statistics:")
        display(summary_stats.style.format('{:.2f}'))
        print(f"📈 Skewness: {skewness:.2f}")
        print(f"📈 Kurtosis: {kurtosis:.2f}")

    # Check for missing values
    missing_count = df[column_name].isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100  # Calculate percentage of missing rows
    if missing_count == 0:
        print("✅ No missing values found.")
    else:
        print(f"⚠️ Rows with missing values: {missing_count} ({missing_percentage:.2f}% of total rows)")
        missing_rows = df[df[column_name].isnull()]
        display(missing_rows)

    # Graphical Analysis (Optional)
    if show_graphs:
        if df[column_name].dtype in ['object', 'category']:
            # Categorical Column Visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
            fig.suptitle(f"Graphical Analysis of '{column_name}'", fontsize=14, fontweight='bold')

            # Countplot with categories sorted by frequency
            value_counts = df[column_name].value_counts()
            sorted_categories = value_counts.index
            sns.countplot(data=df, x=column_name, ax=axes[0], order=sorted_categories)
            axes[0].set_title("Countplot", fontsize=14, fontweight='bold')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_xlabel("") 
            axes[0].set_ylabel("") 

            # Pie Chart with Top 4 Categories + "Others"
            if len(value_counts) > 5:
                top_4 = value_counts[:4]  # Get the top 4 categories
                others = pd.Series(value_counts[4:].sum(), index=["Others"])  # Combine the rest into "Others"
                pie_data = pd.concat([top_4, others])  # Combine top 4 and "Others"
            else:
                pie_data = value_counts  # If there are 5 or fewer categories, use all of them

            wedges, texts, autotexts = axes[1].pie(
                pie_data, 
                labels=pie_data.index,  # Display category names as labels
                autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100 * pie_data.sum())})',  # Format: percentage and count
                startangle=90, 
                colors=sns.color_palette('tab20c'),
                textprops={'fontsize': 10}  # Adjust font size of annotations
            )

            axes[1].set_title("Pie Chart", fontsize=14, fontweight='bold')
            axes[1].set_ylabel("") 

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

        else:
            # Numerical Column Visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f"Univariate Analysis of '{column_name}'", fontsize=18, fontweight='bold')

            # 1. Histogram with KDE
            sns.histplot(data=df, x=column_name, kde=True, bins=bins, color='teal', ax=axes[0, 0])
            axes[0, 0].set_title("Histogram with KDE", fontsize=14, fontweight='bold')

            # 2. Boxplot
            sns.boxplot(data=df, y=column_name, color='lightcoral', ax=axes[0, 1])
            axes[0, 1].set_title("Boxplot", fontsize=14, fontweight='bold')

            # 3. Violin Plot
            sns.violinplot(data=df, y=column_name, color='cornflowerblue', ax=axes[1, 0])
            axes[1, 0].set_title("Violin Plot", fontsize=14, fontweight='bold')

            # 4. QQ Plot
            sm.qqplot(df[column_name].dropna(), line='s', ax=axes[1, 1], markerfacecolor='gold', markeredgecolor='black')
            axes[1, 1].set_title("QQ Plot", fontsize=14, fontweight='bold')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()