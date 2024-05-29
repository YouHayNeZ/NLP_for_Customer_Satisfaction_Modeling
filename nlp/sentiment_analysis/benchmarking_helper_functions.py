import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def plot_sentiment_percentages(merged_sentiments_df):
    # Exclude the 'Comment' column
    columns = merged_sentiments_df.columns.drop('Comment')

    # Calculate the percentage of +1, 0, and -1 values for each column
    percentages = merged_sentiments_df[columns].apply(lambda x: x.value_counts(normalize=True) * 100).fillna(0)

    # Plotting the bar charts
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(15, 5))

    for i, column in enumerate(columns):
        ax = axes[i]
        percentages[column].plot(kind='bar', ax=ax)
        ax.set_title(column, fontsize=10)  # Adjusting the font size of the title
        ax.set_xlabel('Sentiment', fontsize=8)
        ax.set_ylabel('Percentage', fontsize=8)
        ax.set_ylim([0, 100])
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.show()


def calculate_metrics(analysis_df, ground_truth_label, sentiment_columns):
    # Initialize dictionaries to store the metrics
    metrics_results = {
        'accuracy': {},
        'precision': {},
        'recall': {},
        'f1_score': {}
    }

    # Calculate metrics for each sentiment column
    for column in sentiment_columns:
        y_true = analysis_df[ground_truth_label]
        y_pred = analysis_df[column]

        metrics_results['accuracy'][column] = accuracy_score(y_true, y_pred)
        metrics_results['precision'][column] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics_results['recall'][column] = recall_score(y_true, y_pred, average='macro')
        metrics_results['f1_score'][column] = f1_score(y_true, y_pred, average='macro')

    # Convert metrics results to a DataFrame
    metrics_df = pd.DataFrame(metrics_results)
    return metrics_df

# Function to plot metrics
def plot_metrics(metrics_df):
    metrics = metrics_df.columns
    num_metrics = len(metrics)
    
    # Calculate the number of rows and columns for subplots
    nrows = (num_metrics + 1) // 2
    ncols = 2
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10 * nrows))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        sorted_df = metrics_df.sort_values(by=metric, ascending=False)
        colors = ['green' if i == 0 else 'blue' for i in range(len(sorted_df))]
        ax = axes[idx]
        bars = ax.bar(sorted_df.index, sorted_df[metric], color=colors)
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Model')
        ax.set_xticklabels(sorted_df.index, rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, color='black')
    
    plt.tight_layout()
    plt.show()