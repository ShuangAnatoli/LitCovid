import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def analyze_label_distribution(file_path):
    """
    Analyze the distribution of labels in a COVID-19 research articles dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the dataset
        
    Returns:
    --------
    None, but prints analysis results and generates visualizations
    """
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset with {len(df)} records")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Display basic info about the dataset
    print("\nDataset columns:")
    print(df.columns.tolist())
    
    # Check for missing values in the label column
    missing_labels = df['label'].isna().sum()
    print(f"\nNumber of records with missing labels: {missing_labels}")
    
    # Count the occurrence of each label (considering that labels may be semicolon-separated)
    label_counts = {}
    total_articles = len(df)
    
    for labels in df['label'].dropna():
        for label in labels.split(';'):
            label = label.strip()
            if label:
                label_counts[label] = label_counts.get(label, 0) + 1
    
    # Sort labels by frequency
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate percentages
    label_percentages = {label: (count / total_articles) * 100 
                          for label, count in label_counts.items()}
    
    # Print frequency and percentage of each label
    print("\nLabel distribution:")
    print("-------------------")
    for label, count in sorted_labels:
        print(f"{label}: {count} articles ({label_percentages[label]:.2f}%)")
    
    # Count articles with multiple labels
    multiple_labels_count = sum(1 for labels in df['label'].dropna() 
                               if ";" in labels)
    
    print(f"\nArticles with multiple labels: {multiple_labels_count} ({(multiple_labels_count/total_articles)*100:.2f}%)")
    
    # Calculate label co-occurrence
    if multiple_labels_count > 0:
        print("\nLabel co-occurrences:")
        print("--------------------")
        
        label_pairs = {}
        for labels in df['label'].dropna():
            if ";" in labels:
                label_list = [l.strip() for l in labels.split(';') if l.strip()]
                for i, label1 in enumerate(label_list):
                    for label2 in label_list[i+1:]:
                        pair = tuple(sorted([label1, label2]))
                        label_pairs[pair] = label_pairs.get(pair, 0) + 1
        
        # Sort co-occurrences by frequency
        sorted_pairs = sorted(label_pairs.items(), key=lambda x: x[1], reverse=True)
        
        for (label1, label2), count in sorted_pairs:
            print(f"{label1} + {label2}: {count} articles")
    
    # Data visualization
    plt.figure(figsize=(12, 8))
    
    # Bar chart of label distribution
    plt.subplot(2, 2, 1)
    labels, counts = zip(*sorted_labels)
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Label Distribution')
    plt.ylabel('Number of Articles')
    plt.tight_layout()
    
    # Pie chart of label distribution
    plt.subplot(2, 2, 2)
    plt.pie([count for _, count in sorted_labels], 
            labels=[label for label, _ in sorted_labels],
            autopct='%1.1f%%')
    plt.title('Label Distribution (%)')
    
    # Save the visualization if needed
    # plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
    
    # Show the plot (comment this out if running in a non-interactive environment)
    plt.tight_layout()
    plt.show()
    
    # Additional analysis: Time trend if publication date is available
    if 'pub_date' in df.columns:
        try:
            df['pub_date'] = pd.to_datetime(df['pub_date'])
            print("\nAnalyzing publication trends over time...")
            
            # Extract month-year
            df['month_year'] = df['pub_date'].dt.to_period('M')
            
            # Count publications per month
            monthly_counts = df.groupby('month_year').size()
            
            plt.figure(figsize=(12, 6))
            monthly_counts.plot(kind='bar')
            plt.title('Number of Publications per Month')
            plt.xlabel('Month')
            plt.ylabel('Number of Publications')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not analyze time trends: {e}")
    
    return df, label_counts

# Run the analysis
if __name__ == "__main__":
    # Replace with your file path
    file_path = "BC7-LitCovid-Test-GS.csv"  # Update this path
    analyze_label_distribution(file_path)