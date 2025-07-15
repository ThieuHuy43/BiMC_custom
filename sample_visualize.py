import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def count_samples(root_dir):
    class_counts = defaultdict(lambda: defaultdict(int))
    total_samples = defaultdict(int)
    
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            if '_' in class_dir:
                country, fruit = class_dir.split('_')
            elif '-' in class_dir:
                country, fruit = class_dir.split('-')
            else:
                continue
                
            country = country.lower()
            if country in ['chinese', 'china']:
                country = 'China'
            elif country in ['vietnamese', 'vietnam']:
                country = 'Vietnam'
            else:
                country = country.title()
            
            fruit = fruit.title()
            n_samples = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            class_counts[country][fruit] = n_samples
            total_samples[f"{country}-{fruit}"] = n_samples
            
    return class_counts, total_samples

def plot_bar_distribution(total_samples):
    plt.figure(figsize=(15, 8))
    classes = list(total_samples.keys())
    counts = list(total_samples.values())
    
    plt.bar(classes, counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Number of Samples per Class')
    plt.ylabel('Number of Images')
    plt.xlabel('Classes')
    
    for i, v in enumerate(counts):
        plt.text(i, v + 1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('aug_bar_distribution.png')
    plt.close()

def plot_heatmap(class_counts):
    plt.figure(figsize=(12, 8))
    countries = list(class_counts.keys())
    fruits = set()
    for fruit_dict in class_counts.values():
        fruits.update(fruit_dict.keys())
    fruits = list(fruits)
    
    matrix = np.zeros((len(countries), len(fruits)))
    for i, country in enumerate(countries):
        for j, fruit in enumerate(fruits):
            matrix[i][j] = class_counts[country].get(fruit, 0)
    
    im = plt.imshow(matrix, cmap='YlOrRd')
    plt.colorbar(im)
    
    for i in range(len(countries)):
        for j in range(len(fruits)):
            plt.text(j, i, int(matrix[i, j]), ha='center', va='center')
    
    plt.xticks(range(len(fruits)), fruits, rotation=45)
    plt.yticks(range(len(countries)), countries)
    plt.title('Sample Count Distribution (Country vs Fruit)')
    
    plt.tight_layout()
    plt.savefig('aug_heatmap_distribution.png')
    plt.close()

def plot_pie_chart(total_samples):
    plt.figure(figsize=(12, 8))
    classes = list(total_samples.keys())
    counts = list(total_samples.values())
    
    plt.pie(counts, labels=classes, autopct='%1.1f%%')
    plt.title('Class Distribution (% of Total Samples)')
    
    plt.tight_layout()
    plt.savefig('aug_pie_distribution.png')
    plt.close()

def print_statistics(total_samples):
    print("\nDataset Statistics:")
    print("-" * 50)
    print(f"Total number of samples: {sum(total_samples.values())}")
    print(f"Average samples per class: {np.mean(list(total_samples.values())):.1f}")
    print(f"Min samples in a class: {min(total_samples.values())}")
    print(f"Max samples in a class: {max(total_samples.values())}")
    print("-" * 50)

def main():
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fruits_data')
    class_counts, total_samples = count_samples(root_dir)
    
    # Generate separate visualizations
    plot_bar_distribution(total_samples)
    plot_heatmap(class_counts)
    plot_pie_chart(total_samples)
    print_statistics(total_samples)
    
    print("\nVisualization files generated:")
    print("1. bar_distribution.png - Bar plot of samples per class")
    print("2. heatmap_distribution.png - Heatmap of country-fruit distribution")
    print("3. pie_distribution.png - Pie chart of class proportions")

if __name__ == "__main__":
    main()