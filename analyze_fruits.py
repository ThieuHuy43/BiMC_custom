import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def analyze_fruits_directory(root_dir):
    # Initialize counters
    countries = defaultdict(lambda: defaultdict(int))
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' not found!")
        return None
        
    for dir_name in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, dir_name)):
            if '_' in dir_name:
                country, fruit = dir_name.split('_')
            elif '-' in dir_name:
                country, fruit = dir_name.split('-')
            else:
                continue
                
            country = country.lower()
            if country in ['chinese', 'china']:
                country = 'China'
            elif country in ['vietnamese', 'vietnam']:
                country = 'Vietnam'
            else:
                country = country.title()
                
            countries[country][fruit.title()] += 1
            
    return countries

def create_visualizations(data):
    # Convert data to plottable format
    countries = list(data.keys())
    fruits = set()
    for fruit_dict in data.values():
        fruits.update(fruit_dict.keys())
    fruits = list(fruits)

    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 1. Bar Plot
    x = np.arange(len(countries))
    width = 0.8 / len(fruits)
    
    for i, fruit in enumerate(fruits):
        counts = [data[country].get(fruit, 0) for country in countries]
        ax1.bar(x + i * width, counts, width, label=fruit)
    
    ax1.set_xlabel('Countries')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Fruits by Country')
    ax1.set_xticks(x + width * (len(fruits)-1)/2)
    ax1.set_xticklabels(countries, rotation=45)
    ax1.legend()

    # 2. Heatmap
    matrix = np.zeros((len(countries), len(fruits)))
    for i, country in enumerate(countries):
        for j, fruit in enumerate(fruits):
            matrix[i][j] = data[country].get(fruit, 0)

    im = ax2.imshow(matrix, cmap='YlOrRd')
    
    # Add text annotations
    for i in range(len(countries)):
        for j in range(len(fruits)):
            text = ax2.text(j, i, int(matrix[i, j]),
                          ha="center", va="center", color="black")

    ax2.set_xticks(np.arange(len(fruits)))
    ax2.set_yticks(np.arange(len(countries)))
    ax2.set_xticklabels(fruits)
    ax2.set_yticklabels(countries)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_title('Fruits Distribution Heatmap')

    # Add colorbar
    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.savefig('fruits_visualization.png')
    plt.show()

def main():
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fruits_data')
    data = analyze_fruits_directory(root_dir)
    
    if data:
        create_visualizations(data)
        print("Visualizations saved as 'fruits_visualization.png'")

if __name__ == "__main__":
    main()