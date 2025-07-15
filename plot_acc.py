import matplotlib.pyplot as plt
import numpy as np

def plot_task_accuracy():
    # Task accuracies from your results
    task_accuracies = [47.28, 40.64, 35.62, 28.17, 22.69, 17.54, 31.42, 30.28, 28.33, 26.79, 25.34]
    tasks = range(len(task_accuracies))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Line plot
    ax1.plot(tasks, task_accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_title('Task Accuracy Progression', fontsize=14, pad=15)
    ax1.set_xlabel('Task Number', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(tasks)
    
    # Add value labels
    for i, acc in enumerate(task_accuracies):
        ax1.annotate(f'{acc:.2f}%', 
                    (i, acc), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')

    # Bar plot
    bars = ax2.bar(tasks, task_accuracies, color='royalblue', alpha=0.7)
    ax2.set_title('Task Accuracy Distribution', fontsize=14, pad=15)
    ax2.set_xlabel('Task Number', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_xticks(tasks)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')

    # Adjust layout
    plt.tight_layout()
    
    # Save plots
    plt.savefig('task_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_task_accuracy()