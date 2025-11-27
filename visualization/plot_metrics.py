"""
Plot Metrics Module
Generates visualization charts for various evaluation metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import os


def plot_parameter_coverage(pc_scores: Dict[str, float], output_path: Optional[str] = None):
    """
    Plot parameter coverage as radar chart
    
    Args:
        pc_scores: Dictionary mapping method names to PC scores
        output_path: Optional path to save the figure
    """
    methods = list(pc_scores.keys())
    scores = list(pc_scores.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(methods, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Parameter Coverage (PC)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_title('Parameter Space Coverage Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_behavior_coverage(pec_data: Dict[str, Dict], output_path: Optional[str] = None):
    """
    Plot behavior coverage as bar chart showing equivalence class counts
    
    Args:
        pec_data: Dictionary mapping method names to PEC data (with 'num_classes' and 'coverage')
        output_path: Optional path to save the figure
    """
    methods = list(pec_data.keys())
    num_classes = [pec_data[m].get('num_classes', 0) for m in methods]
    coverage = [pec_data[m].get('coverage', 0.0) for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot number of equivalence classes
    ax1.bar(methods, num_classes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Number of Behavior Classes', fontsize=12)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_title('Behavior Equivalence Classes', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot coverage ratio
    ax2.bar(methods, coverage, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Coverage Ratio', fontsize=12)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_title('Behavior Coverage Ratio', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_trajectory_diversity(tcd_data: Dict[str, Dict], output_path: Optional[str] = None):
    """
    Plot trajectory diversity as pie chart and entropy curve
    
    Args:
        tcd_data: Dictionary mapping method names to TCD data (with 'entropy', 'num_clusters', 'diversity_score')
        output_path: Optional path to save the figure
    """
    methods = list(tcd_data.keys())
    entropy_values = [tcd_data[m].get('entropy', 0.0) for m in methods]
    diversity_scores = [tcd_data[m].get('diversity_score', 0.0) for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot entropy values
    ax1.bar(methods, entropy_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_title('Trajectory Pattern Entropy', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot diversity scores
    ax2.bar(methods, diversity_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Diversity Score', fontsize=12)
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_title('Trajectory Diversity Score', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_behavior_matrix(bcm_matrix: np.ndarray, behavior_labels: List[str], 
                         output_path: Optional[str] = None):
    """
    Plot behavior matrix as heatmap
    
    Args:
        bcm_matrix: Behavior-scenario matrix of shape (n_behaviors, n_scenarios)
        behavior_labels: List of behavior label names
        output_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(bcm_matrix, 
                xticklabels=[f'S{i+1}' for i in range(bcm_matrix.shape[1])],
                yticklabels=behavior_labels,
                cmap='YlOrRd',
                cbar_kws={'label': 'Behavior Present (1) / Absent (0)'},
                ax=ax)
    
    ax.set_xlabel('Scenarios', fontsize=12)
    ax.set_ylabel('Behaviors', fontsize=12)
    ax.set_title('Behavior-Scenario Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_comparison_radar(all_metrics: Dict[str, Dict[str, float]], 
                         output_path: Optional[str] = None):
    """
    Plot comparison radar chart for all metrics
    
    Args:
        all_metrics: Dictionary mapping method names to metric dictionaries
        output_path: Optional path to save the figure
    """
    # Extract metric names
    metric_names = ['PC', 'PEC', 'TCD', 'BCM']
    
    # Prepare data
    methods = list(all_metrics.keys())
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, method in enumerate(methods):
        values = []
        for metric in metric_names:
            if metric == 'PC':
                values.append(all_metrics[method].get('pc', 0.0))
            elif metric == 'PEC':
                values.append(all_metrics[method].get('pec', 0.0))
            elif metric == 'TCD':
                values.append(all_metrics[method].get('tcd', 0.0))
            elif metric == 'BCM':
                values.append(all_metrics[method].get('bcm', 0.0))
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim([0, 1])
    ax.set_title('Multi-Dimensional Coverage Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

