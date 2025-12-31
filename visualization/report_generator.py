"""
Report Generator Module
Generates experiment reports in Markdown/PDF format
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Generates comprehensive experiment reports
    """
    
    def __init__(self, output_dir: str = "./reports"):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_markdown_report(self, 
                                 experiment_name: str,
                                 metrics_results: Dict[str, Dict],
                                 comparison_data: Optional[Dict] = None,
                                 output_filename: Optional[str] = None) -> str:
        """
        Generate a Markdown report
        
        Args:
            experiment_name: Name of the experiment
            metrics_results: Dictionary mapping method names to metric results
            comparison_data: Optional comparison data
            output_filename: Optional output filename
            
        Returns:
            Path to generated report file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"report_{experiment_name}_{timestamp}.md"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# Experiment Report: {experiment_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents the results of multi-dimensional coverage evaluation ")
            f.write("for autonomous driving scenario generation methods.\n\n")
            
            # Metrics Results
            f.write("## Metrics Results (PCE/BCE/DPE/CCE)\n\n")
            
            for method_name, results in metrics_results.items():
                f.write(f"### {method_name}\n\n")
                
                if 'pc' in results:
                    f.write(f"- **Parameter Configuration Entropy (PCE):** {results['pc']:.4f}\n")
                if 'pec' in results:
                    f.write(f"- **Behavior Category Entropy (BCE):** {results['pec']:.4f}\n")
                if 'tcd' in results:
                    f.write(f"- **Driving Pattern Entropy (DPE):** {results['tcd']:.4f}\n")
                if 'bcm' in results:
                    f.write(f"- **Combination Coverage Entropy (CCE):** {results['bcm']:.4f}\n")
                
                f.write("\n")
            
            # Comparison Section
            if comparison_data:
                f.write("## Comparison Analysis\n\n")
                f.write("### Method Comparison\n\n")
                f.write("| Method | PCE | BCE | DPE | CCE |\n")
                f.write("|--------|-----|-----|-----|-----|\n")
                
                for method_name, results in metrics_results.items():
                    pc = results.get('pc', 0.0)
                    pec = results.get('pec', 0.0)
                    tcd = results.get('tcd', 0.0)
                    bcm = results.get('bcm', 0.0)
                    f.write(f"| {method_name} | {pc:.4f} | {pec:.4f} | {tcd:.4f} | {bcm:.4f} |\n")
                
                f.write("\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("The multi-dimensional evaluation framework provides comprehensive insights ")
            f.write("into the coverage and diversity of generated test scenarios.\n\n")
        
        print(f"[ReportGenerator] Generated report: {output_path}")
        return output_path
    
    def generate_summary_json(self, metrics_results: Dict[str, Dict], 
                            output_filename: Optional[str] = None) -> str:
        """
        Generate a JSON summary of results
        
        Args:
            metrics_results: Dictionary mapping method names to metric results
            output_filename: Optional output filename
            
        Returns:
            Path to generated JSON file
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"summary_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results': metrics_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"[ReportGenerator] Generated summary: {output_path}")
        return output_path

