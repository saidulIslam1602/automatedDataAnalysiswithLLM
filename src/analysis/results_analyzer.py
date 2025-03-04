import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import logging
from datetime import datetime
from models.image_analyzer import DeepImageAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self, output_dir: str = "results/analysis"):
        """Initialize the results analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.figures_dir = os.path.join(output_dir, f"figures_{self.timestamp}")
        self.metrics_dir = os.path.join(output_dir, f"metrics_{self.timestamp}")
        
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def analyze_training_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training results and generate comprehensive figures and metrics.
        
        Args:
            results: Training results dictionary containing metrics
            
        Returns:
            Dict[str, Any]: Analysis results and paths to generated figures
        """
        analysis_results = {
            "figures": {},
            "metrics": {},
            "summary": {}
        }
        
        # Analyze cross-validation results
        if "cross_validation" in results:
            cv_figures = self._analyze_cross_validation(results["cross_validation"])
            analysis_results["figures"].update(cv_figures)
        
        # Analyze hyperparameter optimization
        if "hyperparameter_optimization" in results:
            hp_figures = self._analyze_hyperparameter_optimization(
                results["hyperparameter_optimization"]
            )
            analysis_results["figures"].update(hp_figures)
        
        # Analyze training and evaluation metrics
        train_eval_figures = self._analyze_train_eval_metrics(
            results["train_results"],
            results["eval_results"]
        )
        analysis_results["figures"].update(train_eval_figures)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics(results)
        analysis_results["summary"] = summary
        
        # Save analysis results
        self._save_analysis_results(analysis_results)
        
        return analysis_results
    
    def _analyze_cross_validation(self, cv_results: Dict[str, Any]) -> Dict[str, str]:
        """Analyze cross-validation results and generate figures."""
        figures = {}
        
        # Create CV metrics distribution plot
        plt.figure(figsize=(12, 6))
        metrics_data = []
        for metric, values in cv_results.items():
            metrics_data.append({
                'Metric': metric,
                'Mean': values['mean'],
                'Std': values['std']
            })
        
        df = pd.DataFrame(metrics_data)
        
        # Plot mean metrics with error bars
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            df['Metric'],
            df['Mean'],
            yerr=df['Std'],
            fmt='o',
            capsize=5
        )
        plt.title('Cross-Validation Metrics (Mean ± Std)')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        # Save figure
        fig_path = os.path.join(self.figures_dir, 'cv_metrics_distribution.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        figures['cv_metrics_distribution'] = fig_path
        return figures
    
    def _analyze_hyperparameter_optimization(
        self,
        hp_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Analyze hyperparameter optimization results and generate figures."""
        figures = {}
        
        # Extract trial data
        trials_data = pd.DataFrame(hp_results['all_trials'])
        
        # Plot parameter importance
        plt.figure(figsize=(12, 6))
        for param in trials_data['params'][0].keys():
            param_values = [t['params'][param] for t in hp_results['all_trials']]
            scores = [t['value'] for t in hp_results['all_trials']]
            
            plt.figure(figsize=(8, 5))
            plt.scatter(param_values, scores, alpha=0.6)
            plt.xlabel(param)
            plt.ylabel('Score (F1)')
            plt.title(f'Parameter Impact: {param}')
            
            # Save figure
            fig_path = os.path.join(self.figures_dir, f'param_impact_{param}.png')
            plt.savefig(fig_path, bbox_inches='tight')
            plt.close()
            
            figures[f'param_impact_{param}'] = fig_path
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        plt.plot(trials_data['number'], trials_data['value'], 'b-', label='Trial Score')
        plt.axhline(y=hp_results['best_score'], color='r', linestyle='--', label='Best Score')
        plt.xlabel('Trial Number')
        plt.ylabel('Score (F1)')
        plt.title('Hyperparameter Optimization History')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        fig_path = os.path.join(self.figures_dir, 'optimization_history.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        figures['optimization_history'] = fig_path
        return figures
    
    def _analyze_train_eval_metrics(
        self,
        train_results: Dict[str, Any],
        eval_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Analyze training and evaluation metrics and generate figures."""
        figures = {}
        
        # Combine train and eval metrics
        metrics = {
            'Training': train_results,
            'Evaluation': eval_results
        }
        
        # Create comparison bar plot
        plt.figure(figsize=(12, 6))
        metric_names = list(train_results.keys())
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x - width/2, [train_results[m] for m in metric_names], width, label='Training')
        plt.bar(x + width/2, [eval_results[m] for m in metric_names], width, label='Evaluation')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Training vs Evaluation Metrics')
        plt.xticks(x, metric_names, rotation=45)
        plt.legend()
        plt.grid(True)
        
        # Save figure
        fig_path = os.path.join(self.figures_dir, 'train_eval_comparison.png')
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
        
        figures['train_eval_comparison'] = fig_path
        return figures
    
    def _generate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from all results."""
        summary = {
            "best_model_performance": {
                "accuracy": results["eval_results"]["eval_accuracy"],
                "precision": results["eval_results"]["eval_precision"],
                "recall": results["eval_results"]["eval_recall"],
                "f1": results["eval_results"]["eval_f1"]
            },
            "cross_validation_stability": {
                metric: {
                    "mean": results["cross_validation"][metric]["mean"],
                    "std": results["cross_validation"][metric]["std"],
                    "coefficient_of_variation": (
                        results["cross_validation"][metric]["std"] /
                        results["cross_validation"][metric]["mean"]
                    )
                }
                for metric in results["cross_validation"]
            }
        }
        
        if "hyperparameter_optimization" in results:
            summary["optimization"] = {
                "best_parameters": results["hyperparameter_optimization"]["best_params"],
                "best_score": results["hyperparameter_optimization"]["best_score"],
                "n_trials": len(results["hyperparameter_optimization"]["all_trials"])
            }
        
        return summary
    
    def _save_analysis_results(self, analysis_results: Dict[str, Any]) -> None:
        """Save analysis results to file."""
        # Save summary metrics
        metrics_path = os.path.join(self.metrics_dir, "analysis_summary.json")
        with open(metrics_path, 'w') as f:
            json.dump(analysis_results["summary"], f, indent=2)
        
        # Create a report with all figure paths
        report = {
            "timestamp": self.timestamp,
            "figures": analysis_results["figures"],
            "metrics_file": metrics_path
        }
        
        report_path = os.path.join(self.output_dir, f"analysis_report_{self.timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Analysis results saved to {self.output_dir}")
        logger.info(f"Generated {len(analysis_results['figures'])} figures")
        logger.info(f"Full report available at: {report_path}")

analyzer = DeepImageAnalyzer()
analysis = analyzer.analyze_image(image)

print(f"Content Type: {analysis.content_type}")
print(f"Description: {analysis.description}")
print(f"Confidence: {analysis.confidence:.2f}") 