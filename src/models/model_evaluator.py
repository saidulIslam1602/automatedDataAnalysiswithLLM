from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import KFold
import torch
from transformers import TrainingArguments, Trainer
import logging
import optuna
from datasets import Dataset
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, output_dir: str = "models/evaluation"):
        """Initialize model evaluator.
        
        Args:
            output_dir (str): Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def compute_metrics(self, pred) -> Dict[str, float]:
        """Compute evaluation metrics for model predictions.
        
        Args:
            pred: Model predictions
            
        Returns:
            Dict[str, float]: Dictionary of metric scores
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def cross_validate(
        self,
        model_init,
        dataset: Dataset,
        n_splits: int = 5,
        training_args: TrainingArguments = None
    ) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation.
        
        Args:
            model_init: Function to initialize model
            dataset: Training dataset
            n_splits: Number of CV folds
            training_args: Training arguments
            
        Returns:
            Dict[str, List[float]]: Cross-validation scores
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            logger.info(f"Starting fold {fold + 1}/{n_splits}")
            
            # Split dataset
            train_dataset = dataset.select(train_idx)
            val_dataset = dataset.select(val_idx)
            
            # Initialize trainer
            trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=self.compute_metrics
            )
            
            # Train and evaluate
            trainer.train()
            metrics = trainer.evaluate()
            
            # Store scores
            for metric in cv_scores.keys():
                cv_scores[metric].append(metrics[f'eval_{metric}'])
        
        # Calculate mean and std for each metric
        cv_summary = {
            metric: {
                'mean': np.mean(scores),
                'std': np.std(scores)
            }
            for metric, scores in cv_scores.items()
        }
        
        # Save CV results
        self._save_cv_results(cv_summary)
        
        return cv_summary
    
    def optimize_hyperparameters(
        self,
        model_init,
        dataset: Dataset,
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna.
        
        Args:
            model_init: Function to initialize model
            dataset: Training dataset
            n_trials: Number of optimization trials
            
        Returns:
            Dict[str, Any]: Best hyperparameters and scores
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
                'num_epochs': trial.suggest_int('num_epochs', 2, 5),
                'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
                'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.1)
            }
            
            # Create training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output_dir, f"trial_{trial.number}"),
                learning_rate=params['learning_rate'],
                per_device_train_batch_size=params['batch_size'],
                num_train_epochs=params['num_epochs'],
                warmup_ratio=params['warmup_ratio'],
                weight_decay=params['weight_decay'],
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1"
            )
            
            # Perform cross-validation
            cv_scores = self.cross_validate(
                model_init,
                dataset,
                n_splits=3,  # Use fewer splits for optimization
                training_args=training_args
            )
            
            # Return mean F1 score
            return cv_scores['f1']['mean']
        
        # Create study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and scores
        best_params = study.best_params
        best_value = study.best_value
        
        # Save optimization results
        self._save_optimization_results(best_params, best_value, study.trials)
        
        return {
            'best_params': best_params,
            'best_score': best_value
        }
    
    def _save_cv_results(self, cv_summary: Dict[str, Dict[str, float]]) -> None:
        """Save cross-validation results."""
        output_path = os.path.join(self.output_dir, "cv_results.json")
        with open(output_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        logger.info(f"Saved cross-validation results to {output_path}")
    
    def _save_optimization_results(
        self,
        best_params: Dict[str, Any],
        best_value: float,
        trials: List[optuna.Trial]
    ) -> None:
        """Save hyperparameter optimization results."""
        results = {
            'best_parameters': best_params,
            'best_score': best_value,
            'all_trials': [
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value
                }
                for t in trials
            ]
        }
        
        output_path = os.path.join(self.output_dir, "optimization_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved optimization results to {output_path}") 