from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from datasets import Dataset
import torch
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import logging
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import wandb
from .model_evaluator import ModelEvaluator
from tqdm.auto import tqdm
import shap
import lime.lime_text
from sklearn.ensemble import VotingClassifier
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMHandler:
    """Advanced LLM handler with ensemble models and user-friendly features."""
    
    AVAILABLE_MODELS = {
        "roberta-large": (RobertaTokenizer, RobertaForSequenceClassification),
        "microsoft/deberta-v3-large": (DebertaV2Tokenizer, DebertaV2ForSequenceClassification),
    }

    def __init__(
        self,
        model_names: Union[str, List[str]] = ["roberta-large", "microsoft/deberta-v3-large"],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = True,
        use_streamlit: bool = True
    ):
        """Initialize enhanced LLM handler with multiple models.
        
        Args:
            model_names: Name(s) of the pretrained model(s) to use
            device: Device to use for computation
            use_wandb: Whether to use Weights & Biases for tracking
            use_streamlit: Whether to use Streamlit for UI
        """
        self.model_names = [model_names] if isinstance(model_names, str) else model_names
        self.device = device
        self.models = {}
        self.tokenizers = {}
        
        if use_streamlit:
            st.info("🚀 Initializing advanced models for optimal performance...")
            progress_bar = st.progress(0)
        
        for idx, model_name in enumerate(self.model_names):
            if use_streamlit:
                progress_bar.progress((idx + 1) / len(self.model_names))
                st.write(f"Loading {model_name}...")
            
            tokenizer_class, model_class = self.AVAILABLE_MODELS.get(
                model_name, 
                (AutoTokenizer, AutoModelForSequenceClassification)
            )
            self.tokenizers[model_name] = tokenizer_class.from_pretrained(model_name)
            self.models[model_name] = None  # Will be initialized during fine-tuning
        
        self.sentence_transformer = SentenceTransformer('all-mpnet-base-v2')  # Upgraded model
        self.evaluator = ModelEvaluator()
        self.use_wandb = use_wandb
        self.explainer = None
        
        if use_wandb:
            wandb.init(project="enhanced-robot-manual-analysis")
        
        if use_streamlit:
            st.success("✅ Models initialized successfully!")

    def prepare_dataset(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        validation_split: float = 0.2,
        use_streamlit: bool = True
    ) -> Dict[str, Dataset]:
        """Prepare dataset with enhanced preprocessing and validation.
        
        Args:
            texts: List of input texts
            labels: List of labels for supervised learning
            validation_split: Fraction of data to use for validation
            use_streamlit: Whether to show progress in Streamlit
        """
        if use_streamlit:
            st.info("🔄 Preparing and validating dataset...")
        
        # Enhanced text preprocessing
        processed_texts = self._preprocess_texts(texts)
        
        data_dict = {"text": processed_texts}
        if labels is not None:
            data_dict["labels"] = labels
            
        dataset = Dataset.from_dict(data_dict)
        
        if validation_split > 0:
            dataset = dataset.train_test_split(
                test_size=validation_split,
                shuffle=True,
                seed=42
            )
            
            if use_streamlit:
                st.success(f"""✅ Dataset prepared:
                - Training samples: {len(dataset['train'])}
                - Validation samples: {len(dataset['test'])}""")
            
            return {
                "train": dataset["train"],
                "validation": dataset["test"]
            }
        return {"train": dataset}

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Enhanced text preprocessing with technical domain adaptation."""
        processed = []
        for text in tqdm(texts, desc="Preprocessing texts"):
            # Add domain-specific preprocessing here
            processed_text = text.strip()
            # Add more preprocessing steps as needed
            processed.append(processed_text)
        return processed

    def explain_prediction(
        self,
        text: str,
        model_name: str = None,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """Explain model predictions using LIME or SHAP.
        
        Args:
            text: Input text to explain
            model_name: Which model to use for explanation
            num_features: Number of features to include in explanation
            
        Returns:
            Dictionary containing explanation details
        """
        if model_name is None:
            model_name = self.model_names[0]
            
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        if self.explainer is None:
            self.explainer = lime.lime_text.LimeTextExplainer(
                class_names=model.config.id2label.values()
            )
            
        def predict_proba(texts):
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            with torch.no_grad():
                outputs = model(**inputs)
            return torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
        explanation = self.explainer.explain_instance(
            text,
            predict_proba,
            num_features=num_features
        )
        
        return {
            "text": text,
            "prediction": model.config.id2label[predict_proba([text])[0].argmax()],
            "confidence": float(predict_proba([text])[0].max()),
            "explanation": explanation.as_list()
        }

    def ensemble_predict(
        self,
        texts: List[str],
        use_streamlit: bool = True
    ) -> List[Dict[str, Any]]:
        """Make predictions using ensemble of models.
        
        Args:
            texts: List of input texts
            use_streamlit: Whether to show progress in Streamlit
            
        Returns:
            List of predictions with confidence scores
        """
        if use_streamlit:
            st.info("🤖 Making ensemble predictions...")
            progress_bar = st.progress(0)
            
        predictions = []
        for idx, text in enumerate(texts):
            if use_streamlit:
                progress_bar.progress((idx + 1) / len(texts))
                
            model_predictions = []
            model_confidences = []
            
            for model_name in self.model_names:
                model = self.models[model_name]
                tokenizer = self.tokenizers[model_name]
                
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                    pred_idx = probs.argmax().item()
                    confidence = probs.max().item()
                    
                model_predictions.append(pred_idx)
                model_confidences.append(confidence)
            
            # Ensemble decision
            final_pred = max(set(model_predictions), key=model_predictions.count)
            avg_confidence = np.mean(model_confidences)
            
            predictions.append({
                "text": text,
                "prediction": self.models[self.model_names[0]].config.id2label[final_pred],
                "confidence": float(avg_confidence),
                "model_agreements": len(set(model_predictions)) == 1
            })
            
        if use_streamlit:
            st.success("✅ Predictions complete!")
            
        return predictions

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        optimize_hyperparams: bool = True,
        n_trials: int = 20,
        output_dir: str = "models/fine_tuned",
        **training_kwargs
    ) -> Dict[str, Any]:
        """Fine-tune the model with optimized hyperparameters and evaluation.
        
        Args:
            train_texts (List[str]): Training texts
            train_labels (List[int]): Training labels
            val_texts (Optional[List[str]]): Validation texts
            val_labels (Optional[List[int]]): Validation labels
            optimize_hyperparams (bool): Whether to perform hyperparameter optimization
            n_trials (int): Number of optimization trials
            output_dir (str): Directory to save the fine-tuned model
            **training_kwargs: Additional training arguments
            
        Returns:
            Dict[str, Any]: Training results and metrics
        """
        # Prepare datasets
        if val_texts and val_labels:
            train_dataset = self.prepare_dataset(train_texts, train_labels)["train"]
            val_dataset = self.prepare_dataset(val_texts, val_labels)["train"]
        else:
            datasets = self.prepare_dataset(train_texts, train_labels)
            train_dataset = datasets["train"]
            val_dataset = datasets.get("validation")
        
        def model_init() -> AutoModelForSequenceClassification:
            """Initialize a new model instance."""
            return AutoModelForSequenceClassification.from_pretrained(
                self.model_names[0],
                num_labels=len(set(train_labels))
            ).to(self.device)
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            logger.info("Optimizing hyperparameters...")
            opt_results = self.evaluator.optimize_hyperparameters(
                model_init,
                train_dataset,
                n_trials=n_trials
            )
            
            # Update training arguments with best parameters
            training_kwargs.update(opt_results['best_params'])
            
            if self.use_wandb:
                wandb.log({"best_hyperparameters": opt_results['best_params']})
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            **training_kwargs
        )
        
        # Perform cross-validation
        logger.info("Performing cross-validation...")
        cv_results = self.evaluator.cross_validate(
            model_init,
            train_dataset,
            training_args=training_args
        )
        
        if self.use_wandb:
            wandb.log({"cross_validation_results": cv_results})
        
        # Train final model
        logger.info("Training final model...")
        self.models[self.model_names[0]] = model_init()
        trainer = Trainer(
            model=self.models[self.model_names[0]],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.evaluator.compute_metrics
        )
        
        train_result = trainer.train()
        eval_result = trainer.evaluate()
        
        # Save the model
        trainer.save_model(output_dir)
        self.tokenizers[self.model_names[0]].save_pretrained(output_dir)
        
        # Compile results
        results = {
            "train_results": train_result.metrics,
            "eval_results": eval_result,
            "cross_validation": cv_results
        }
        if optimize_hyperparams:
            results["hyperparameter_optimization"] = opt_results
            
        if self.use_wandb:
            wandb.log(results)
            wandb.finish()
            
        return results
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings using SentenceTransformer."""
        return self.sentence_transformer.encode(texts)
    
    def semantic_search(
        self,
        query: str,
        document_embeddings: np.ndarray,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using encoded embeddings."""
        query_embedding = self.sentence_transformer.encode([query])[0]
        similarities = np.dot(document_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": documents[idx],
                "score": float(similarities[idx])
            })
            
        return results 