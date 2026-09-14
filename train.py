import sys
from training.train_pipeline import train_and_evaluate_all
from utils.logger import get_logger

logger = get_logger("TrainCLI")

def main():
    logger.info("Initializing complete Machine Learning training pipeline via CLI...")
    try:
        metrics = train_and_evaluate_all()
        logger.info("Pipeline execution completed successfully!")
        print("\n" + "="*50)
        print("SUMMARY METRICS:")
        print(f"Champion Model: {metrics['best_model_name']}")
        print(f"F1 Score:       {metrics['best_model_f1']}")
        print(f"Accuracy:       {metrics['best_model_accuracy']}")
        print(f"ROC AUC:        {metrics['best_model_roc_auc']}")
        print("="*50 + "\n")
    except Exception as e:
        logger.error(f"Training pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
