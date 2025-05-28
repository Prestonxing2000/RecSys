import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir='evaluation_results'):
    """Load results from all models"""
    results = {}
    
    for model_name in ['mvke', 'mmoe', 'shared_bottom']:
        model_dir = os.path.join(results_dir, model_name)
        if os.path.exists(model_dir):
            results_file = os.path.join(model_dir, 'test_results.pkl')
            if os.path.exists(results_file):
                with open(results_file, 'rb') as f:
                    results[model_name] = pickle.load(f)
    
    return results

def create_comparison_table(results):
    """Create comparison table of metrics"""
    data = []
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        row = {'Model': model_name.upper()}
        
        for metric_name, value in metrics.items():
            if 'auc' in metric_name or 'f1' in metric_name:
                row[metric_name.upper()] = f"{value:.4f}"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def plot_comparison(results):
    """Create comparison plots"""
    # Extract AUC scores
    models = []
    ctr_aucs = []
    cvr_aucs = []
    
    for model_name, model_results in results.items():
        models.append(model_name.upper())
        ctr_aucs.append(model_results['metrics']['ctr_auc'])
        cvr_aucs.append(model_results['metrics']['cvr_auc'])
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(models))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], ctr_aucs, width, label='CTR AUC', alpha=0.8)
    ax.bar([i + width/2 for i in x], cvr_aucs, width, label='CVR AUC', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('AUC Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (ctr, cvr) in enumerate(zip(ctr_aucs, cvr_aucs)):
        ax.text(i - width/2, ctr + 0.005, f'{ctr:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, cvr + 0.005, f'{cvr:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()

def main():
    print("Loading evaluation results...")
    results = load_results()
    
    if not results:
        print("No evaluation results found. Please run evaluate.py first.")
        return
    
    print(f"\nFound results for {len(results)} models: {list(results.keys())}")
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Save table
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\nComparison table saved to model_comparison.csv")
    
    # Create plots
    plot_comparison(results)
    print("\nComparison plot saved to model_comparison.png")
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    # Find best model for each task
    best_ctr_model = max(results.items(), key=lambda x: x[1]['metrics']['ctr_auc'])
    best_cvr_model = max(results.items(), key=lambda x: x[1]['metrics']['cvr_auc'])
    
    print(f"\nBest model for CTR: {best_ctr_model[0].upper()} (AUC: {best_ctr_model[1]['metrics']['ctr_auc']:.4f})")
    print(f"Best model for CVR: {best_cvr_model[0].upper()} (AUC: {best_cvr_model[1]['metrics']['cvr_auc']:.4f})")
    
    # Calculate improvements
    if 'shared_bottom' in results:
        baseline = results['shared_bottom']
        print("\nImprovement over Shared Bottom baseline:")
        
        for model_name in ['mmoe', 'mvke']:
            if model_name in results:
                ctr_improvement = (results[model_name]['metrics']['ctr_auc'] - baseline['metrics']['ctr_auc']) / baseline['metrics']['ctr_auc'] * 100
                cvr_improvement = (results[model_name]['metrics']['cvr_auc'] - baseline['metrics']['cvr_auc']) / baseline['metrics']['cvr_auc'] * 100
                
                print(f"\n{model_name.upper()}:")
                print(f"  CTR AUC improvement: {ctr_improvement:+.2f}%")
                print(f"  CVR AUC improvement: {cvr_improvement:+.2f}%")

if __name__ == '__main__':
    main()