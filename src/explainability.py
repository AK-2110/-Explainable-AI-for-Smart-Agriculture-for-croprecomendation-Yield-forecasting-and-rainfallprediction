import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import os

class XAIExplainer:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def explain_shap_crop(self, model, X_train, X_test, feature_names):
        """
        Global explanation using SHAP for Crop Recommendation (SVM).
        Using KernelExplainer as SVM with RBF is model-agnostic here.
        Using a small background sample for speed.
        """
        print("Generating SHAP explanations for Crop Recommendation...")
        # Summarize background data using k-means to speed up
        X_train_summary = shap.kmeans(X_train, 10)
        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary)
        
        # Explain a subset of test data
        shap_values = explainer.shap_values(X_test[:10]) 
        
        # Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test[:10], feature_names=feature_names, show=False)
        plt.savefig(os.path.join(self.output_dir, "shap_crop_summary.png"), bbox_inches='tight')
        plt.close()
        
    def explain_lime_crop(self, model, X_train, X_instance, feature_names, class_names):
        """
        Local explanation using LIME for a specific crop prediction.
        """
        print("Generating LIME explanation for a single instance...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )
        
        exp = explainer.explain_instance(
            data_row=X_instance,
            predict_fn=model.predict_proba
        )
        
        # Save plot
        fig = exp.as_pyplot_figure()
        fig.savefig(os.path.join(self.output_dir, "lime_crop_instance.png"), bbox_inches='tight')
        plt.close(fig)

    # Note: SHAP/LIME for Time Series (LSTM/Transformer) is complex and often requires DeepExplainer
    # For this project scope, we focus XAI on the Tabular Crop Recommendation part as per common research constraints
    # but we can add simple Feature Importance for others if needed.
