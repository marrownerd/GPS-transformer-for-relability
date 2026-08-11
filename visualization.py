%%writefile visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

class ExpertVisualizer:
    @staticmethod
    def plot_dashboard(history, results):
        sns.set(style="whitegrid")
        fig, axes = plt.subplots(2, 4, figsize=(26, 12))
        fig.suptitle('Zero-Shot Reliability Analysis (Trained on ZINC Molecules)', fontsize=22, fontweight='bold')

        axes[0,0].plot(history['base']['val_loss'], label='Baseline GIN', color='#e74c3c', lw=2, ls='--')
        axes[0,0].plot(history['gated']['val_loss'], label='SOTA GPS', color='#2ecc71', lw=3)
        axes[0,0].set_title("Training Convergence (KL Loss)", fontsize=14)
        axes[0,0].legend()

        axes[0,1].scatter(results['gated']['mu_true'], results['gated']['mu_pred'], alpha=0.3, color='#2ecc71', s=10, label='SOTA GPS')
        axes[0,1].plot([0,1], [0,1], 'k--', alpha=0.6)
        axes[0,1].set_title("Reliability $\mu$ Parity Plot", fontsize=14)
        axes[0,1].set_xlabel("True Value")
        axes[0,1].set_ylabel("Predicted Value")

        residuals = np.array(results['gated']['mu_pred']) - np.array(results['gated']['mu_true'])
        axes[0,2].scatter(results['gated']['mu_true'], residuals, alpha=0.3, color='#3498db', s=10)
        axes[0,2].axhline(0, color='black', lw=1)
        axes[0,2].set_title("Residual Plot (Error vs True)", fontsize=14)

        err_b = np.abs(np.array(results['base']['mu_pred']) - np.array(results['base']['mu_true']))
        err_g = np.abs(np.array(results['gated']['mu_pred']) - np.array(results['gated']['mu_true']))
        t_stat, p_val = stats.ttest_rel(err_b, err_g)
        sns.kdeplot(err_b, ax=axes[0,3], label='Baseline', color='#e74c3c', fill=True)
        sns.kdeplot(err_g, ax=axes[0,3], label='PINN', color='#2ecc71', fill=True)
        axes[0,3].set_title(f"Error Density (T-test p: {p_val:.2e})", fontsize=14)
        axes[0,3].legend()

        true_mu = np.array(results['gated']['mu_true'])
        pred_mu = np.array(results['gated']['mu_pred'])
        pred_std = np.sqrt(np.array(results['gated']['var_pred']))
        idx = np.argsort(true_mu)
        axes[1,0].errorbar(np.arange(len(idx))[:100], pred_mu[idx][:100], yerr=3*pred_std[idx][:100], fmt='o', color='#2ecc71', ecolor='gray', alpha=0.5, label='Pred $\mu \pm 3\sigma$')
        axes[1,0].scatter(np.arange(len(idx))[:100], true_mu[idx][:100], color='red', s=5, zorder=5, label='True $\mu$')
        axes[1,0].set_title("Confidence Interval ($3\sigma$)", fontsize=14)
        axes[1,0].legend()

        axes[1,1].scatter(results['gated']['var_true'], results['gated']['var_pred'], alpha=0.4, color='#9b59b6', s=10)
        axes[1,1].plot([min(results['gated']['var_true']), max(results['gated']['var_true'])], [min(results['gated']['var_true']), max(results['gated']['var_true'])], 'k--', alpha=0.6)
        axes[1,1].set_title("Variance $\sigma^2$ Calibration", fontsize=14)
        axes[1,1].set_yscale('log')
        axes[1,1].set_xscale('log')

        mae_b, rmse_b = np.mean(err_b), np.sqrt(np.mean(err_b**2))
        mae_g, rmse_g = np.mean(err_g), np.sqrt(np.mean(err_g**2))
        sns.barplot(x=['MAE Base', 'MAE PINN', 'RMSE Base', 'RMSE PIN'], y=[mae_b, mae_g, rmse_b, rmse_g], ax=axes[1,2], palette=['#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71'])
        axes[1,2].set_title("Accuracy Comparison", fontsize=14)

        axes[1,3].axis('off')
        r2_b = r2_score(results['base']['mu_true'], results['base']['mu_pred'])
        r2_g = r2_score(results['gated']['mu_true'], results['gated']['mu_pred'])
        pearson_b, _ = stats.pearsonr(results['base']['mu_true'], results['base']['mu_pred'])
        pearson_g, _ = stats.pearsonr(results['gated']['mu_true'], results['gated']['mu_pred'])

        coverage = np.mean(np.abs(pred_mu - true_mu) <= 3 * pred_std) * 100
        stats_text = (f"Baseline GIN:\n- MAE: {mae_b:.5f}\n- RMSE: {rmse_b:.5f}\n- R2: {r2_b:.4f}\n- Pearson r: {pearson_b:.4f}\n\n"
                      f"SOTA GPS:\n- MAE: {mae_g:.5f}\n- RMSE: {rmse_g:.5f}\n- R2: {r2_g:.4f}\n- Pearson r: {pearson_g:.4f}\n\n"
                      f"Improvement: {((mae_b-mae_g)/mae_b)*100:.1f}%\n3σ Coverage: {coverage:.1f}%")
        axes[1,3].text(0.1, 0.5, stats_text, fontsize=15, family='monospace', fontweight='bold', va='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("final_scientific_dashboard.png", dpi=250)
        plt.show()