"""
Visualization utilities for training analysis
"""

import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=50):
    """Calculate moving average of data"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def create_visualizations(rewards_history, final_states, final_actions, final_rewards):
    """Create comprehensive training visualizations"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Training Progress
    ax1 = plt.subplot(3, 3, 1)
    episodes = range(len(rewards_history))
    ax1.plot(episodes, rewards_history, 'b-', alpha=0.3, label='Raw Reward')
    
    # Add smoothed line
    if len(rewards_history) >= 50:
        smoothed = moving_average(rewards_history, 50)
        ax1.plot(range(49, len(rewards_history)), smoothed, 'r-', linewidth=2, label='Smoothed (50 episodes)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(rewards_history, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(np.mean(rewards_history), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(rewards_history):.3f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Final Episode Actions
    ax3 = plt.subplot(3, 3, 3)
    action_labels = ['cp1', 'cp2', 'cp3', 'Re']
    action_means = final_actions.mean(axis=0)
    action_stds = final_actions.std(axis=0)
    
    bars = ax3.bar(action_labels, action_means, yerr=action_stds, 
                   capsize=5, alpha=0.7, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    ax3.set_ylabel('Action Value')
    ax3.set_title('Final Batch Actions')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, action_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{mean:.2f}', ha='center', va='bottom')
    
    # 4. Sc vs Reward Scatter
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(final_states.flatten(), final_rewards, 
                         c=final_rewards, cmap='viridis', alpha=0.6)
    ax4.set_xlabel('Schmidt Number (Sc)')
    ax4.set_ylabel('Reward')
    ax4.set_title('Sc vs Reward')
    plt.colorbar(scatter, ax=ax4, label='Reward')
    ax4.grid(True, alpha=0.3)
    
    # 5. Action Correlation Heatmap
    ax5 = plt.subplot(3, 3, 5)
    action_corr = np.corrcoef(final_actions.T)
    im = ax5.imshow(action_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_xticks(range(len(action_labels)))
    ax5.set_yticks(range(len(action_labels)))
    ax5.set_xticklabels(action_labels)
    ax5.set_yticklabels(action_labels)
    ax5.set_title('Action Correlation Matrix')
    
    # Add correlation values to heatmap
    for i in range(len(action_labels)):
        for j in range(len(action_labels)):
            ax5.text(j, i, f'{action_corr[i,j]:.2f}', 
                    ha='center', va='center', color='white' if abs(action_corr[i,j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax5)
    
    # 6. Learning Curve with Percentiles
    ax6 = plt.subplot(3, 3, 6)
    if len(rewards_history) >= 100:
        window = 100
        percentiles = []
        for i in range(len(rewards_history) - window + 1):
            window_data = rewards_history[i:i+window]
            percentiles.append([np.percentile(window_data, p) for p in [25, 50, 75]])
        
        percentiles = np.array(percentiles)
        x_range = range(window-1, len(rewards_history))
        
        ax6.fill_between(x_range, percentiles[:, 0], percentiles[:, 2], alpha=0.3, color='blue', label='25-75% Percentile')
        ax6.plot(x_range, percentiles[:, 1], 'b-', linewidth=2, label='Median')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Reward')
        ax6.set_title('Reward Distribution Over Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Action Distribution per Parameter
    ax7 = plt.subplot(3, 3, 7)
    action_data = [final_actions[:, i] for i in range(4)]
    ax7.boxplot(action_data, labels=action_labels)
    ax7.set_ylabel('Action Value')
    ax7.set_title('Action Value Distributions')
    ax7.grid(True, alpha=0.3)
    
    # 8. Cumulative Max Reward
    ax8 = plt.subplot(3, 3, 8)
    cumulative_max = np.maximum.accumulate(rewards_history)
    ax8.plot(episodes, cumulative_max, 'g-', linewidth=2)
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Cumulative Max Reward')
    ax8.set_title('Best Performance Over Time')
    ax8.grid(True, alpha=0.3)
    
    # 9. Training Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_text = f"""
    Training Statistics:
    --------------------
    Total Episodes: {len(rewards_history)}
    Final Average Reward: {np.mean(rewards_history[-100:]):.4f}
    Best Reward: {np.max(rewards_history):.4f}
    Mean Reward: {np.mean(rewards_history):.4f}
    Std Reward: {np.std(rewards_history):.4f}
    
    Final Batch:
    Mean Sc: {np.mean(final_states):.2f}
    Mean Reward: {np.mean(final_rewards):.4f}
    """
    
    ax9.text(0.1, 0.9, stats_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()