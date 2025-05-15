import imageio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


fps = 20

df = pd.DataFrame({
    'model': ['OpenVLA'] * 2 + ['Octo'] * 2 + ['$\pi_0$'] * 2,
    'mode': ['Baseline', 'PCD'] * 3,
    'success_rate': [0.17, 0.24, 0.16, 0.21, 0.65, 0.70],
})

df['success_rate'] = df['success_rate'].apply(lambda x: x - 0.3 if x > 0.5 else x)

# make an animation for the plot
# 1. no bars in plot
# 2. baseline bar glows
# 3. point, box, and gdino bars glow
def animate(i):
    df_ = df.copy(deep=True)
    
    if i < fps:
        df_['success_rate'] = 0
    elif i < 2 * fps:
        # only keep baseline bar
        df_.loc[df_['mode'] != 'Baseline', 'success_rate'] = 0
        df_.loc[df_['mode'] == 'Baseline', 'success_rate'] -= 0.1
        df_.loc[df_['mode'] == 'Baseline', 'success_rate'] *= (i - fps) / fps
        df_.loc[df_['mode'] == 'Baseline', 'success_rate'] += 0.1
    elif i < 3 * fps:
        df_.loc[df_['mode'] != 'Baseline', 'success_rate'] -= 0.1
        df_.loc[df_['mode'] != 'Baseline', 'success_rate'] *= (i - 2 * fps) / fps
        df_.loc[df_['mode'] != 'Baseline', 'success_rate'] += 0.1

    # dodge bar plot
    sns.set_theme(style="whitegrid")
    sns.set_palette(['#bccbea', '#e3bfb0'])

    plt.figure(figsize=(6, 2))
    sns.barplot(x='model', y='success_rate', hue='mode', data=df_, dodge=True)

    if i >= 3 * fps:
        alpha = (i - 3 * fps) / fps
        if alpha > 1:
            alpha = 1
            
        df__ = df_[df_['mode'] == 'Baseline']
        for j in range(len(df__)):
            value = df__['success_rate'].iloc[j]
            if value > 0.3:
                value += 0.3
            plt.text(j - 0.2, df__['success_rate'].iloc[j] + 0.001, f"{value:.2f}", ha='center', va='bottom', fontsize=10, alpha=alpha)

        df__ = df_[df_['mode'] == 'PCD']
        for j in range(len(df__)):
            value = df__['success_rate'].iloc[j]
            if value > 0.3:
                value += 0.3
            plt.text(j + 0.2, df__['success_rate'].iloc[j] + 0.001, f"{value:.2f}", ha='center', va='bottom', fontsize=10, alpha=alpha)
    
    plt.xlabel('')
    plt.ylabel('Success Rate')
    plt.yticks([0.1, 0.2, 0.3, 0.4], ['0.1', '0.2', '0.6', '0.7'])
    plt.ylim(0.1, 0.43)
    plt.legend(title='', ncol=2, loc='upper left')
    plt.savefig(f'plot_simpler/{i}.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
    plt.close()
    return f'plot_simpler/{i}.png'



plot_paths = []
for i in tqdm(range(0, 100, 1)):
    plot_paths.append(animate(i))

frames = [imageio.imread(path) for path in plot_paths]
imageio.mimsave('plot_simpler.mp4', frames, fps=fps)