import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
window = 50

games = {
    'Pong': {
        'path': '/Users/kerekmen/Developer/agnostic_rl/experiments_log_data/method_bayes_metric_Evaluation_Score-PongNoFrameskip-v4_batchsize_32-64_ewclambda_75-175-1000-4000_runcap_10/pong_min_max_norm.csv',
        'first_line': 0,
        'game_order': ['Pong', 'SpaceInvaders', 'BeamRider']
    },
    'SpaceInvaders': {
        'path': '/Users/kerekmen/Developer/agnostic_rl/experiments_log_data/method_bayes_metric_Evaluation_Score-PongNoFrameskip-v4_batchsize_32-64_ewclambda_75-175-1000-4000_runcap_10/space_min_max_norm.csv',
        'first_line': 100_000,
        'game_order': ['Pong', 'SpaceInvaders', 'BeamRider']
    },
    'BeamRider': {
        'path': '/Users/kerekmen/Developer/agnostic_rl/experiments_log_data/method_bayes_metric_Evaluation_Score-PongNoFrameskip-v4_batchsize_32-64_ewclambda_75-175-1000-4000_runcap_10/beam_min_max_norm.csv',
        'first_line': 200_000,
        'game_order': ['Pong', 'SpaceInvaders', 'BeamRider']
    }
}

fig, axs = plt.subplots(1, len(games), figsize=(15, 5))

def kilo_formatter(x, pos):
    'The two args are the value and tick position'
    return f"{int(x / 1000)}k" if x >= 1000 else f"{int(x)}"

def plot_for_game(ax, df, timestep_column, score_columns, game_name, first_line, game_order, window=20):
    df['mean_score'] = df[score_columns].mean(axis=1, skipna=True)
    df['smooth_mean'] = df['mean_score'].rolling(window=window).mean()
    df['rolling_std'] = df['mean_score'].rolling(window=window).std()
    
    ax.plot(df[timestep_column], df['smooth_mean'], label='Mean Score')
    ax.fill_between(df[timestep_column], 
                    df['smooth_mean'] - 0.4*df['rolling_std'], 
                    df['smooth_mean'] + 0.4*df['rolling_std'], 
                    color='b', alpha=0.2, label='Std. Dev.')
    
    # Regression line
    clean_df = df[[timestep_column, 'smooth_mean']].dropna()
    if not clean_df[timestep_column].empty and not clean_df['smooth_mean'].empty:  # Check for non-empty vectors
        m, b = np.polyfit(clean_df[timestep_column], clean_df['smooth_mean'], 1)
        ax.plot(clean_df[timestep_column], m * clean_df[timestep_column] + b, 'y--', label='Trend')
    
    subsequent_step = 300000
    num_lines = (int(df[timestep_column].max()) - first_line) // subsequent_step + 1
    x_positions = [first_line + i * subsequent_step for i in range(num_lines)]
    
    for x in x_positions:
        ax.axvline(x, color='r', linestyle='dotted')
        
    colors = {'Pong': 'k', 'SpaceInvaders': 'b', 'BeamRider': 'y'}
    
    special_section_mean = df['smooth_mean'].max()
    
    # Drawing a single, divided horizontal line for each game in the game_order
    total_length = df[timestep_column].max()  # This goes until the end of the graph
    section_length = total_length // 9  # Dividing total length into 9 sections
    
    # Starting from 0
    current_start = 0
    
    for i in range(9):
        game = game_order[i % len(game_order)]
        ax.hlines(special_section_mean, 
                  current_start + i * section_length, 
                  current_start + (i + 1) * section_length, 
                  colors=colors[game], 
                  linestyles='-', 
                  label=f"{game} Section" if i < len(game_order) else "",
                  linewidth=2)
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Score')
    ax.set_title(f'{game_name}')
    ax.xaxis.set_major_formatter(FuncFormatter(kilo_formatter))

def plot_for_game_normalized(ax, df, timestep_column, score_columns, game_name, first_line, game_order, window=20):
    df['mean_score'] = df[score_columns].mean(axis=1, skipna=True)
    df['smooth_mean'] = df['mean_score'].rolling(window=window).mean()
    df['rolling_std'] = df['mean_score'].rolling(window=window).std()

    # Z-Score normalization
    mean_score = df['smooth_mean'].mean()
    std_score = df['smooth_mean'].std()
    df['zscore'] = (df['smooth_mean'] - mean_score) / std_score

    ax.plot(df[timestep_column], df['zscore'], label='Z-Score Normalized Score')  # Plotting Z-Score normalized scores
    ax.fill_between(df[timestep_column], 
                    df['zscore'] - 0.8 * df['rolling_std'] / std_score, 
                    df['zscore'] + 0.8 * df['rolling_std'] / std_score, 
                    color='gray', alpha=0.2, label='Std. Dev.')

    subsequent_step = 300000
    num_lines = (int(df[timestep_column].max()) - first_line) // subsequent_step + 1
    x_positions = [first_line + i * subsequent_step for i in range(num_lines)]
    for x in x_positions:
        ax.axvline(x, color='r', linestyle='--')
        
    colors = {'Pong': 'k', 'SpaceInvaders': 'b', 'BeamRider': 'y'}
    total_length = df[timestep_column].max()
    section_length = total_length // 9
    current_start = 0
    for i in range(9):
        game = game_order[i % len(game_order)]
        ax.hlines(0, 
                  current_start + i * section_length, 
                  current_start + (i + 1) * section_length, 
                  colors=colors[game], 
                  linestyles='-', 
                  label=f"{game} Section" if i < len(game_order) else "",
                  linewidth=2)
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Z-Score')
    ax.set_title(f'{game_name}')
    ax.legend()
    ax.xaxis.set_major_formatter(FuncFormatter(kilo_formatter))



for ax, (game, game_info) in zip(axs, games.items()):
    df = pd.read_csv(game_info['path'])
    timestep_column = 'Evaluation/Timesteps-' + game + 'NoFrameskip-v4'
    score_columns = [
        f'seed: 49 - Evaluation/Score-{game}NoFrameskip-v4',
        f'seed: 30 - Evaluation/Score-{game}NoFrameskip-v4',
        f'seed: 1 - Evaluation/Score-{game}NoFrameskip-v4'
    ]
    plot_for_game(ax, df, timestep_column, score_columns, game, game_info['first_line'], game_info['game_order'], window=window)

plt.tight_layout()
plt.legend(loc='lower right')
plt.show()
