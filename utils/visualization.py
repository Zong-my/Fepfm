"""
@File    : visualization.py
@Time    : 2025/3/5 1:52
@Author  : mingyang.zong
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_distribution(df, y1_col, y2_col, save_path='svr.png'):
    """
    Plot the distribution of two variables (y1 and y2) in a given DataFrame.

    Parameters:
    df (DataFrame): Input data containing columns for y1 and y2.
    y1_col (str): Column name for variable y1 in the DataFrame.
    y2_col (str): Column name for variable y2 in the DataFrame.
    save_path (str, optional): Path to save the generated figure. Defaults to 'svr.png'.
    """
    plt.figure(figsize=(14, 6))

    # Set font sizes
    title_fontsize = 18
    label_fontsize = 14

    # Plot distribution curve for y1
    plt.subplot(1, 2, 1)
    sns.histplot(df[y1_col], kde=True, color='blue')
    plt.title(f'Distribution of {y1_col}', fontsize=title_fontsize)
    plt.xlabel(y1_col, fontsize=label_fontsize)
    plt.ylabel('Frequency', fontsize=label_fontsize)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines

    # Plot distribution curve for y2
    plt.subplot(1, 2, 2)
    sns.histplot(df[y2_col], kde=True, color='green')
    plt.title(f'Distribution of {y2_col}', fontsize=title_fontsize)
    plt.xlabel(y2_col, fontsize=label_fontsize)
    plt.ylabel('Frequency', fontsize=label_fontsize)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines

    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=600)


def plot_prediction(y1_test, y1_pred, y2_test, y2_pred,
                   f1_title_name='f_deltamax True vs f_deltamax Predicted',
                   f1_y1_txt='True',
                   f1_y2_txt='Predicted',
                   f1_x_label='Number',
                   f1_y_label='Value',
                   f2_title_name='t_delta True vs t_delta Predicted',
                   f2_y1_txt='True',
                   f2_y2_txt='Predicted',
                   f2_x_label='Number',
                   f2_y_label='Value',
                   save_path=None,
                   dpi=600):
    """
    Visualize predicted values versus true values for y1 and y2.

    Parameters:
    y1_test (array-like): True values of f_deltamax.
    y1_pred (array-like): Predicted values of f_deltamax.
    y2_test (array-like): True values of t_delta.
    y2_pred (array-like): Predicted values of t_delta.
    f1_title_name (str): Title for the first subplot (f_deltamax).
    f1_y1_txt (str): Legend label for true values in the first subplot.
    f1_y2_txt (str): Legend label for predicted values in the first subplot.
    f1_x_label (str): X-axis label for the first subplot.
    f1_y_label (str): Y-axis label for the first subplot.
    f2_title_name (str): Title for the second subplot (t_delta).
    f2_y1_txt (str): Legend label for true values in the second subplot.
    f2_y2_txt (str): Legend label for predicted values in the second subplot.
    f2_x_label (str): X-axis label for the second subplot.
    f2_y_label (str): Y-axis label for the second subplot.
    save_path (str, optional): Path to save the generated image; if None, image is not saved.
    dpi (int, optional): Image resolution in dots per inch (DPI), defaults to 600.
    """
    plt.figure(figsize=(14, 8))
    title_fontsize = 20
    label_fontsize = 16

    # Set seaborn style
    sns.set(style="whitegrid")

    # Plot comparison of true and predicted values for y1
    plt.subplot(2, 1, 1)
    sns.lineplot(x=range(len(y1_test)), y=y1_test, label=f1_y1_txt, color='blue')
    sns.lineplot(x=range(len(y1_pred)), y=y1_pred, label=f1_y2_txt, color='red', linestyle='--')
    plt.title(f1_title_name, fontsize=title_fontsize)
    plt.xlabel(f1_x_label, fontsize=label_fontsize)
    plt.ylabel(f1_y_label, fontsize=label_fontsize)
    plt.legend()
    plt.grid(True)

    # Plot comparison of true and predicted values for y2
    plt.subplot(2, 1, 2)
    sns.lineplot(x=range(len(y2_test)), y=y2_test, label=f2_y1_txt, color='green')
    sns.lineplot(x=range(len(y2_pred)), y=y2_pred, label=f2_y2_txt, color='orange', linestyle='--')
    plt.title(f2_title_name, fontsize=title_fontsize)
    plt.xlabel(f2_x_label, fontsize=label_fontsize)
    plt.ylabel(f2_y_label, fontsize=label_fontsize)
    plt.legend()
    plt.grid(True)

    # Adjust subplot spacing
    plt.tight_layout()

    # Save the figure
    if save_path:
        if save_path.lower().endswith('.pdf'):
            plt.savefig(save_path, format='pdf')
        elif save_path.lower().endswith('.svg'):
            plt.savefig(save_path, format='svg')
        else:
            plt.savefig(save_path, dpi=dpi)


def plot_predictions(data_save, save_path=None, dpi=600, title_fontsize=20, label_fontsize=16, legend_fontsize=14):
    """
    Visualize predicted values versus true values for f_deltamax and t_delta.

    Parameters:
    data_save (DataFrame): Input data containing predictions and ground truth values.
    save_path (str, optional): Path to save the generated figure; if None, image is not saved.
    dpi (int, optional): Image resolution in dots per inch (DPI), defaults to 600.
    title_fontsize (int, optional): Font size for titles, defaults to 20.
    label_fontsize (int, optional): Font size for axis labels, defaults to 16.
    legend_fontsize (int, optional): Font size for legends, defaults to 14.
    """
    plt.figure(figsize=(14, 8))

    # Set seaborn style
    sns.set(style="whitegrid")

    # Extract columns related to f_deltamax and t_delta
    f_deltamax_columns = [col for col in data_save.columns if 'f_deltamax' in col]
    t_delta_columns = [col for col in data_save.columns if 't_delta' in col]

    # Plot comparison of true and predicted values for f_deltamax
    plt.subplot(2, 1, 1)
    for col in f_deltamax_columns:
        sns.lineplot(x=data_save.index, y=data_save[col], label=col, marker='o', markersize=3)
    plt.title('f_deltamax True vs Predicted', fontsize=title_fontsize)
    plt.xlabel('Number', fontsize=label_fontsize)
    plt.ylabel('Value', fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)

    # Plot comparison of true and predicted values for t_delta
    plt.subplot(2, 1, 2)
    for col in t_delta_columns:
        sns.lineplot(x=data_save.index, y=data_save[col], label=col, marker='o', markersize=3)
    plt.title('t_delta True vs Predicted', fontsize=title_fontsize)
    plt.xlabel('Number', fontsize=label_fontsize)
    plt.ylabel('Value', fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.grid(True)

    # Adjust subplot spacing
    plt.tight_layout()

    # Save the figure
    if save_path:
        if save_path.lower().endswith('.pdf'):
            plt.savefig(save_path, format='pdf', dpi=dpi)
        elif save_path.lower().endswith('.svg'):
            plt.savefig(save_path, format='svg', dpi=dpi)
        else:
            plt.savefig(save_path, dpi=dpi)