from typing import List, Optional, Union
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class NumericalEDA:
    def __init__(self, dataframe, cat_cols: Optional[List[str]] = None, target_col: str = None):
        """
        dataframe: pd.DataFrame.
        cat_cols (Optional[List[str]]): Columns to treat as categorical, exclude from corr.
        target_col (Optional[Union[str, List[str]]]): Column(s) to always include (target(s)).
        """
        self.dataframe = dataframe
        self.cat_cols = cat_cols
        self.target_col = target_col

    ## Box plot and hist plot
    def plot_box_and_hist_per_feature(self,
                                      compare_cols=None,
                                      hue_col=None, 
                                      cols_per_row=2, 
                                      figsize_per_plot=(6,5), 
                                      bins=30,
                                      rotate_xticks=False,
                                      n_bins_for_continuous_target=4
                                      )-> None:
        """
        For each numerical feature, plots a boxplot (by hue) on top and histogram below.
        
        Parameters:
        - self.dataframe: pandas self.dataframe
        - hue_col: Target/grouping column 
        - cols_per_row: Number of features per row (each has 2 subplots: box + hist)
        - figsize_per_plot: Size per feature block (width, height)
        - bins: Bins for histogram
        - rotate_xticks: Whether to rotate x-tick labels
        """
        if compare_cols:
            columns = compare_cols
        else:
            if self.cat_cols is not None:
                num_cols = list(set(self.dataframe.columns) - set(self.cat_cols) - set([self.target_col]))
                num_cols += [self.target_col]
                columns = num_cols
            else:
                columns = self.dataframe.select_dtypes(include='number').columns.tolist()

            if hue_col and hue_col in columns:
                columns.remove(hue_col)
        
        n_features = len(columns)
        n_cols = cols_per_row
        n_rows = math.ceil(n_features / n_cols)

        fig, axes = plt.subplots(
            nrows=n_rows * 2, ncols=n_cols,
            figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows * 2)
        )
        axes = axes.reshape((n_rows * 2, n_cols))

        # decide what to do with hue_col
        hue_data = None
        if hue_col:
            if pd.api.types.is_numeric_dtype(self.dataframe[hue_col]):
                # If continuous, bin it
                hue_data = pd.qcut(self.dataframe[hue_col], q=n_bins_for_continuous_target, duplicates='drop').astype(str)
                legend_title = f'{hue_col} (binned)'
            else:
                hue_data = self.dataframe[hue_col]
                legend_title = hue_col
        else:
            legend_title = None

        for idx, col in enumerate(columns):
            row_idx = (idx // n_cols) * 2
            col_idx = idx % n_cols

            # Boxplot (top)
            if hue_col:
                sns.boxplot(
                    x=hue_data if hue_data is not None else self.dataframe[hue_col],
                    y=self.dataframe[col],
                    ax=axes[row_idx][col_idx],
                    palette='tab10',
                    showfliers=False
                )
                axes[row_idx][col_idx].set_title(f'{col} - Boxplot by {legend_title}')
                axes[row_idx][col_idx].set_xlabel('')
            else:
                sns.boxplot(
                    y=self.dataframe[col],
                    ax=axes[row_idx][col_idx],
                    color='#1f77b4',
                    showfliers=False
                )
                axes[row_idx][col_idx].set_title(f'{col} - Boxplot')

            if rotate_xticks:
                axes[row_idx][col_idx].tick_params(axis='x', rotation=45)

            # Histogram (bottom)
            if hue_col:
                sns.histplot(
                    x=self.dataframe[col],
                    hue=hue_data if hue_data is not None else self.dataframe[hue_col],
                    kde=True,
                    bins=bins,
                    ax=axes[row_idx + 1][col_idx],
                    stat="density",
                    common_norm=False,
                    palette='tab10'
                )
                axes[row_idx + 1][col_idx].set_title(f'{col} - Histogram by {legend_title}')
            else:
                sns.histplot(
                    x=self.dataframe[col],
                    kde=True,
                    bins=bins,
                    ax=axes[row_idx + 1][col_idx],
                    stat="density",
                    color='#1f77b4'
                )
                axes[row_idx + 1][col_idx].set_title(f'{col} - Histogram')

            axes[row_idx + 1][col_idx].set_ylabel('Density')
            if rotate_xticks:
                axes[row_idx + 1][col_idx].tick_params(axis='x', rotation=45)

        # Hide any unused subplots
        total_axes = n_rows * 2 * n_cols
        for i in range(idx + 1, total_axes // 2):
            axes[(i // n_cols) * 2][i % n_cols].set_visible(False)
            axes[(i // n_cols) * 2 + 1][i % n_cols].set_visible(False)

        ## adding boarder for each feature
        for group in range(1, n_rows):
                y_pos = 1 - (group * 2 / (n_rows * 2))  # Figure coords
                rect = patches.Rectangle(
                    (0, y_pos), 1, 0.001,  # x, y, width, height
                    transform=fig.transFigure,
                    color="grey",
                    alpha=0.1,
                    zorder=0
                )
                fig.patches.append(rect)
        
        # Optionally: Add vertical separator lines between column groups
        for col in range(1, n_cols):
            x_pos = col / n_cols
            rect = patches.Rectangle(
                (x_pos, 0), 0.001, 1,  # x, y, width, height
                transform=fig.transFigure,
                color="grey",
                alpha=0.1,
                zorder=0
            )
            fig.patches.append(rect)
        plt.tight_layout()
        plt.show()
    
    def plot_corr_heat_map(
        self, 
        figsize: tuple = (12, 10),
        annot: bool = True, 
        rotation: int = 45
    ) -> None:
        """
        Plots a correlation heatmap for numerical features.

        Args:
            figsize (tuple): Figure size.
            annot (bool): Annotate with correlation values.
            rotation (int): Rotation for x tick labels.

        Returns:
            None.
        """
        if self.cat_cols is not None:
            num_cols = list(set(self.dataframe.columns) - set(self.cat_cols) - set([self.target_col]))
            num_cols += [self.target_col]
            data_numerical = self.dataframe[num_cols]
        else:
            data_numerical = self.dataframe.select_dtypes(include='number')

        corr_matrix = data_numerical.corr()

        plt.figure(figsize=figsize)
        ax = sns.heatmap(
            corr_matrix, 
            annot=annot, 
            fmt=".2f", 
            cmap='coolwarm', 
            square=True, 
            linewidths=0.5, 
            cbar=True
        )
        plt.title("Correlation Matrix of Numerical Features")
        plt.xticks(rotation=rotation, ha='right')
        plt.yticks(rotation=0, ha='right')
        
        print("\n📊 Plotting Correlation Matrix of Numerical Features...")

        plt.tight_layout()
        plt.show()

class CategoricalEDA:
    def __init__(self, dataframe, cat_cols, target_col: Optional[Union[str, List[str]]] = None):
        self.dataframe = dataframe
        self.cat_cols = cat_cols
        self.target_col = target_col

    def plot_bar_and_normalized_bars(
        self,
        cols_per_row=2,
        figsize_per_plot=(6, 5),
        rotate_xticks=True,
        n_bins_for_continuous_target=4,
        top_n=5  # Number of categories to keep, rest are "Other"
    ):
        """
        Plots count and normalized bar charts for each categorical feature.
        Groups rare categories into 'Other' if too many categories.
        """
        n_features = len(self.cat_cols)
        n_cols = cols_per_row
        n_rows = math.ceil(n_features / n_cols)

        fig, axes = plt.subplots(
            nrows=n_rows * 2,
            ncols=n_cols,
            figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows * 2)
        )
        axes = axes.reshape((n_rows * 2, n_cols))

        # Prepare hue/target info
        hue_data = None
        legend_title = self.target_col
        show_hue = False
        if self.target_col:
            if pd.api.types.is_numeric_dtype(self.dataframe[self.target_col]) and self.dataframe[self.target_col].nunique() > n_bins_for_continuous_target:
                hue_data = pd.qcut(self.dataframe[self.target_col], q=n_bins_for_continuous_target, duplicates="drop").astype(str)
                legend_title = f"{self.target_col} (binned)"
                show_hue = True
            elif self.dataframe[self.target_col].nunique() <= 20:
                hue_data = self.dataframe[self.target_col]
                show_hue = True
            else:
                hue_data = None
                show_hue = False

        for idx, col in enumerate(self.cat_cols):
            row_idx = (idx // n_cols) * 2
            col_idx = idx % n_cols

            # ---- Lumping logic ----
            top_categories = self.dataframe[col].value_counts().nlargest(top_n).index
            if len(self.dataframe[col].unique()) <= top_n:
                col_grouped = self.dataframe[col]
                cats = list(top_categories)
                plt_header = ""
            else:
                col_grouped = self.dataframe[col].where(self.dataframe[col].isin(top_categories), "Other")
                cats = list(top_categories) + ["Other"]
                plt_header = f"(top {top_n} + Other)"

            # Top: Countplot
            if show_hue:
                sns.countplot(
                    x=col_grouped,
                    hue=hue_data,
                    data=self.dataframe,
                    order=cats,
                    palette="tab10",
                    ax=axes[row_idx][col_idx]
                )
                axes[row_idx][col_idx].set_title(f"{col} {plt_header} - Count by {legend_title}")
                axes[row_idx][col_idx].legend(title=legend_title, fontsize=8, title_fontsize=9)
            else:
                sns.countplot(
                    x=col_grouped,
                    data=self.dataframe,
                    order=cats,
                    color="#1f77b4",
                    ax=axes[row_idx][col_idx]
                )
                axes[row_idx][col_idx].set_title(f"{col} {plt_header} - Count")

            axes[row_idx][col_idx].set_xlabel("")
            axes[row_idx][col_idx].set_ylabel("Count")
            if rotate_xticks:
                axes[row_idx][col_idx].tick_params(axis='x', rotation=45)

            # Add count labels
            for container in axes[row_idx][col_idx].containers:
                for bar in container:
                    height = bar.get_height()
                    if height > 0:
                        axes[row_idx][col_idx].text(
                            bar.get_x() + bar.get_width() / 2,
                            height + 0.01 * height,
                            f"{int(height)}",
                            ha="center", va="bottom", fontsize=8, color="black"
                        )

            # Bottom: Normalized bar
            if show_hue:
                norm_df = pd.crosstab(col_grouped, hue_data, normalize="index")
                norm_df = norm_df.loc[[cat for cat in cats if cat in norm_df.index]]
                norm_df.plot(
                    kind="bar",
                    stacked=True,
                    colormap="tab10",
                    ax=axes[row_idx + 1][col_idx],
                    legend=False
                )
                axes[row_idx + 1][col_idx].set_title(f"{col} {plt_header} - Normalized % by {legend_title}")
                axes[row_idx + 1][col_idx].set_ylabel("Proportion")
                axes[row_idx + 1][col_idx].set_xlabel(col)
                for idx2, category in enumerate(norm_df.index):
                    bottom = 0
                    for cls in norm_df.columns:
                        height = norm_df.loc[category, cls]
                        if height > 0.01:
                            axes[row_idx + 1][col_idx].text(
                                idx2, bottom + height / 2, f"{height*100:.1f}%",
                                ha="center", va="center", fontsize=8, color="white"
                            )
                        bottom += height
                axes[row_idx + 1][col_idx].legend(title=legend_title, fontsize=8, title_fontsize=9)
            else:
                norm_df = col_grouped.value_counts(normalize=True).reindex(cats, fill_value=0)
                norm_df.plot(
                    kind="bar",
                    color="#1f77b4",
                    ax=axes[row_idx + 1][col_idx]
                )
                axes[row_idx + 1][col_idx].set_title(f"{col} {plt_header} - Normalized %")
                axes[row_idx + 1][col_idx].set_ylabel("Proportion")
                axes[row_idx + 1][col_idx].set_xlabel(col)
                for idx2, val in enumerate(norm_df.values):
                    axes[row_idx + 1][col_idx].text(
                        idx2, val + 0.01, f"{val*100:.1f}%",
                        ha="center", va="bottom", fontsize=8, color="black"
                    )

            if rotate_xticks:
                axes[row_idx + 1][col_idx].tick_params(axis='x', rotation=45)

        # Hide unused plots
        total_axes = n_rows * 2 * n_cols
        for i in range(idx + 1, total_axes // 2):
            axes[(i // n_cols) * 2][i % n_cols].set_visible(False)
            axes[(i // n_cols) * 2 + 1][i % n_cols].set_visible(False)

        ## adding boarder for each feature
        for group in range(1, n_rows):
                y_pos = 1 - (group * 2 / (n_rows * 2)) - 0.005 # Figure coords
                rect = patches.Rectangle(
                    (0, y_pos), 1, 0.001,  # x, y, width, height
                    transform=fig.transFigure,
                    color="grey",
                    alpha=0.1,
                    zorder=0
                )
                fig.patches.append(rect)
        
        # Optionally: Add vertical separator lines between column groups
        for col in range(1, n_cols):
            x_pos = col / n_cols
            rect = patches.Rectangle(
                (x_pos, 0), 0.001, 1,  # x, y, width, height
                transform=fig.transFigure,
                color="grey",
                alpha=0.1,
                zorder=0
            )
            fig.patches.append(rect)
            
        plt.tight_layout()
        plt.show()

