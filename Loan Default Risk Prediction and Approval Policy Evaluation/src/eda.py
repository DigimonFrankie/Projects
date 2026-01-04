import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import List, Optional

class NumericalEDA:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cat_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        exclude_col: Optional[List[str]] = None,
        categorical_target: bool = True
    ):
        self.df = dataframe
        self.cat_cols = cat_cols or []
        self.target_col = target_col
        self.exclude_col = exclude_col or []
        self.categorical_target = categorical_target

    def _get_num_columns(self):
        cols = (
            set(self.df.columns)
            - set(self.cat_cols)
            - set(self.exclude_col)
            - ({self.target_col} if self.target_col else set())
        )
        return sorted(cols)

    def _binary_palette(self):
        # Always use strings!
        return {"0": "#1f77b4", "1": "#ff7f0e"}

    def _auto_palette(self):
        # Use tab10 for non-binary/multiclass
        labels = self.df[self.target_col].dropna().astype(str).unique()
        return dict(zip(labels, sns.color_palette("tab10", len(labels))))

    def _hide_unused(self, axes, used, n_rows, n_cols):
        total = n_rows * n_cols
        for i in range(used, total):
            r = (i // n_cols) * 2
            c = i % n_cols
            axes[r][c].set_visible(False)
            axes[r + 1][c].set_visible(False)

    def _add_borders(self, fig, n_rows, n_cols):
        for g in range(1, n_rows):
            y = 1 - (g * 2 / (n_rows * 2))
            fig.patches.append(
                patches.Rectangle((0, y), 1, 0.001,
                                    transform=fig.transFigure,
                                    color="grey", alpha=0.1)
            )
        for c in range(1, n_cols):
            x = c / n_cols
            fig.patches.append(
                patches.Rectangle((x, 0), 0.001, 1,
                                    transform=fig.transFigure,
                                    color="grey", alpha=0.1)
            )

    def plot_box_and_hist_per_feature(
        self,
        cols_per_row: int = 2,
        figsize_per_plot: tuple = (6, 5),
        bins: int = 30,
        rotate_xticks: bool = True
    ):
        num_cols = self._get_num_columns()
        n_features = len(num_cols)
        n_cols = cols_per_row
        n_rows = math.ceil(n_features / n_cols)

        fig, axes = plt.subplots(
            nrows=n_rows * 2,
            ncols=n_cols,
            figsize=(figsize_per_plot[0] * n_cols,
                     figsize_per_plot[1] * n_rows * 2),
            squeeze=False
        )

        # Choose palette for categorical target
        """
        Set target value to string to avoid type inconsistency.
        """
        palette = None
        if self.categorical_target and self.target_col:
            unique_vals = sorted(self.df[self.target_col].dropna().astype(str).unique())
            if set(unique_vals) == {"0", "1"}:
                palette = self._binary_palette()
            else:
                palette = self._auto_palette()

        showed_legend = False

        for idx, col in enumerate(num_cols):
            r = (idx // n_cols) * 2
            c = idx % n_cols

            # ---------- BOXPLOT ----------
            if self.categorical_target and self.target_col:
                plot_df = self.df.assign(**{self.target_col: self.df[self.target_col].astype(str)})
                sns.boxplot(
                    data=plot_df,
                    x=self.target_col,
                    y=col,
                    hue=self.target_col,
                    palette=palette,
                    showfliers=False,
                    ax=axes[r][c],
                    dodge=False  # Avoids duplicate legends for binary
                )
                axes[r][c].set_title(f"{col} vs {self.target_col}")
                # Only need legend for one plot in each column
                leg = axes[r][c].get_legend()
                if leg and not showed_legend:
                    leg.set_title(self.target_col)
                    showed_legend = True
                elif leg:
                    leg.remove()
            else:
                sns.boxplot(
                    y=self.df[col],
                    showfliers=False,
                    color="#1f77b4",
                    ax=axes[r][c]
                )
                axes[r][c].set_title(f"{col} Distribution")

            axes[r][c].set_xlabel("")

            # ---------- HISTOGRAM ----------
            ax_hist = axes[r + 1][c]
            if self.categorical_target and self.target_col:
                plot_df = self.df.assign(**{self.target_col: self.df[self.target_col].astype(str)})
                sns.histplot(
                    data=plot_df,
                    x=col,
                    hue=self.target_col,
                    kde=True,
                    bins=bins,
                    stat="density",
                    common_norm=False,
                    palette=palette,
                    alpha=0.6,
                    ax=ax_hist
                )
                ax_hist.set_title(f"{col} by {self.target_col}")
                # Only legend in the first plot of the grid
                leg = ax_hist.get_legend()
                if leg:
                    leg.remove()
            else:
                sns.histplot(
                    self.df[col],
                    kde=True,
                    bins=bins,
                    stat="density",
                    color="#1f77b4",
                    ax=ax_hist
                )
                ax_hist.set_title(f"{col} Histogram")

            ax_hist.set_ylabel("Density")

            if rotate_xticks:
                for label in axes[r][c].get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")
                for label in ax_hist.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha("right")

        self._hide_unused(axes, n_features, n_rows, n_cols)
        self._add_borders(fig, n_rows, n_cols)
        plt.tight_layout()
        plt.show()

    def plot_corr_heat_map(
        self,
        figsize: tuple = (12, 10),
        annot: bool = True,
        rotation: int = 45
    ):
        columns = self._get_num_columns()
        data_numerical = self.df[columns]
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
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cat_cols: List[str],
        target_col: Optional[str] = None,
        exclude_col: Optional[List[str]] = None
    ):
        self.dataframe = dataframe
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.exclude_col = exclude_col or []

    def _get_plot_cols(self):
        return [c for c in self.cat_cols if c not in self.exclude_col]

    def _prep_hue(self, n_bins_for_continuous_target=4):
        hue_data, legend_title, show_hue = None, None, False
        if self.target_col:
            target = self.dataframe[self.target_col]
            if pd.api.types.is_numeric_dtype(target) and target.nunique() > n_bins_for_continuous_target:
                hue_data = pd.qcut(target, q=n_bins_for_continuous_target, duplicates="drop").astype(str)
                legend_title = f"{self.target_col} (binned)"
                show_hue = True
            elif target.nunique() <= 20:
                hue_data = target.astype(str)
                legend_title = self.target_col
                show_hue = True
        return hue_data, legend_title, show_hue

    def _lump_categories(self, ser, top_n):
        vc = ser.value_counts(dropna=False)
        top_categories = vc.nlargest(top_n).index
        if len(vc) <= top_n:
            grouped_col = ser
            categories = list(top_categories)
            title_suffix = ""
        else:
            grouped_col = ser.where(ser.isin(top_categories), "Other")
            categories = list(top_categories) + ["Other"]
            title_suffix = f"(top {top_n} + Other)"
        return grouped_col, categories, title_suffix

    def _hide_unused_axes(self, axes, n_features, n_rows, n_cols):
        total_slots = n_rows * n_cols
        for i in range(n_features, total_slots):
            r = (i // n_cols) * 2
            c = i % n_cols
            axes[r][c].set_visible(False)
            axes[r + 1][c].set_visible(False)

    def plot_bar_and_normalized_bars(
        self,
        cols_per_row: int = 2,
        figsize_per_plot: tuple = (6, 5),
        rotate_xticks: bool = True,
        n_bins_for_continuous_target: int = 4,
        top_n: int = 5
    ):
        df = self.dataframe
        plot_cols = self._get_plot_cols()
        n_features = len(plot_cols)
        if n_features == 0:
            print("⚠️ No categorical columns to plot.")
            return
        n_cols = cols_per_row
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(
            nrows=n_rows * 2,
            ncols=n_cols,
            figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows * 2),
            squeeze=False
        )
        hue_data, legend_title, show_hue = self._prep_hue(n_bins_for_continuous_target)

        # Set binary palette for 0/1 targets; else fallback to tab10
        palette = None
        if show_hue:
            unique_hue = sorted(pd.Series(hue_data).dropna().unique())
            if set(unique_hue) == {"0", "1"}:
                palette = {"0": "#1f77b4", "1": "#ff7f0e"}
            else:
                palette = dict(zip([str(u) for u in unique_hue], sns.color_palette("tab10", len(unique_hue))))

        for idx, col in enumerate(plot_cols):
            row_idx = (idx // n_cols) * 2
            col_idx = idx % n_cols
            grouped_col, categories, title_suffix = self._lump_categories(df[col], top_n)

            # --- Countplot ---
            ax_top = axes[row_idx][col_idx]
            if show_hue:
                sns.countplot(
                    x=grouped_col.astype(str),
                    hue=hue_data,
                    order=[str(cat) for cat in categories],
                    palette=palette,
                    ax=ax_top
                )
                ax_top.legend(title=legend_title, fontsize=8, title_fontsize=9)
            else:
                sns.countplot(
                    x=grouped_col,
                    order=categories,
                    color="#1f77b4",
                    ax=ax_top
                )
            ax_top.set_title(f"{col} {title_suffix} - Count")
            ax_top.set_xlabel("")
            ax_top.set_ylabel("Count")
            if rotate_xticks:
                ax_top.tick_params(axis="x", rotation=45)
            # --- Bar labels ---
            for container in ax_top.containers:
                for bar in container:
                    if bar.get_height() > 0:
                        ax_top.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height(),
                            f"{int(bar.get_height())}",
                            ha="center",
                            va="bottom",
                            fontsize=8
                        )

            # --- Normalized barplot ---
            ax_bottom = axes[row_idx + 1][col_idx]
            if show_hue:
                norm_df = pd.crosstab(grouped_col.astype(str), hue_data, normalize="index")
                all_hue_levels = [str(u) for u in pd.Series(hue_data).unique()]
                norm_df = norm_df.reindex(index=[str(cat) for cat in categories], columns=all_hue_levels, fill_value=0)

                norm_df.plot(
                    kind="bar",
                    stacked=True,
                    color=[palette[str(u)] for u in all_hue_levels],
                    ax=ax_bottom,
                    legend=False
                )
                ax_bottom.legend(title=legend_title, fontsize=8, title_fontsize=9)

                # Centered % labels for stacked bars
                for x_idx, category in enumerate(norm_df.index):
                    bottom = 0
                    for cls in norm_df.columns:
                        height = norm_df.loc[category, cls]
                        if height >= 0.05:
                            ax_bottom.text(
                                x_idx,
                                bottom + height / 2,
                                f"{height * 100:.1f}%",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="white"
                            )
                        bottom += height
            else:
                norm = grouped_col.value_counts(normalize=True).reindex(categories).fillna(0)
                norm.plot(kind="bar", color="#1f77b4", ax=ax_bottom)

                for i, val in enumerate(ax_bottom.patches):
                    height = val.get_height()
                    if height > 0.02:
                        ax_bottom.text(
                            val.get_x() + val.get_width() / 2,
                            height / 2,
                            f"{height * 100:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white"
                        )
        self._hide_unused_axes(axes, n_features, n_rows, n_cols)

        # XTICK ROTATION FOR ALL SUBPLOTS
        for axrow in axes:
            for ax in axrow:
                for label in ax.get_xticklabels():
                    label.set_rotation(45 if rotate_xticks else 0)
                    label.set_horizontalalignment('right' if rotate_xticks else 'center')

        plt.tight_layout()
        plt.show()