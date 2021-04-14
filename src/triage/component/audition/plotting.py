import matplotlib
import numpy as np
import matplotlib.lines as mlines

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_bounds(observed_min, observed_max):
    """Compute the plot bounds for observed data

    Args:
        observed_min (number) -- Lowest number found in data
        observed_max (number) -- Highest number found in data

    Returns: (number, number) A minimum and maximum x value for the plot
    """
    if observed_min >= 0.0 and observed_max <= 1.0:
        plot_min, plot_max = (0.0, 1.0)
    else:
        # 10% padding on the high end
        padding = 0.1 * (observed_max - float(observed_min))
        plot_min = observed_min
        plot_max = observed_max + padding

    return plot_min, plot_max


def category_colordict(cmap_name, categories, highlight_grp=None):
    # want to step through the discrete color map rather than sampling
    # across the entire range, so create an even spacing from 0 to 1
    # with as many steps as in the color map (cmap.N), then repeat it
    # enough times to ensure we cover all our categories
    cmap = plt.get_cmap(cmap_name)
    categories_with_colors = [cat for cat in categories if cat != highlight_grp]
    ncyc = int(np.ceil(1.0 * len(categories_with_colors) / cmap.N))
    colors = (cmap.colors * ncyc)[: len(categories_with_colors)]
    base_colors = dict(zip(categories_with_colors, colors))
    if highlight_grp:
        base_colors[highlight_grp] = "#000000"
    return base_colors


def category_styledict(colordict, highlight_grp):
    """Generate a dictionary mapping categories to styles.

    The only styling implemented at present is converting the highlighted group to a dashed line.

    Args:
        colordict (dict) A mapping of categories to colors
        highlight_grp (string) The name of a group/category that should be highlighted

    Returns: (dict) A mapping of categories to matplotlib styles
    """
    return dict(
        (key, "--" if key == highlight_grp else "-") for key in colordict.keys()
    )


def _plot_lines(frame, x_col, y_col, ax, grp_col, colordict, cat_col, styledict, alpha):
    # plot the lines, one for each model group,
    # looking up the color by model type from above
    for grp_val in np.unique(frame[grp_col]):
        df = frame.loc[frame[grp_col] == grp_val]
        cat = df.iloc[0][cat_col]
        df.plot(
            x_col, y_col, ax=ax, c=colordict[cat], style=styledict[cat], legend=False, alpha=alpha
        )


def generate_plot_lines(colordict, label_fcn, styledict):
    plot_lines = []
    # plot_labs = []
    for cat_val in sorted(colordict.keys()):
        # http://matplotlib.org/users/legend_guide.html
        lin = mlines.Line2D(
            xdata=[],
            ydata=[],
            linestyle=styledict[cat_val],
            color=colordict[cat_val],
            label=label_fcn(cat_val),
        )
        plot_lines.append(lin)
        # plot_labs.append(mt)
    return plot_lines


def _config_axes(
    ax,
    x_ticks,
    y_ticks,
    x_lim,
    y_lim,
    title,
    title_fontsize,
    x_label,
    y_label,
    label_fontsize,
):
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    elif x_lim is not None:
        ax.set_xlim(x_lim)
    else:
        ax.set_xlim([0, 1.1])

    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    elif y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim([0, 1.1])
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=label_fontsize)
    ax.set_xlabel(x_label, fontsize=label_fontsize)


def _get_leaf(path):
    return path.rsplit(".", 1)[-1]


def _no_op(arg):
    return arg


def plot_cats(
    frame,
    x_col,
    y_col,
    cat_col="model_type",
    grp_col="model_group_id",
    highlight_grp=None,
    title="",
    x_label="",
    y_label="",
    cmap_name="tab10",
    figsize=[12, 6],
    x_ticks=None,
    y_ticks=None,
    x_lim=None,
    y_lim=None,
    legend_loc=None,
    legend_fontsize=12,
    label_fontsize=12,
    title_fontsize=16,
    label_fcn=None,
    path_to_save=None,
    alpha=0.4,
    colordict=None,
    styledict=None,
):
    """Plot a line plot with each line colored by a category variable.

    Arguments:
        frame (DataFrame) -- a dataframe containing the data to be plotted
        x_col (string) -- name of the x-axis column
        y_col (string) -- name of the y-axis column
        cat_col (string) -- name of the catagory column to color lines
        grp_col (string) -- column that identifies each group of
                            (x_col, y_col) points for each line
        highlight_grp (string) -- name of group that should be highlighted in some way
            (like a baseline)
        title (string) -- allows specifying a custom title for the graph
        x_label (string) -- allows specifying a custom label for the x-axis
        y_label (string) -- allows specifying a custom label for the y-axis
        cmap_name (string) -- matplotlib color map name to use for plot
        figsize (tuple) -- figure size to pass to matplotlib
        x_ticks (sequence) -- optional ticks to use for x-axis
        y_ticks (sequence) -- optional ticks to use for y-axis
        x_lim (tuple) -- optional min-max to use for x-axis
        y_lim (tuple) -- optional min-max to use for y-axis
        legend_loc (string) -- allows specifying location of plot legend
        legend_fontsize (int) -- allows specifying font size for legend
        label_fontsize (int) -- allows specifying font size for axis labels
        title_fontsize (int) -- allows specifying font size for plot title
        label_fcn (method) -- function to map category names to more readable
                                names, accepting values of cat_col
        path_to_save (str) -- optional file path to save plot to disk
        alpha (float) -- value of alpha for plotting lines
        colordict (dict) -- optional dict mapping categories to colors
        styledict (dict) -- optional dict mapping categories to styles

    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # function for parsing cat_col values into more readable legend lables
    if label_fcn is None:
        label_fcn = _get_leaf if cat_col == "model_type" else _no_op

    categories = np.unique(frame[cat_col])

    colordict = colordict or category_colordict(cmap_name, categories, highlight_grp)
    styledict = styledict or category_styledict(colordict, highlight_grp)

    # plot the lines, one for each model group,
    # looking up the color by model type from above
    _plot_lines(frame, x_col, y_col, ax, grp_col, colordict, cat_col, styledict, alpha)

    # have to set the legend manually since we don't want one legend
    # entry per line on the plot, just one per model type.

    # I had to upgrade matplotlib to get handles working, otherwise
    # had to call like this with plot_labs as a separate list
    # plt.legend(plot_patches, plot_labs, loc=4, fontsize=10)
    plot_lines = generate_plot_lines(colordict, label_fcn, styledict)

    plt.legend(handles=plot_lines, loc=legend_loc, fontsize=legend_fontsize)

    _config_axes(
        ax=ax,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        x_lim=x_lim,
        y_lim=y_lim,
        title=title,
        title_fontsize=title_fontsize,
        x_label=x_label,
        y_label=y_label,
        label_fontsize=label_fontsize,
    )

    plt.show()

    if path_to_save:
        plt.savefig(path_to_save)
