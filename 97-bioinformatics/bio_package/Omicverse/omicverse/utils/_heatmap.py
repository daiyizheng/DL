import numpy as np
import pandas as pd
from scipy.sparse import issparse
from matplotlib.colors import cnames, is_color_like, ListedColormap, to_rgb
from matplotlib import patheffects, rcParams
import scanpy as sc
import anndata
from typing import Union, Sequence

#from scvelo import logging as logg



def plot_heatmap(
    adata:anndata.AnnData,
    var_names: Union[str, Sequence[str]],
    sortby:str="latent_time",
    layer:str="Ms",
    color_map:str="RdBu_r",
    col_color=None,
    palette:str="viridis",
    n_convolve:int=30,
    standard_scale:int=0,
    sort:bool=True,
    colorbar=None,
    col_cluster:bool=False,
    row_cluster:bool=False,
    context=None,
    font_scale=None,
    figsize:tuple=(8, 4),
    show=None,
    save=None,
    **kwargs,
):
    """Plot time series for genes as heatmap.

    Arguments:
        adata: Annotated data matrix.
        var_names: Names of variables to use for the plot.
        sortby: Observation key to extract time data from.
        layer: Layer key to extract count data from.
        color_map: String denoting matplotlib color map.
        col_color: String denoting matplotlib color map to use along the columns.
        palette: Colors to use for plotting groups (categorical annotation).
        n_convolve: If int is given, data is smoothed by convolution
            along the x-axis with kernel size n_convolve.
        standard_scale: Either 0 (rows) or 1 (columns). Whether or not to standardize that dimension
            (each row or column), subtract minimum and divide each by its maximum.
        sort: Wether to sort the expression values given by xkey.
        colorbar: Whether to show colorbar.
        {row,col}_cluster: If True, cluster the {rows, columns}.
        context: A dictionary of parameters or the name of a preconfigured set.
        font_scale: Scaling factor to scale the size of the font elements.
        figsize: Figure size.
        show: Show the plot, do not return axis.
        save: If True or a str, save the figure. A string is appended to the default
            filename. Infer the filetype if ending on {'.pdf', '.png', '.svg'}.
        kwargs: Arguments passed to seaborns clustermap,
            e.g., set `yticklabels=True` to display all gene names in all rows.

    Returns:
        matplotlib.Axis`

    """
    import seaborn as sns

    var_names = [name for name in var_names if name in adata.var_names]

    tkey, xkey = kwargs.pop("tkey", sortby), kwargs.pop("xkey", layer)
    time = adata.obs[tkey].values
    time = time[np.isfinite(time)]

    X = (
        adata[:, var_names].layers[xkey]
        if xkey in adata.layers.keys()
        else adata[:, var_names].X
    )
    if issparse(X):
        X = X.A
    df = pd.DataFrame(X[np.argsort(time)], columns=var_names)

    if n_convolve is not None:
        weights = np.ones(n_convolve) / n_convolve
        for gene in var_names:
            # TODO: Handle exception properly
            try:
                df[gene] = np.convolve(df[gene].values, weights, mode="same")
            except ValueError as e:
                print(f"Skipping variable {gene}: {e}")
                pass  # e.g. all-zero counts or nans cannot be convolved

    if sort:
        max_sort = np.argsort(np.argmax(df.values, axis=0))
        df = pd.DataFrame(df.values[:, max_sort], columns=df.columns[max_sort])
    strings_to_categoricals(adata)

    if col_color is not None:
        col_colors = to_list(col_color)
        col_color = []
        for _, col in enumerate(col_colors):
            if not is_categorical(adata, col):
                obs_col = adata.obs[col]
                cat_col = np.round(obs_col / np.max(obs_col), 2) * np.max(obs_col)
                adata.obs[f"{col}_categorical"] = pd.Categorical(cat_col)
                col += "_categorical"
                set_colors_for_categorical_obs(adata, col, palette)
            col_color.append(interpret_colorkey(adata, col)[np.argsort(time)])

    if "dendrogram_ratio" not in kwargs:
        kwargs["dendrogram_ratio"] = (
            0.1 if row_cluster else 0,
            0.2 if col_cluster else 0,
        )
    if "cbar_pos" not in kwargs or not colorbar:
        kwargs["cbar_pos"] = None

    kwargs.update(
        {
            "col_colors": col_color,
            "col_cluster": col_cluster,
            "row_cluster": row_cluster,
            "cmap": color_map,
            "xticklabels": False,
            "standard_scale": standard_scale,
            "figsize": figsize,
        }
    )

    args = {}
    if font_scale is not None:
        args = {"font_scale": font_scale}
        context = context or "notebook"

    with sns.plotting_context(context=context, **args):
        # TODO: Remove exception by requiring appropriate seaborn version
        try:
            cm = sns.clustermap(df.T, **kwargs)
        except ImportWarning:
            print("Please upgrade seaborn with `pip install -U seaborn`.")
            kwargs.pop("dendrogram_ratio")
            kwargs.pop("cbar_pos")
            cm = sns.clustermap(df.T, **kwargs)

    #savefig_or_show("heatmap", save=save, show=show)
    if show is False:
        return cm




# TODO: Add docstrings
def default_color(adata, add_outline=None):
    """TODO."""
    if (
        isinstance(add_outline, str)
        and add_outline in adata.var.keys()
        and "recover_dynamics" in adata.uns.keys()
        and add_outline in adata.uns["recover_dynamics"]
    ):
        return adata.uns["recover_dynamics"][add_outline]
    return (
        "clusters"
        if "clusters" in adata.obs.keys()
        else "louvain"
        if "louvain" in adata.obs.keys()
        else "grey"
    )

# TODO: Add docstrings
def make_dense(X):
    """TODO."""
    if issparse(X):
        XA = X.A if X.ndim == 2 else X.A1
    else:
        XA = X.A1 if isinstance(X, np.matrix) else X
    return np.array(XA)

# TODO: Add docstrings
def default_palette(palette=None):
    """TODO."""
    from cycler import Cycler, cycler
    if palette is None:
        return rcParams["axes.prop_cycle"]
    elif not isinstance(palette, Cycler):
        return cycler(color=palette)
    else:
        return palette
    
# TODO: Add docstrings
def adjust_palette(palette, length):
    """TODO."""
    from cycler import Cycler, cycler
    islist = False
    if isinstance(palette, list):
        islist = True
    if (islist and len(palette) < length) or (
        not isinstance(palette, list) and len(palette.by_key()["color"]) < length
    ):
        if length <= 28:
            palette = sc.pl.palettes.zeileis_28
        elif length <= len(sc.pl.palettes.zeileis_102):  # 103 colors
            palette = sc.pl.palettes.zeileis_102
        else:
            palette = ["grey" for _ in range(length)]
            print("more than 103 colors would be required, initializing as 'grey'")
        return palette if islist else cycler(color=palette)
    elif islist:
        return palette
    elif not isinstance(palette, Cycler):
        return cycler(color=palette)
    else:
        return palette

# TODO: Add docstrings
def get_colors(adata, c):
    """TODO."""
    if is_color_like(c):
        return c
    else:
        if f"{c}_colors" not in adata.uns.keys():
            palette = default_palette(None)
            palette = adjust_palette(palette, length=len(adata.obs[c].cat.categories))
            n_cats = len(adata.obs[c].cat.categories)
            adata.uns[f"{c}_colors"] = palette[:n_cats].by_key()["color"]
        if isinstance(adata.uns[f"{c}_colors"], dict):
            cluster_ix = adata.obs[c].values
        else:
            cluster_ix = adata.obs[c].cat.codes.values
        return np.array(
            [
                adata.uns[f"{c}_colors"][cluster_ix[i]]
                if cluster_ix[i] != -1
                else "lightgrey"
                for i in range(adata.n_obs)
            ]
        )

# TODO: Add docstrings
def interpret_colorkey(adata, c=None, layer=None, perc=None, use_raw=None):
    """TODO."""
    if c is None:
        c = default_color(adata)
    if issparse(c):
        c = make_dense(c).flatten()
    if is_categorical(adata, c):
        c = get_colors(adata, c)
    elif isinstance(c, str):
        if is_color_like(c) and c not in adata.var_names:
            pass
        elif c in adata.obs.keys():  # color by observation key
            c = adata.obs[c]
        elif c in adata.var_names or (
            use_raw and adata.raw is not None and c in adata.raw.var_names
        ):  # by gene
            if layer in adata.layers.keys():
                if perc is None and any(
                    layer_name in layer
                    for layer_name in ["spliced", "unspliced", "Ms", "Mu", "velocity"]
                ):
                    perc = [1, 99]  # to ignore outliers in non-logarithmized layers
                c = adata.obs_vector(c, layer=layer)
            elif layer is not None and np.any(
                [
                    layer_name in layer or "X" in layer
                    for layer_name in adata.layers.keys()
                ]
            ):
                l_array = np.hstack(
                    [
                        adata.obs_vector(c, layer=layer)[:, None]
                        for layer in adata.layers.keys()
                    ]
                )
                l_array = pd.DataFrame(l_array, columns=adata.layers.keys())
                l_array.insert(0, "X", adata.obs_vector(c))
                c = np.array(l_array.astype(np.float32).eval(layer))
            else:
                if layer is not None and layer != "X":
                    print(layer, "not found. Using .X instead.")
                if adata.raw is None and use_raw:
                    raise ValueError("AnnData object does not have `raw` counts.")
                c = adata.raw.obs_vector(c) if use_raw else adata.obs_vector(c)
            c = c.A.flatten() if issparse(c) else c
        elif c in adata.var.keys():  # color by observation key
            c = adata.var[c]
        elif np.any([var_key in c for var_key in adata.var.keys()]):
            var_keys = [
                k for k in adata.var.keys() if not isinstance(adata.var[k][0], str)
            ]
            var = adata.var[list(var_keys)]
            c = var.astype(np.float32).eval(c)
        elif np.any([obs_key in c for obs_key in adata.obs.keys()]):
            obs_keys = [
                k for k in adata.obs.keys() if not isinstance(adata.obs[k][0], str)
            ]
            obs = adata.obs[list(obs_keys)]
            c = obs.astype(np.float32).eval(c)
        elif not is_color_like(c):
            raise ValueError(
                "color key is invalid! pass valid observation annotation or a gene name"
            )
        if not isinstance(c, str) and perc is not None:
            c = clip(c, perc=perc)
    else:
        c = np.array(c).flatten()
        if perc is not None:
            c = clip(c, perc=perc)
    return c

# TODO: Add docstrings
def clip(c, perc):
    """TODO."""
    if np.size(perc) < 2:
        perc = [perc, 100] if perc < 50 else [0, perc]
    lb, ub = np.percentile(c, perc)
    return np.clip(c, lb, ub)

# TODO: Finish docstrings
def strings_to_categoricals(adata):
    """Transform string annotations to categoricals."""
    from pandas import Categorical
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype

    def is_valid_dtype(values):
        return (
            is_string_dtype(values) or is_integer_dtype(values) or is_bool_dtype(values)
        )

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

# TODO: Add docstrings
def is_list(key):
    """TODO."""
    return isinstance(key, (list, tuple, np.record))

# TODO: Add docstrings
def is_list_of_str(key, max_len=None):
    """TODO."""
    if max_len is not None:
        return (
            is_list_or_array(key)
            and len(key) < max_len
            and all(isinstance(item, str) for item in key)
        )
    else:
        return is_list(key) and all(isinstance(item, str) for item in key)
    
# TODO: Add docstrings
def is_list_or_array(key):
    """TODO."""
    return isinstance(key, (list, tuple, np.record, np.ndarray))


# TODO: Add docstrings
def to_list(key, max_len=20):
    """TODO."""
    from pandas import Index
    if isinstance(key, Index) or is_list_of_str(key, max_len):
        key = list(key)
    return key if is_list(key) and (max_len is None or len(key) < max_len) else [key]

# TODO: Add docstrings
def is_view(adata):
    """TODO."""
    return (
        adata.is_view
        if hasattr(adata, "is_view")
        else adata.isview
        if hasattr(adata, "isview")
        else adata._isview
        if hasattr(adata, "_isview")
        else True
    )

# TODO: Add docstrings
def is_categorical(data, c=None):
    """TODO."""
    from pandas.api.types import is_categorical_dtype as cat

    if c is None:
        return cat(data)  # if data is categorical/array
    if not is_view(data):  # if data is anndata view
        strings_to_categoricals(data)
    return isinstance(c, str) and c in data.obs.keys() and cat(data.obs[c])


# colors in addition to matplotlib's colors
additional_colors = {
    'gold2': '#eec900', 'firebrick3': '#cd2626', 'khaki2': '#eee685',
    'slategray3': '#9fb6cd', 'palegreen3': '#7ccd7c', 'tomato2': '#ee5c42',
    'grey80': '#cccccc', 'grey90': '#e5e5e5', 'wheat4': '#8b7e66', 'grey65': '#a6a6a6',
    'grey10': '#1a1a1a', 'grey20': '#333333', 'grey50': '#7f7f7f', 'grey30': '#4d4d4d',
    'grey40': '#666666', 'antiquewhite2': '#eedfcc', 'grey77': '#c4c4c4',
    'snow4': '#8b8989', 'chartreuse3': '#66cd00', 'yellow4': '#8b8b00',
    'darkolivegreen2': '#bcee68', 'olivedrab3': '#9acd32', 'azure3': '#c1cdcd',
    'violetred': '#d02090', 'mediumpurple3': '#8968cd', 'purple4': '#551a8b',
    'seagreen4': '#2e8b57', 'lightblue3': '#9ac0cd', 'orchid3': '#b452cd',
    'indianred 3': '#cd5555', 'grey60': '#999999', 'mediumorchid1': '#e066ff',
    'plum3': '#cd96cd', 'palevioletred3': '#cd6889'
}

# adapted from scanpy
def set_colors_for_categorical_obs(adata, value_to_plot, palette=None):
    """Sets adata.uns[f'{value_to_plot}_colors'] to given palette or default colors.

    Parameters
    ----------
    adata
        annData object
    value_to_plot
        name of a valid categorical observation
    palette
        Palette should be either a valid :func:`~matplotlib.pyplot.colormaps` string,
        a sequence of colors (in a format that can be understood by matplotlib,
        eg. RGB, RGBS, hex, or a cycler object with key='color'
    """
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as pl
    from cycler import Cycler, cycler


    color_key = f"{value_to_plot}_colors"
    valid = True
    categories = adata.obs[value_to_plot].cat.categories
    length = len(categories)

    if isinstance(palette, str) and "default" in palette:
        palette = sc.pl.palettes.zeileis_28 if length <= 28 else sc.pl.palettes.zeileis_102
    if isinstance(palette, str) and palette in adata.uns:
        palette = (
            [adata.uns[palette][c] for c in categories]
            if isinstance(adata.uns[palette], dict)
            else adata.uns[palette]
        )
    if palette is None and color_key in adata.uns:
        color_keys = adata.uns[color_key]
        # Check if colors already exist in adata.uns and if they are a valid palette
        if isinstance(color_keys, np.ndarray) and isinstance(color_keys[0], dict):
            adata.uns[color_key] = adata.uns[color_key][0]
        # Flatten the dict to a list (mainly for anndata compatibilities)
        if isinstance(adata.uns[color_key], dict):
            adata.uns[color_key] = [adata.uns[color_key][c] for c in categories]
        color_keys = adata.uns[color_key]
        for color in color_keys:
            if not is_color_like(color):
                # check if valid color translate to a hex color value
                if color in additional_colors:
                    color = additional_colors[color]
                else:
                    print(
                        f"The following color value found in "
                        f"adata.uns['{value_to_plot}_colors'] is not valid: '{color}'. "
                        f"Default colors will be used instead."
                    )
                    valid = False
                    break
        if len(adata.uns[color_key]) < len(adata.obs[value_to_plot].cat.categories):
            valid = False
    elif palette is not None:
        # check is palette given is a valid matplotlib colormap
        if isinstance(palette, str) and palette in pl.colormaps():
            # this creates a palette from a colormap. E.g. 'Accent, Dark2, tab20'
            cmap = pl.get_cmap(palette)
            colors_list = [to_hex(x) for x in cmap(np.linspace(0, 1, length))]

        else:
            # check if palette is an array of length n_obs
            if isinstance(palette, (list, np.ndarray)) or is_categorical(palette):
                if len(adata.obs[value_to_plot]) == len(palette):
                    cats = pd.Categorical(adata.obs[value_to_plot])
                    colors = pd.Categorical(palette)
                    if len(cats) == len(colors):
                        palette = dict(zip(cats, colors))
            # check if palette is as dict and convert it to an ordered list
            if isinstance(palette, dict):
                palette = [palette[c] for c in categories]
            # check if palette is a list and convert it to a cycler
            if isinstance(palette, abc.Sequence):
                if len(palette) < length:
                    print(
                        "Length of palette colors is smaller than the number of "
                        f"categories (palette length: {len(palette)}, "
                        f"categories length: {length}. "
                        "Some categories will have the same color."
                    )
                # check that colors are valid
                _color_list = []
                for color in palette:
                    if not is_color_like(color):
                        # check if valid color and translate to a hex color value
                        if color in additional_colors:
                            color = additional_colors[color]
                        else:
                            print(
                                f"The following color value is not valid: '{color}'. "
                                f"Default colors will be used instead."
                            )
                            valid = False
                            break
                    _color_list.append(color)
                palette = cycler(color=_color_list)

            if not isinstance(palette, Cycler) or "color" not in palette.keys:
                print(
                    "Please check that the value of 'palette' is a valid "
                    "matplotlib colormap string (eg. Set2), a list of color names or "
                    "a cycler with a 'color' key. Default colors will be used instead."
                )
                valid = False

            if valid:
                cc = palette()
                colors_list = [to_hex(next(cc)["color"]) for x in range(length)]
        if valid:
            adata.uns[f"{value_to_plot}_colors"] = colors_list
    else:
        # No valid palette exists or was given
        valid = False

    # Set to defaults:
    if not valid:
        # check if default matplotlib palette has enough colors
        if len(rcParams["axes.prop_cycle"].by_key()["color"]) >= length:
            cc = rcParams["axes.prop_cycle"]()
            palette = [next(cc)["color"] for _ in range(length)]
        # Else fall back to default palettes
        else:
            if length <= 28:
                palette = sc.pl.palettes.zeileis_28
            elif length <= len(sc.pl.palettes.zeileis_102):  # 103 colors
                palette = sc.pl.palettes.zeileis_102
            else:
                palette = ["grey" for _ in range(length)]
                print(
                    f"the obs value {value_to_plot!r} has more than 103 categories. "
                    f"Uniform 'grey' color will be used for all categories."
                )

        adata.uns[f"{value_to_plot}_colors"] = palette[:length]

