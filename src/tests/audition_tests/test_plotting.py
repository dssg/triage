from unittest.mock import patch

import pandas as pd
from matplotlib import lines as mlines

from triage.component.audition.plotting import (
    generate_plot_lines,
    category_colordict,
    category_styledict,
    plot_cats,
)


def test_generate_plot_lines():
    colordict = {"cat1": "#001122", "cat2": "#112233", "cat3": "#223344"}
    styledict = {"cat1": "-", "cat2": "--", "cat3": "-"}
    plot_lines = generate_plot_lines(colordict, lambda x: "Cat {}".format(x), styledict)
    assert len(plot_lines) == 3
    for line in plot_lines:
        assert type(line) == mlines.Line2D
        assert "Cat " in line._label
        assert "-" in line._linestyle
        if line._label == "Cat 2":
            assert line._linestyle == "--"


def test_category_colordict():
    cmap_name = "tab10"
    categories = ["Cat1", "Cat2", "Cat3", "Cat4"]
    colordict = category_colordict(cmap_name, categories)
    assert len(colordict.keys()) == 4


def test_category_colordict_with_highlight():
    cmap_name = "tab10"
    colordict_with_highlight = category_colordict(
        cmap_name, ["Cat1", "Cat2", "Cat3", "Cat4"], "Cat2"
    )
    colordict_without_highlight = category_colordict(
        cmap_name, ["Cat1", "Cat3", "Cat4"]
    )
    for cat in ["Cat1", "Cat3", "Cat4"]:
        assert colordict_with_highlight[cat] == colordict_without_highlight[cat]
    assert colordict_with_highlight["Cat2"] == "#000000"


def test_category_styledict():
    colordict = {"cat1": "#001122", "cat2": "#112233", "cat3": "#223344"}
    assert category_styledict(colordict, "cat3") == {
        "cat1": "-",
        "cat2": "-",
        "cat3": "--",
    }


def test_plot_cats():
    test_df = pd.DataFrame.from_dict(
        {
            "cats": ["tuxedo", "maine coon", "lion!"],
            "groups": ["i", "dont", "know"],
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
            "col3": [7, 8, 9],
        }
    )
    # hard to make many assertions, but we can make sure it gets to the end
    # and shows the contents
    with patch("triage.component.audition.plotting.plt.show") as show_patch:
        plot_cats(test_df, "col1", "col2", cat_col="cats", grp_col="groups")
        assert show_patch.called
