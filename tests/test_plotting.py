from audition.plotting import generate_plot_lines, category_colordict, plot_cats
import matplotlib.lines as mlines
import pandas
from unittest.mock import patch


def test_generate_plot_lines():
    colordict = {
        'cat1': '#001122',
        'cat2': '#112233',
        'cat3': '#223344',
    }
    label_fcn = lambda x: 'Cat {}'.format(x)

    plot_lines = generate_plot_lines(colordict, label_fcn)
    assert len(plot_lines) == 3
    for line in plot_lines:
        assert type(line) == mlines.Line2D
        assert 'Cat ' in line._label


def test_category_colordict():
    cmap_name = 'Vega10'
    categories = ['Cat1', 'Cat2', 'Cat3', 'Cat4']
    colordict = category_colordict(cmap_name, categories)
    assert len(colordict.keys()) == 4


def test_plot_cats():
    test_df = pandas.DataFrame.from_dict({
        'cats': ['tuxedo', 'maine coon', 'lion!'],
        'groups': ['i', 'dont', 'know'],
        'col1': [1, 2, 3],
        'col2': [4, 5, 6],
        'col3': [7, 8, 9],
    })
    # hard to make many assertions, but we can make sure it gets to the end
    # and shows the contents
    with patch('audition.plotting.plt.show') as show_patch:
        plot_cats(test_df, 'col1', 'col2', cat_col='cats', grp_col='groups')
        assert show_patch.called
