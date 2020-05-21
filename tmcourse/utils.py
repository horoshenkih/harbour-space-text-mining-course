from collections import defaultdict
from datetime import timedelta, date, datetime
import typing as tp


def calendar_table(
        times: tp.List[tp.Union[date, datetime]],
        values: tp.List[tp.Any],
        weights: tp.List[float],
        from_datetime: tp.Optional[tp.Union[date, datetime]] = None,
        to_datetime: tp.Optional[tp.Union[date, datetime]] = None,
        font_size=20
) -> str:
    table_data = defaultdict(dict)
    sorted_weights = list(sorted(weights))
    if len(weights) > 20:
        margin = int(0.05 * len(weights))
    else:
        margin = 0
    min_weight = sorted_weights[margin]
    max_weight = sorted_weights[-margin]
    for tvw in zip(times, values, weights):
        dt, value, weight = tvw
        if from_datetime is not None and dt < from_datetime:
            continue
        if to_datetime is not None and dt > to_datetime:
            continue
        year, week, weekday = dt.isocalendar()
        if max_weight > min_weight:
            r = (weight - min_weight) / (max_weight - min_weight)
        else:
            r = 1.
        color = "rgba(0, 0, 255, {})".format(0.8 * r + 0.2)
        value = '<div style="color:{};font-weight:bold">{}</div>'.format(color, value)
        week_start = dt - timedelta(days=dt.weekday())
        week_end = week_start + timedelta(days=6)
        table_data[week_start.strftime('%Y-%m-%d') + " - " + week_end.strftime('%Y-%m-%d')][weekday] = value
    header = ["week", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    table = [[k] + [table_data[k].get(i, "") for i in range(1, 8)] for k in sorted(table_data)]
    style = """<style>
    th, td {{
      padding: 8px;
      text-align: center;
      border-bottom: 1px solid #ddd;
      font-size: {font_size};
    }}
    tr {{
      font-size: {font_size};
    }}
    </style>""".format(font_size=font_size)
    html = style + '<table><tr>{}</tr><tr>{}</tr></table>'.format(
        ''.join(['<th>{}</th>'.format(h) for h in header]),
        '</tr><tr>'.join('<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in table)
    )
    return html


def visualize_term_counter(term_counter, start, title, shift=10, font_size=22, figsize=(12, 8), use_ggplot=True):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': font_size})

    if use_ggplot:
        plt.style.use('ggplot')

    plt.figure(figsize=figsize)

    ordered_term_count = term_counter.most_common()[start:start + shift]
    x_pos = range(len(ordered_term_count))
    words = [x[0] for x in reversed(ordered_term_count)]
    counts = [x[1] for x in reversed(ordered_term_count)]
    plt.barh(x_pos, counts, color="green")
    plt.yticks(x_pos, words)
    plt.ylabel("Lemma")
    plt.xlabel("Frequency")
    plt.title(title)
    plt.show()


def enable_mathjax_in_cell():
    from IPython.display import HTML, display
    # https://colab.research.google.com/gist/blois/cea02123a82a4dd40500b31c39dfcb23/mathjaxoutputs.ipynb#scrollTo=QTO5EXGMxaIj
    display(HTML('''
        <script src="https://www.gstatic.com/external_hosted/mathjax/latest/MathJax.js?config=TeX-AMS_HTML-full,Safe&delayStartupUntil=configured"></script>
        <script>
            (() => {
            const mathjax = window.MathJax;
            mathjax.Hub.Config({
            'tex2jax': {
                'inlineMath': [['$', '$'], ['\\(', '\\)']],
                'displayMath': [['$$', '$$'], ['\\[', '\\]']],
                'processEscapes': true,
                'processEnvironments': true,
                'skipTags': ['script', 'noscript', 'style', 'textarea', 'code'],
                'displayAlign': 'center',
            },
            'HTML-CSS': {
                'styles': {'.MathJax_Display': {'margin': 0}},
                'linebreaks': {'automatic': true},
                // Disable to prevent OTF font loading, which aren't part of our
                // distribution.
                'imageFont': null,
            },
            'messageStyle': 'none'
            });
            mathjax.Hub.Configured();
        })();
        </script>
        '''))


def display_cv_results(clf):
    from IPython.display import display
    import pandas as pd
    params = [k for k in clf.cv_results_.keys() if k.startswith("param_")]
    df_cv_results = pd.DataFrame(clf.cv_results_).sort_values(by=['rank_test_score'])[["mean_test_score"] + params]
    df_cv_results.reset_index(drop=True, inplace=True)
    display(df_cv_results)


def display_token_importance(token_importances):
    from IPython.display import display, HTML
    min_token_importance = min([ti[1] for ti in token_importances])
    max_token_importance = max([ti[1] for ti in token_importances])
    html_tokens = []
    for token, importance in token_importances:
        if max_token_importance == min_token_importance:
            r = 0.5
        else:
            r = (importance - min_token_importance) / (max_token_importance - min_token_importance)
        color = "rgba(0, 0, 255, {})".format(0.8 * r + 0.2)
        html_tokens.append("<span style='color:{}'>{}</span>".format(color, token))
    display(HTML(" ".join(html_tokens)))


def plot_confusion_matrix(target, prediction, normalize=None):
    import seaborn as sns; sns.set()  # for plot styling
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    mat = confusion_matrix(target, prediction, normalize=normalize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(mat, square=True, annot=True, fmt=fmt, cbar=False, cmap="coolwarm")
    plt.ylabel('true label')
    plt.xlabel('predicted label')


def _sample_words(model, words=None, sample=0):
    import numpy as np
    if words is None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]
    return [w for w in words if w in model.vocab]


def display_pca_scatterplot(model, words=None, sample=0):
    # https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/materials/Gensim%20word%20vector%20visualization.html

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    words = _sample_words(model, words=words, sample=sample)
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:, :2]

    plt.figure(figsize=(6, 6))
    plt.scatter(twodim[:, 0], twodim[:, 1], edgecolors='k', c='r')
    for word, (x, y) in zip(words, twodim):
        plt.text(x + 0.05, y + 0.05, word)


def display_pca_scatterplot_interactive(
    model, words=None, sample=0,
    radius=10, alpha=0.25, color='blue', width=600, height=400, show=True):
    # https://github.com/yandexdataschool/nlp_course/blob/2019/week01_embeddings/seminar.ipynb

    """ draws an interactive plot for data points with auxilirary info on hover """
    from sklearn.decomposition import PCA
    import numpy as np

    import bokeh.models as bm, bokeh.plotting as pl
    from bokeh.io import output_notebook
    output_notebook()

    words = _sample_words(model, words=words, sample=sample)

    kwargs = {
        'token': words
    }
    word_vectors = np.array([model[w] for w in words])

    twodim = PCA().fit_transform(word_vectors)[:,:2]
    x, y = twodim[:,0], twodim[:,1]

    if isinstance(color, str): color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show: pl.show(fig)
    return fig


def show_vectorizer_largest_components(vectorizer, vectors, n_components=20):
    from pprint import pprint
    vocab = {i: v for v, i in vectorizer.vocabulary_.items()}
    for center in vectors:
        pprint([vocab[i] for i in center.argsort()[-n_components:]], compact=True)
