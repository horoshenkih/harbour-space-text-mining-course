from .ipyquiz import Quiz


def quiz_bumps():
    from IPython.display import display
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    description = widgets.Output()
    with description:
        w = widgets.HTMLMath(value=r"Consider the following function $f(x)$:")
        display(w)

    with description:
        _x = [-1, 0, 0, 1, 1, 2]
        _y = [2, 2, 3, 3, 4, 4]
        plt.figure(figsize=(8, 4))
        plt.plot(_x, _y)
        plt.show()

    with description:
        display(widgets.HTMLMath(r"Which formula for $f(x)$ is correct?"))
        display(widgets.HTMLMath(r"""
            <ol>
                <li>$f(x) = \Pi_{-1, 0, 2}(x) + \Pi_{0, 1, 1}(x) + \Pi_{1, 2, 1}(x)$</li>
                <li>$f(x) = \Pi_{-1, 0, 2}(x) + \Pi_{0, 1, 3}(x) + \Pi_{1, 2, 4}(x)$</li>
                <li>$f(x) = \Pi_{-1, 0, 0}(x) + \Pi_{0, 1, 1}(x) + \Pi_{1, 2, 2}(x)$</li>
            </ol>
        """))
        display(widgets.HTML("Choose the correct answer below"))

    q = Quiz(description, [1, 2, 3], 2)
    q()
