from .ipyquiz import Quiz, Function


def quiz_count_tokens():
    # lesson 1, quiz 1
    def solution(s):
        from spacy.lang.en import English
        nlp = English()
        return len(nlp(s))

    return Function(
        "Compute the number of tokens in an input string using spaCy.",
        etalon_solution=solution,
        input_list=[
            ("This is a text.",),
            ("Don't reinvent the wheel, use spaCy.",),
            ("Easy-peasy lemon squeezy",),
            ("Easy-peasy lemon squeezy.",),
            ("Two sentences. With exclamation mark!",),
        ],
        input_output_list=[],
        show_n_answers=2
    )


def quiz_count_lemmas():
    # lesson 1, quiz 2
    def etalon(s):
        import spacy
        nlp = spacy.load("en")
        return len({t.lemma_ for t in nlp(s)})

    return Function(
        "Compute the number of unique lemmas in the input string",
        etalon_solution=etalon,
        input_list=[
            ("Let it be, let it be",),
            ("When I find myself in times of trouble, Mother Mary comes to me",),
            ("Speaking words of wisdom, \"let it be\"",),
            ("And in my hour of darkness, she is standing right in front of me",),
            ("Speaking words of wisdom, \"let it be\"",),
            ("Let it be, let it be",),
            ("Let it be, let it be",),
            ("Let it be, let it be",),
            ("Whisper words of wisdom",),
            ("Let it be",),
        ],
        input_output_list=[],
        show_n_answers=2
    )


def quiz_ner():
    # lesson 1, quiz 3
    def etalon(s):
        import spacy
        nlp = spacy.load("en")
        return set(ent.label_ for ent in nlp(s).ents)

    return Function(
        "Return the set of all named entity labels found in the text",
        etalon_solution=etalon,
        input_list=[
            ("No named entities",),
            ("London is the capital of Great Britain.",),
            ("Donald Trump is the President of the U.S.",),
            ("As funding slows in Boston, its early-stage market could shine",),
            ("California turns to vote-by-mail to keep residents safe come November",),
        ],
        input_output_list=[],
        show_n_answers=2
    )


def quiz_bumps():
    # lesson 6, quiz 1
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

    return Quiz(description, [1, 2, 3], 2)


def quiz_derivative():
    # lesson 6, quiz 2
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        w = widgets.HTMLMath(
            value=r"Consider the function $$\tilde h(x)=\sum\limits_{1 \leq k \leq K} w_k g(a_k x + b_k) + s_k$$")
        display(w)

    with description:
        display(widgets.HTMLMath(r"What is the derivative $\dfrac{\partial \ell(y, \tilde h(x))}{\partial s_k}$?"))
        display(widgets.HTMLMath(r"""
            <ol>
                <li>$\dfrac{\partial \ell(y, \tilde h(x))}{\partial s_k} = (y - \tilde h(x)) \cdot g(a_k x + b_k)$</li>
                <li>$\dfrac{\partial \ell(y, \tilde h(x))}{\partial s_k} = w_k \cdot (y - \tilde h(x))$</li>
                <li>$\dfrac{\partial \ell(y, \tilde h(x))}{\partial s_k} = y - \tilde h(x)$</li>
            </ol>
        """))
        display(widgets.HTML("Choose the correct answer below"))

    return Quiz(description, [1, 2, 3], 3)


def quiz_derivative_pytorch():
    from IPython.display import display

    def solution(x):
        import torch
        x = torch.tensor(x, requires_grad=True)
        z = torch.pow(x, x)
        z.backward()
        return x.grad.item()

    import ipywidgets as widgets
    description = widgets.Output()

    with description:
        w = widgets.HTMLMath(r"Compute the derivative of $f(x) = x^x$ with PyTorch. Hint: use torch.pow function.<hr>")
        display(w)

    return Function(
        description,
        etalon_solution=solution,
        input_list=[[float(x)] for x in range(1, 20, 2)],
        input_output_list=[],
        show_n_answers=2
    )
