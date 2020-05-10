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


def quiz_tfidf():
    # lesson 1, quiz 4
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTMLMath(value=r"Consider the collection $D$ containing 3 documents"))
        display(widgets.HTMLMath(value=r'''1. $d_1$ = "If you tell the truth you don't have to remember anything."'''))
        display(widgets.HTMLMath(
            value=r'''2. $d_2$ = "If you don't read the newspaper, you're uninformed. If you read the newspaper, you're misinformed."'''))
        display(widgets.HTMLMath(
            value=r'''3. $d_3$ = "A lie can travel half way around the world while the truth is putting on its shoes."'''))
        display(widgets.HTMLMath(value=r'''Compute $$TFIDF(\mathrm{''If''}, d_1, D)$$'''))
    return Quiz(description, ["2.2", "0.81", "0.405", "1.5"], "0.405")


def quiz_vector_distance():
    # lesson 1, quiz 5
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTMLMath(value=r"Consider the following set of vectors"))
        display(widgets.HTMLMath(value=r"$$\mathbb{a} = (0.5, 0.5)$$"))
        display(widgets.HTMLMath(value=r"$$\mathbb{b} = (0.4, 0.8)$$"))
        display(widgets.HTMLMath(value=r"$$\mathbb{c} = (0, 0.5)$$"))

        display(widgets.HTMLMath(value=r"Which one is the closest to $\mathbb{w} = (1, 0)$?"))

    return Quiz(
        description,
        ["a", "b", "c"],
        "a"
    )


def quiz_vectorizer_shape():
    # lesson 1, quiz 6
    import ipywidgets as widgets
    from IPython.display import display, HTML

    description = widgets.Output()

    with description:
        display(HTML("""
            With the following code
            <p style="font-family:'Lucida Console', monospace">data = ["one, two", "three, four"]</p>
            <p style="font-family:'Lucida Console', monospace">vectorizer = TfidfVectorizer().fit(data)</p>
            <p style="font-family:'Lucida Console', monospace">X = vectorizer.transform(data)</p>
            What is the shape of X?
        """))

    return Quiz(
        description,
        [
            "1 row, 4 columns",
            "2 rows, 2 columns",
            "2 rows, 4 columns",
            "4 rows, 2 columns",
            "4 rows, 1 column",
        ]
        ,
        "2 rows, 4 columns"
    )


def quiz_conditional_probability():
    # lesson 2 quiz 1
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTML(value="Given Pr(AB) = 0.2, Pr(A) = 0.4, Pr(B) = 0.8, find Pr(A|B) and Pr(B|A)."))

    return Quiz(
        description,
        [
            "Pr(A|B) = 0.2, Pr(B|A) = 0.2",
            "Pr(A|B) = 0.5, Pr(B|A) = 0.25",
            "Pr(A|B) = 0.25, Pr(B|A) = 0.5",
        ],
        "Pr(A|B) = 0.25, Pr(B|A) = 0.5"
    )


def quiz_chain_rule():
    # lesson 2 quiz 2
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTML(
            value="Given Pr(C) = 0.5, Pr(B|C) = 0.3, Pr(A|C) = 0.8, Pr(B|AC) = 0.25, Pr(C|AB) = 0.5, find P(ABC)."))

    return Quiz(
        description,
        [
            "0.1",
            "0.2",
            "0.3",
            "0.5",
            "Not enough data"
        ],
        "0.1"
    )


def quiz_bigram_lm():
    # lesson 2 quiz 3
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTMLMath(value=r"""
            Consider a bigram language model with the following probabilities:
            <ul>
                <li>$\Pr(\textrm{"am so"})$ = 0.0007</li>
                <li>$\Pr(\textrm{"am very"})$ = 0.0009</li>
                <li>$\Pr(\textrm{"am the"})$ = 0.0009</li>
                <li>$\Pr(\textrm{"I am"})$ = 0.019</li>
                <li>$\Pr(\textrm{"I"})$ = 0.16</li>
                <li>$\Pr(\textrm{"am"})$ = 0.02</li>
                <li>$\Pr(\textrm{"so"})$ = 0.04</li>
            </ul>
            Find $\Pr(\textrm{"I am so"})$
        """))

    return Quiz(
        description,
        [
            "0.00076",
            "0.000665",
            "0.0003325",
        ],
        "0.000665"
    )


def quiz_count_ngrams():
    # lesson 2 quiz 4
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTMLMath(value=r"""
            Given the vocabulary $V$, what is the number of all possible $n$-grams constructed from $V$?
            Assume there is no padding.
        """))

    return Quiz(
        description,
        [
            r"$|V|$",
            r"$n + |V|$",
            r"$n\cdot|V|$",
            r"$|V|^n$",
        ],
        r"$|V|^n$"
    )


def quiz_perplexity():
    # lesson 2 quiz 5
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTMLMath(value=r"""
            Consider a unigram language model with the following probabilities:
            <ul>
                <li>$\Pr(\textrm{"A"})$ = 0.5</li>
                <li>$\Pr(\textrm{"B"})$ = 0.1</li>
                <li>$\Pr(\textrm{"C"})$ = 0.25</li>
            </ul>
            Compute $\textrm{Perplexity}(\textrm{"AABCC"})$
        """))

    return Quiz(
        description,
        [
            "3.64",
            "0.27",
            "640",
            "0.0016",
        ],
        "3.64"
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
