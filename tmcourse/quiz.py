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
        nlp = spacy.load("en_core_web_sm")
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
        nlp = spacy.load("en_core_web_sm")
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


def quiz_random_benchmark():
    # lesson 2 quiz 6
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()

    with description:
        display(widgets.HTMLMath(value=r"""
            Suppose that we compute accuracy on the dataset with 4 classes, and there is equal number of samples of each class.
            (In other words, all classes are balanced.)
            What is accuracy of "random" classifier, that predicts random category with probability 0.25?
        """))

    return Quiz(
        description,
        [
            "0",
            "0.0625",
            "0.25",
            "0.5",
        ],
        "0.25"
    )


def quiz_pipeline_parameter():
    # lesson 2, quiz 7
    import ipywidgets as widgets
    from IPython.display import display, HTML

    description = widgets.Output()

    with description:
        display(HTML("""
            Consider the following pipeline of <span style="font-family:'Lucida Console', monospace">TfidfVectorizer</span> and <span style="font-family:'Lucida Console', monospace">SGDClassifier</span>:
            <p style="font-family:'Lucida Console', monospace">pipeline = Pipeline([</p>
            <p style="font-family:'Lucida Console', monospace">&nbsp;&nbsp;&nbsp;&nbsp;("vectorizer", TfidfVectorizer()),</p>
            <p style="font-family:'Lucida Console', monospace">&nbsp;&nbsp;&nbsp;&nbsp;("clf", SGDClassifier()),</p>
            <p style="font-family:'Lucida Console', monospace">])</p>
            Which parameter of <span style="font-family:'Lucida Console', monospace">pipeline</span> corresponds to the parameter <span style="font-family:'Lucida Console', monospace">"max_features"</span> of <span style="font-family:'Lucida Console', monospace">TfidfVectorizer</span>?
        """))

    return Quiz(
        description,
        [
            "max_features",
            "vec__max_features",
            "vectorizer__max_features",
            "clf__max_features",
        ]
        ,
        "vectorizer__max_features"
    )


def quiz_kmeans():
    # lecture 3, quiz 1
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTMLMath(value=r"""
            Suppose you run $k$-means algorithm on 4 points:
            <ul>
                <li>$x_1 = (3, 1)$</li>
                <li>$x_2 = (4, 2)$</li>
                <li>$x_3 = (3, 3)$</li>
                <li>$x_4 = (6, 2)$</li>
            </ul>
            with initial cluster centers
            <ul>
                <li>$c_1 = (0, 2)$</li>
                <li>$c_2 = (7, 2)$</li>
            </ul>
        """))
        display(widgets.HTMLMath(value=r"Compute clusters and new coordinates of $c_1$ and $c_2$ after one full iteration, i.e. after one E-step and one M-step."))
        display(widgets.HTMLMath(value=r"""
            Options:
            <ol>
                <li>cluster 1: $x_1$ and $x_3$; cluster 2: $x_2$ and $x_4$; $c_1 = (0, 2)$; $c_2 = (7, 2)$</li>
                <li>cluster 1: $x_1$ and $x_3$; cluster 2: $x_2$ and $x_4$; $c_1 = (3, 2)$; $c_2 = (5, 2)$</li>
                <li>cluster 1: $x_1$ and $x_2$; cluster 2: $x_3$ and $x_4$; $c_1 = (0, 2)$; $c_2 = (7, 2)$</li>
                <li>cluster 1: $x_1$ and $x_2$; cluster 2: $x_3$ and $x_4$; $c_1 = (3, 2)$; $c_2 = (5, 2)$</li>
            </ol>
        """))
        display(widgets.HTML("Choose the correct answer below."))
        display(widgets.HTML("<b>Hint</b>: draw a picture."))

    return Quiz(
        description,
        ["option 1", "option 2", "option 3", "option 4"],
        "option 2"
    )


def quiz_estimate_clustering_quality():
    # lecture 3, quiz 2
    import ipywidgets as widgets
    from IPython.display import display
    from tabulate import tabulate

    description = widgets.Output()
    with description:
        display(widgets.HTMLMath(value=r"""
            Suppose you have 4 points with the following true labels and clusters:
        """))
        table = [
            ("$x_1$", "1", "2"),
            ("$x_2$", "1", "2"),
            ("$x_3$", "2", "1"),
            ("$x_4$", "1", "1"),
        ]
        display(widgets.HTMLMath(value=tabulate(table, headers=("point", "label", "cluster"), tablefmt="html")))
        display(widgets.HTML("What is the clustering quality according to the definition above?"))

    return Quiz(
        description,
        ["0", "0.25", "0.5", "0.75"],
        "0.5"
    )


def quiz_nmf():
    # lecture 3, quiz 3
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTMLMath(value=r"""
            Choose the correct NMF for the matrix
            $$
            X = \begin{pmatrix}
                    1 & 2 \\
                    2 & 4
                \end{pmatrix}
            $$
            <ul>
                <li> option 1:
                    $$
                    X =
                    \begin{pmatrix}
                        1 & 1\\
                        1 & 1
                    \end{pmatrix}
                    \cdot

                    \begin{pmatrix}
                        1 & 2 \\
                        2 & 4
                    \end{pmatrix}
                    $$
                </li>
                <li> option 2:
                    $$
                    X =
                    \begin{pmatrix}
                        1 & 0\\
                        0 & 1
                    \end{pmatrix}
                    \cdot

                    \begin{pmatrix}
                        1 & 2 \\
                        2 & 2
                    \end{pmatrix}
                    $$
                </li>
                <li> option 3:
                    $$
                    X =
                    \begin{pmatrix}
                        1 \\
                        2 
                    \end{pmatrix}
                    \cdot

                    \begin{pmatrix}
                        1 & 2
                    \end{pmatrix}
                    $$
                </li>

                <li> option 4:
                    $$
                    X =
                    \begin{pmatrix}
                        1 & 2
                    \end{pmatrix}
                    \cdot

                    \begin{pmatrix}
                        1 \\
                        2
                    \end{pmatrix}
                    $$
                </li>
            </ul>
        """))
    return Quiz(
        description,
        ["option 1", "option 2", "option 3", "option 4",],
        "option 3"
    )


def quiz_coherence():
    # lecture 3, quiz 4
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTMLMath(value=r"""
            Consider three words $A$, $B$, $C$ that occur with probabilities
            $$
            \Pr(A) = \frac{1}{2}, \Pr(B) = \frac{1}{4}, \Pr(C) = \frac{1}{2}
            $$
            and co-occur with probabilities
            $$
            \Pr(A, B) = \frac{1}{4}, \Pr(B, C) = \frac{1}{16}, \Pr(A, C) = \frac{1}{4}
            $$
            Choose the pair of words with the highest value of $PMI$.
        """))

    return Quiz(
        description,
        ["A and B", "B and C", "A and C", ],
        "A and B"
    )


def quiz_skibidi():
    # lesson 4, quiz 1
    # https://www.thisworddoesnotexist.com/w/skibidi/eyJ3IjogInNraWJpZGkiLCAiZCI6ICJhIHRhbGwsIHNsZW5kZXIsIHRhcGVyaW5nLCBicm9hZC1zaG91bGRlcmVkIGZpc2ggb2Ygd2FybSBzZWFzLiIsICJwIjogIm5vdW4iLCAiZSI6ICJhIHNraWJpZGkgaXMgY2F1Z2h0IG91dCBvZiB0aGUgc2VhIGF0IENocmlzdG1hcyJ9.EvCkp3VeiAeCzKO1ZokB96-k4Cn5VOwxTvLz11foBk8=

    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTML("""
            Consider the following sentence:
            "a skibidi is caught out of the sea at Christmas".
            
            What is "skibidi"?
            <ol>
                <li>the Chinese tea used to purify tea glasses for eye makeup and to make it more attractive</li>
                <li>a small, slender elongated tubular labyrinth, typically having a rounded body and a spiral, twisted, curved part</li>
                <li>a tall, slender, tapering, broad-shouldered fish of warm seas.</li>
                <li>the stem of the long rib of a fungus, especially one which reveals the outer edge of the flower and can be found on the stalk of a variety of trees</li>
            </ol>
            Choose the correct answer below
        """))

    return Quiz(
        description,
        [
            "definition 1",
            "definition 2",
            "definition 3",
            "definition 4",
        ],
        "definition 3"
    )


def quiz_word2vec_context():
    # lesson 4, quiz 2

    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTML("""
            Choose the correct context for the word "quick" in the following sentence:
            <br>
            <it>the quick brown fox jumps over the lazy dog</it>
            <br>
            Window size is 2.
        """))

    return Quiz(
        description,
        [
            "(the, brown)",
            "(the, quick, brown, fox)",
            "(the, brown, fox)",
            "Cannot compute the context.",
        ],
        "(the, brown, fox)",
    )


def quiz_word2vec_word_vector():
    # lesson 4, quiz 3
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTML("""
            word2vec algorithm learns two matrices: the matrix of central vectors and the matrix of context vectors.
            In the implementation above, how is the word vector computed for a given word?
        """))

    return Quiz(
        description,
        [
            "Taken from the matrix of context vectors",
            "Taken from the matrix of central vectors",
            "It is the average of central and context vectors",
            "None of the above",
        ],
        "Taken from the matrix of central vectors",
    )


def quiz_word2vec_subsampling():
    # lesson 4, quiz 4

    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTML("""
            In the implementation above, which words are subsampled (i.e. too frequent words are thrown away with some probability)?
        """))

    return Quiz(
        description,
        [
            "Central words",
            "Context words",
            "Negative samples",
            "All of them",
        ],
        "Central words",
    )


def quiz_word2vec_negative_sampling():
    # lecture 4, quiz 5

    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTML("""
            How is negative sampling performed? Why?
        """))

    return Quiz(
        description,
        [
            "without replacement; negative samples are i.i.d.",
            "with replacement; negative samples are i.i.d.",
            "without replacement; it is efficient",
            "with replacement; it is efficient",
        ],
        "with replacement; negative samples are i.i.d.",
    )


def quiz_most_similar():
    # lecture 4, quiz 6
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTML("""
            What is the most similar word to the word "twitter" according to the model "glove-wiki-gigaword-100"?
        """))

    return Quiz(
        description,
        [
            'instagram',
            'facebook',
            'youtube',
            'internet',
        ],
        "facebook",
    )


def quiz_earl():
    # lecture 4, quiz 7
    import ipywidgets as widgets
    from IPython.display import display

    description = widgets.Output()
    with description:
        display(widgets.HTML("""
            Solve the following word analogy task:
            $$
            \overrightarrow{\mathrm{earl}} - \overrightarrow{\mathrm{man}} \\approx X - \overrightarrow{\mathrm{woman}}
            $$
            Use "glove-wiki-gigaword-100".
        """))

    return Quiz(
        description,
        [
            'X = duchess',
            'X = marquess',
            'X = countess',
            'X = baronet',
        ],
        'X = countess',
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


def quiz_correct_computational_graph():
    # lecture 6, quiz 3
    import ipywidgets as widgets
    from IPython.display import display
    description = widgets.Output()
    with description:
        display(widgets.HTMLMath(value=r"""
        Choose the correct computational graph for $F(\alpha, \beta) = f\left(\dfrac{\alpha}{\beta}\right)$
        """))
    with description:
        display(widgets.HTML(
            """
            <b>Option 1</b>
            <br>
            <img src="https://raw.githubusercontent.com/horoshenkih/harbour-space-text-mining-course/master/pic/backprop-1.png">
            <br>

            <b>Option 2</b>
            <br>
            <img src="https://raw.githubusercontent.com/horoshenkih/harbour-space-text-mining-course/master/pic/backprop-2.png">
            <br>

            <b>Option 3</b>
            <br>
            <img src="https://raw.githubusercontent.com/horoshenkih/harbour-space-text-mining-course/master/pic/backprop-3.png">
            <br>

            <b>Option 4</b>
            <br>
            <img src="https://raw.githubusercontent.com/horoshenkih/harbour-space-text-mining-course/master/pic/backprop-4.png">
            <br>
            """
        ))

    return Quiz(
        description,
        ["Option 1", "Option 2", "Option 3", "Option 4"],
        "Option 4"
    )


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
