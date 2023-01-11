def demo_generate_text_ngram(language_model, prefix, seed=0):
    # demo for lesson 2
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    from tabulate import tabulate
    from itertools import islice
    import numpy as np

    np.random.seed(seed)
    generated_text = prefix[:]
    left_layout = widgets.Layout(width="25%")

    def split_head_tail(tokens, n):
        tail = tokens[-(n-1):]
        head = tokens[:-len(tail)]
        return head, tail

    def tokens_window_html(tokens, n):
        head, tail = split_head_tail(tokens, n)
        return " ".join(head) + " " + "<span style='border: 2px solid green;'>" + " ".join(tail) + "</span>"

    def get_token_probabilities(language_model, prefix):
        all_tokens = list(language_model.vocab)
        all_token_probabilities = np.array([language_model.predict_token_probability(generated_text, token) for token in all_tokens])
        return zip(all_tokens, all_token_probabilities)

    def get_prefix_count(language_model, prefix):
        n = language_model.n
        _, tail = split_head_tail([None for _ in range(n-1)] + prefix, n)
        return language_model.nm1_grams_counter[tuple(tail)]

    def generate_next_token(language_model, prefix):
        tokens, probs = zip(*get_token_probabilities(language_model, prefix))
        return np.random.choice(tokens, size=1, p=probs)[0]

    def update_widgets(
        language_model,
        generated_text,
        widget_text_with_sliding_window,
        widget_probabilities_table,
        widget_current_window_count
    ):
        n = language_model.n
        token_probabilities = get_token_probabilities(language_model, generated_text)
        with widget_text_with_sliding_window:
            clear_output()
            display(HTML(tokens_window_html(generated_text, n)))
        with widget_probabilities_table:
            clear_output()
            table = [t_p for t_p in islice(sorted(token_probabilities, key=lambda t_p: -t_p[1]), 0, 10)]
            table_html = tabulate(table, ["token", "probability"], tablefmt="html")
            table_html = table_html.replace("<table>", "<table style='width:100%;border:1px solid'>")
            display(HTML(table_html))
        with widget_current_window_count:
            clear_output()
            _, tail = split_head_tail(generated_text, language_model.n)
            cnt = get_prefix_count(language_model, generated_text)
            print("count(\"{}\") = {}".format(" ".join(tail), cnt))

    widget_button = widgets.Button(description="generate next token")
    widget_probabilities_table = widgets.Output(layout=left_layout)
    widget_text_with_sliding_window = widgets.Output()
    widget_current_window_count = widgets.Output()

    def on_button_clicked(b):
        nonlocal generated_text
        nonlocal language_model
        next_token = generate_next_token(language_model, generated_text)
        if next_token is None:
            with widget_probabilities_table:
                clear_output()
            with widget_text_with_sliding_window:
                clear_output()
                display(HTML("<font color='green'>" + " ".join(generated_text) + "</font>"))
            with widget_current_window_count:
                clear_output()
        else:
            generated_text += [next_token]
            update_widgets(
                language_model,
                generated_text,
                widget_text_with_sliding_window,
                widget_probabilities_table,
                widget_current_window_count
            )
    widget_button.on_click(on_button_clicked)

    display(
        widgets.HBox([
            widgets.VBox([widget_button, widget_probabilities_table], layout=left_layout),
            widgets.VBox([widget_text_with_sliding_window, widget_current_window_count])
        ])
    )
    update_widgets(
        language_model,
        generated_text,
        widget_text_with_sliding_window,
        widget_probabilities_table,
        widget_current_window_count
    )


def demo_kmeans(
        num_samples=300,
        num_centers=4,
        clusters_random_state=0,
        k=4,
        kmeans_random_state=2,
        cluster_std=0.6,
        figsize=(8, 8),
):
    # demo for lesson 3
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import pairwise_distances_argmin
    from sklearn.datasets import make_blobs
    from IPython.display import display, HTML
    import warnings
    warnings.filterwarnings('ignore')

    X, y = make_blobs(
        n_samples=num_samples,
        centers=num_centers,
        random_state=clusters_random_state,
        cluster_std=cluster_std
    )
    x_min = np.min(X[:, 0])
    x_max = np.max(X[:, 0])
    y_min = np.min(X[:, 1])
    y_max = np.max(X[:, 1])
    rng = np.random.RandomState(kmeans_random_state)
    labels = np.zeros(X.shape[0])
    centers = rng.rand(k, 2)
    # scale centers
    centers[:, 0] *= (x_max - x_min)
    centers[:, 0] += x_min
    centers[:, 1] *= (y_max - y_min)
    centers[:, 1] += y_min

    def plot_points(X, labels):
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis',
                    vmin=0, vmax=k - 1);

    def plot_centers(centers):
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c=np.arange(centers.shape[0]),
                    s=200, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c='black', s=50)

    CURRENT_STEP = 0

    out = widgets.Output()
    out_next_step = widgets.Output()

    def step():
        nonlocal CURRENT_STEP
        nonlocal labels
        nonlocal centers
        if CURRENT_STEP % 2 == 0:
            # expectation
            labels = pairwise_distances_argmin(X, centers)
        else:
            # maximization
            old_centers = centers
            centers = np.array([X[labels == j].mean(0) for j in range(k)])
            nans = np.isnan(centers)
            centers[nans] = old_centers[nans]
        CURRENT_STEP += 1

    def visualize():
        plt.figure(figsize=figsize)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        with out:
            out.clear_output()
            plt.axis("equal")
            if CURRENT_STEP == 0:
                plt.scatter(X[:, 0], X[:, 1], s=50)
            else:
                plot_points(X, labels)
            plot_centers(centers)
            plt.show()
        with out_next_step:
            out_next_step.clear_output()
            if CURRENT_STEP % 2 == 0:
                display(HTML("next step: re-assign labels (E-step)"))
            else:
                display(HTML("next step: update centroids (M-step)"))

    button = widgets.Button(description="update")

    def update(b):
        step()
        visualize()

    button.on_click(update)
    visualize()

    return widgets.VBox([out_next_step, button, out])


def demo_word2vec_batch(tokens, window_size):
    # demo for lesson 4
    import ipywidgets as widgets
    from IPython.display import display, HTML

    POS = 0

    out = widgets.Output()

    def visualize():
        with out:
            out.clear_output()
            left_border = max(0, POS - window_size)
            right_border = min(len(tokens), POS + window_size + 1)
            center = tokens[POS]
            left_tail = tokens[:left_border]
            left_context = tokens[left_border:POS]
            right_context = tokens[POS + 1:right_border]
            right_tail = tokens[right_border:]
            tokens_html = " ".join([
                " ".join(left_tail),
                "<span style='border: 2px solid red;'>" + " ".join(left_context) + "</span>",
                "<span style='border: 2px solid green;'>" + center + "</span>",
                "<span style='border: 2px solid red;'>" + " ".join(right_context) + "</span>",
                " ".join(right_tail)
            ])
            display(HTML(tokens_html))
            print(f"Position: {POS}")
            print(f"Center word: '{center}'")
            print(f"Context: {left_context + right_context}")

    button_next = widgets.Button(description=">>")

    def inc(b):
        nonlocal POS
        if POS < len(tokens) - 1:
            POS += 1
        visualize()
    button_next.on_click(inc)
    button_prev = widgets.Button(description="<<")

    def dec(b):
        nonlocal POS
        if POS > 0:
            POS -= 1
        visualize()
    button_prev.on_click(dec)

    visualize()
    return widgets.VBox([widgets.HBox([button_prev, button_next]), out])


def demo_function_approximation(
        num_functions=1,
        default_transform="step",
        seed=0,
        figsize=(12, 6),
        static=False,
        **init_weights
):
    # demo for lesson 6
    from ipywidgets import interactive_output
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(seed)

    def f(transform, **kwargs):
        """
        kwargs is a dict with keys "w1", "w2", ..., "wK", "a1", "a2", ..., "aK", "b1", "b2", ..., "bK"
        """
        weights = {}
        slopes = {}
        biases = {}
        for k, v in kwargs.items():
            t = k[0]
            idx = k[1:]
            if t == "w":
                weights[idx] = v
            elif t == "a":
                slopes[idx] = v
            elif t == "b":
                biases[idx] = v
            else:
                raise ValueError(k)
        assert set(weights.keys()) == set(slopes.keys()) == set(biases.keys())

        plt.figure(2, figsize=figsize)
        N = 300
        x = np.linspace(-1, 2, num=N)
        true_values = np.cosh(x)
        plt.plot(x, true_values)
        if transform == "identity":
            f = lambda x: x
        elif transform == "step":
            f = lambda x: np.heaviside(x, 1)
        elif transform == "relu":
            f = lambda x: np.maximum(x, 0)
        else:
            raise ValueError(transform)

        approx_values = np.zeros(N)
        for i in weights.keys():
            w = weights[i]
            a = slopes[i]
            b = biases[i]
            approx_values += w * f(a * x + b)

        plt.plot(x, approx_values)
        print("MSE:", np.mean((true_values - approx_values)**2))
        plt.ylim(-1, 5)
        plt.show()

    if static:
        f(default_transform, **init_weights)
        return

    ui_elements = []
    kwargs = {}
    for i in range(num_functions):
        k = str(i+1)

        wk = "w" + k
        weights_range = widgets.FloatSlider(min=-5, max=5, step=0.1, value=init_weights.get(wk, 1), description=wk)
        kwargs[wk] = weights_range

        ak = "a" + k
        slopes_range = widgets.FloatSlider(min=-20, max=20, step=0.1, value=init_weights.get(ak, 1), description=ak)
        kwargs[ak] = slopes_range

        bk = "b" + k
        biases_range = widgets.FloatSlider(min=-10, max=10, step=0.1, value=init_weights.get(bk, 0), description=bk)
        kwargs[bk] = biases_range

        controls = [weights_range, slopes_range, biases_range]
        ui_elements.append(widgets.HBox([widgets.Label("f"+k)] + controls))

    transforms = widgets.Dropdown(options=["step", "relu", "identity"], description=r"transform", value=default_transform)
    kwargs["transform"] = transforms

    demo = interactive_output(f, kwargs)
    return widgets.VBox([widgets.VBox(ui_elements), transforms, demo])


def demo_gradient_descent(
    f,
    theta_0,
    learning_rate,
    theta_min=-2,
    theta_max=2,
    y_min=-1,
    y_max=4,
    figsize=(12, 6),
    eps=1e-5
):
    # demo for lesson 6
    import numpy as np
    import matplotlib.pyplot as plt
    import ipywidgets as widgets

    def grad(theta):
        return (f(theta + eps) - f(theta - eps)) / (2 * eps)

    out = widgets.Output()
    PATH = [(theta_0, f(theta_0))]
    THETA = np.linspace(theta_min, theta_max, 100)
    F = [f(t) for t in THETA]

    def step():
        prev_theta, prev_f = PATH[-1]
        theta = prev_theta - learning_rate * grad(prev_theta)
        PATH.append((theta, f(theta)))

    def visualize():
        plt.figure(figsize=figsize)
        plt.ylim(y_min, y_max)
        with out:
            out.clear_output()
            plt.plot(THETA, F)
            for _i in PATH:
                plt.plot(_i[0], _i[1], 'ro')
            # visualize last point
            last_theta, last_f = PATH[-1]
            nabla_f = grad(last_theta)
            step = -learning_rate * nabla_f
            annot_theta = r"$\theta = {:.2f}$".format(last_theta)
            annot_grad = r"$\nabla f(\theta) = {:.2f}$".format(nabla_f)
            annot_step = r"$-\lambda \cdot \nabla f(\theta) = {:.2f}$".format(step)
            # draw an arrow using plt.annotate
            plt.annotate(
                "",
                xy=(last_theta+step, f(last_theta+step)),
                xytext=(last_theta, last_f),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="angle,angleA=180,angleB=-90,rad=0",
                    color="r",
                )
            )
            # annotate step
            plt.annotate(
                annot_step,
                (last_theta + 0.5 * step, last_f - 0.05 * (y_max - y_min)),
                c="r",
                ha="center"
            )
            plt.annotate(
                annot_theta + "\n" + annot_grad,
                (last_theta, last_f),
                c="r"
            )

            plt.show()

    button = widgets.Button(description="next step")
    def update(b):
        step()
        visualize()

    button.on_click(update)
    visualize()

    return widgets.VBox([button, out])


def demo_computational_graph(
        connections,
        labels=None,
        title=None,
        figsize=(12, 6),
        font_size=12,
        node_size=None,
        forward_color="k",
        backward_color="r",
        static=False,
        forward_idx=0,
        scale_x=1.0,
        scale_y=1.0,
        backward_idx=0
):
    # demo for lessons 6, 7 (neural networks)
    # each connection is a tuple (src, dst, forward, backward)
    # labels is a dict with mapping (node, label)
    import matplotlib.pyplot as plt
    import networkx as nx
    import ipywidgets as widgets
    import math
    import numpy as np

    if labels is None:
        labels = {}
    if node_size is None:
        node_size = font_size ** 2

    G_forward = nx.DiGraph()
    G_backward = nx.DiGraph()
    # forward passes
    G_forward.add_edges_from([(c[0], c[1], dict(label=c[2])) for c in connections])
    # backward passes
    G_backward.add_edges_from([(c[1], c[0], dict(label=c[3])) for c in connections])

    # topological sort of edges
    def order_edges(G):
        return list(nx.lexicographical_topological_sort(nx.line_graph(G)))

    forward_edge_label = [(e, G_forward.edges[e]["label"]) for e in order_edges(G_forward)]
    backward_edge_label = [(e, G_backward.edges[e]["label"]) for e in order_edges(G_backward)]

    # place nodes left-to-right
    # "regular" top-to-bottom layout
    nodes_layout = nx.nx_pydot.graphviz_layout(G_forward, prog='dot')
    # make left-to-right from top-to-bottom
    nodes_layout = {k: (-v[1], v[0]) for k, v in nodes_layout.items()}
    for node, coords in nodes_layout.items():
        x, y = coords
        nodes_layout[node] = (x * scale_x, y * scale_y)

    # fill in node labels
    # by default, the node label is the node itself
    node_labels = {}
    for node in nodes_layout:
        node_labels[node] = labels.get(node, node)

    # visualize
    x_min = min([c[0] for c in nodes_layout.values()])
    x_max = max([c[0] for c in nodes_layout.values()])
    y_min = min([c[1] for c in nodes_layout.values()])
    y_max = max([c[1] for c in nodes_layout.values()])
    padding = 10

    def visualize(forward_idx=0, backward_idx=0):
        plt.figure(figsize=figsize)
        plt.xlim(x_min - padding, x_max + padding)
        plt.ylim(y_min - padding, y_max + padding)
        plt.axis('equal')
        if title:
            plt.title(r"Computational graph for " + title, fontsize=font_size)

        # draw nodes
        nx.draw_networkx_nodes(G_forward, node_color="#a2c4fc", pos=nodes_layout, node_size=node_size)
        nx.draw_networkx_labels(G_forward, pos=nodes_layout, labels=node_labels, font_size=font_size)
        edges_kwargs = dict(pos=nodes_layout, connectionstyle="arc3,rad=-0.1", width=2)
        # draw forward edges
        nx.draw_networkx_edges(G_forward, edge_color=forward_color, alpha=0.5, **edges_kwargs)
        # draw backward edges
        nx.draw_networkx_edges(G_backward, edge_color=backward_color, alpha=0.5, **edges_kwargs)

        # draw labels
        def label_coordinates(v1, v2, norm_fraction=0.15, backward=False):
            v1 = np.array(v1)
            v2 = np.array(v2)
            center = 0.5 * (v1 + v2)
            if backward:
                d = v1 - v2
            else:
                d = v2 - v1
            norm = np.sqrt(d @ d)
            d /= norm
            shift = np.array([-d[1], d[0]]) * norm_fraction * norm
            if backward:
                shift = -shift
            angle = math.degrees(math.atan2(d[1], d[0]))
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            return tuple(center + shift), angle

        active_forward_edges = []
        for e, l in forward_edge_label[:forward_idx]:
            active_forward_edges.append(e)
            norm_fraction = 0.1 + 0.05 * len(l.split("\n"))  # larger shift for multiline labels
            coords, angle = label_coordinates(nodes_layout[e[0]], nodes_layout[e[1]], norm_fraction=norm_fraction)
            plt.annotate(l, coords, color=forward_color, size=font_size, ha='center', rotation=angle,
                         rotation_mode='anchor')

        # highlight forward edges
        nx.draw_networkx_edges(G_forward, edgelist=active_forward_edges, edge_color=forward_color, alpha=1,
                               **edges_kwargs)

        active_backward_edges = []
        for e, l in backward_edge_label[:backward_idx]:
            active_backward_edges.append(e)
            norm_fraction = 0.1 + 0.05 * len(l.split("\n"))  # larger shift for multiline labels
            coords, angle = label_coordinates(nodes_layout[e[0]], nodes_layout[e[1]], norm_fraction=norm_fraction,
                                              backward=True)
            plt.annotate(l, coords, color=backward_color, size=font_size, ha='center', rotation=angle,
                         rotation_mode='anchor')

        # highlight backward edges
        nx.draw_networkx_edges(G_backward, edgelist=active_backward_edges, edge_color=backward_color, alpha=1,
                               **edges_kwargs)

        plt.show()

    if static:
        visualize(forward_idx=forward_idx, backward_idx=backward_idx)
        return

    kwargs = {
        "forward_idx": widgets.IntSlider(min=0, max=len(forward_edge_label), value=forward_idx),
        "backward_idx": widgets.IntSlider(min=0, max=len(backward_edge_label), value=backward_idx),
    }

    def inc(slider):
        if slider.value < slider.max:
            slider.value = slider.value + 1

    def dec(slider):
        if slider.value > slider.min:
            slider.value = slider.value - 1

    button_layout = widgets.Layout(width='100%')
    forward_dec_button = widgets.Button(description="-", layout=button_layout)
    forward_dec_button.on_click(lambda b: dec(kwargs["forward_idx"]))
    forward_inc_button = widgets.Button(description="+", layout=button_layout)
    forward_inc_button.on_click(lambda b: inc(kwargs["forward_idx"]))
    backward_dec_button = widgets.Button(description="-", layout=button_layout)
    backward_dec_button.on_click(lambda b: dec(kwargs["backward_idx"]))
    backward_inc_button = widgets.Button(description="+", layout=button_layout)
    backward_inc_button.on_click(lambda b: inc(kwargs["backward_idx"]))

    labels = widgets.VBox([widgets.Label("forward"), widgets.Label("backward")])
    dec_buttons = widgets.VBox([forward_dec_button, backward_dec_button])
    sliders = widgets.VBox([kwargs["forward_idx"], kwargs["backward_idx"]])
    inc_buttons = widgets.VBox([forward_inc_button, backward_inc_button])
    demo = widgets.interactive_output(visualize, kwargs)

    return widgets.VBox([widgets.HBox([labels, dec_buttons, sliders, inc_buttons]), demo])


def demo_universal_approximator(K=1, font_size=12, node_size=400):
    connections = []
    labels = {}
    x = r"$x$"
    sum_all = "++"
    labels[sum_all] = r"$h(x)$"
    loss = r"$l(y, h(x))$"
    for k in range(1, K+1):
        a = r"$a_{}$".format(k)
        b = r"$b_{}$".format(k)
        w = r"$w_{}$".format(k)
        a_times_x = "a*x_"+str(k)
        labels[a_times_x] = r"$\times$"  # the same label for many sum functions
        b_times_1 = "b*1_"+str(k)
        labels[b_times_1] = r"$\times$"  # the same label for many sum functions
        ax_plus_b = "+"+str(k)
        labels[ax_plus_b] = "+"  # the same label for many sum functions
        g = "g"+str(k)
        labels[g] = "g"
        w_times_g = "w*g_"+str(k)
        labels[w_times_g] = r"$\times$"  # the same label for many sum functions
        connections.append((x, a_times_x, "x", r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k} \cdot g'(ax_{k}+b_{k}) \cdot a_{k}$".format(k=k)))
        connections.append((a, a_times_x, a, r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k} \cdot g'(ax_{k}+b_{k}) \cdot x$".format(k=k) + " "*16))
        connections.append(("1", b_times_1, "1", r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k} \cdot g'(ax_{k}+b_{k}) \cdot b_{k}$".format(k=k)))
        connections.append((b, b_times_1, b, r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k} \cdot g'(ax_{k}+b_{k}) \cdot 1$".format(k=k) + " "*10))
        connections.append((a_times_x, ax_plus_b, a + r"$\cdot$" + x, r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k} \cdot g'(ax_{k}+b_{k})$".format(k=k)))
        connections.append((b_times_1, ax_plus_b, b, r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k} \cdot g'(ax_{k}+b_{k})$".format(k=k)))
        connections.append((ax_plus_b, g, r"$a_{k}x+b_{k}$".format(k=k), r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k} \cdot g'(ax_{k}+b_{k})$".format(k=k)))
        connections.append((g, w_times_g, r"$g(a_{k}x+b_{k})$".format(k=k), r"$\dfrac{{\partial l}}{{\partial h}} \cdot w_{k}$".format(k=k)))
        connections.append((w, w_times_g, w, r"$\dfrac{{\partial l}}{{\partial h}} \cdot g(a_{k}x+b_{k})$".format(k=k)))
        connections.append((w_times_g, sum_all, r"$w_{k} g(a_{k}x+b_{k})$".format(k=k), r"$\dfrac{{\partial l}}{{\partial h}}$".format(k=k)))
        connections.append((sum_all, loss, r"$\sum_k w_{k} g(a_{k}x+b_{k})\left[\equiv h(x)\right]$", r"$\dfrac{\partial l}{\partial h} \left[ = 2(y - h(x))\right]$"))
    return demo_computational_graph(connections, labels, font_size=font_size, node_size=node_size)


def demo_pytorch_computational_graph(
    t,
    init_gradient=None,
    figsize=(12, 6),
    padding=40
):
    import torch
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt

    if init_gradient is None:
        init_gradient = torch.tensor(1.0)

    G_forward = nx.DiGraph()  # for layout
    G = nx.MultiDiGraph()
    stack = [(t.grad_fn, t.grad_fn(init_gradient))]
    node_labels = {}
    while stack:
        node, outer_gradient = stack.pop()
        node_id = str(id(node)) + node.name()
        node_labels[node_id] = node.name()
        for n, n_outer_gradient in zip(node.next_functions, outer_gradient):
            if n[0] is not None:
                n_id = str(id(n[0])) + n[0].name()
                node_labels[n_id] = f'"{n[0].name()}"'
                G_forward.add_edge(n_id, node_id)
                G.add_edge(node_id, n_id, label=n_outer_gradient.item())
                stack.append((n[0], n[0](n_outer_gradient)))
    # place nodes left-to-right
    # "regular" top-to-bottom layout
    nodes_layout = nx.nx_pydot.graphviz_layout(G_forward, prog='dot')
    # make right-to-left from top-to-bottom
    nodes_layout = {k: (-v[1], v[0]) for k, v in nodes_layout.items()}

    # visualize
    x_min = min([c[0] for c in nodes_layout.values()])
    x_max = max([c[0] for c in nodes_layout.values()])
    y_min = min([c[1] for c in nodes_layout.values()])
    y_max = max([c[1] for c in nodes_layout.values()])

    plt.figure(figsize=figsize)
    plt.xlim(x_min - padding, x_max + padding)
    plt.ylim(y_min - padding, y_max + padding)
    # draw nodes
    nx.draw_networkx_nodes(G, node_color="#a2c4fc", pos=nodes_layout)
    nx.draw_networkx_labels(G, pos=nodes_layout, labels=node_labels)
    # draw edges
    for e in G.edges:
        l = G.edges[e]["label"]
        current_multiplicity = e[2]
        coords = tuple(0.5 * (np.array(nodes_layout[e[0]]) + np.array(nodes_layout[e[1]])) - 10 * current_multiplicity)
        plt.annotate(l, coords, color="r", ha='center')
        # hacky arrows
        plt.annotate(
            "",
            xy=nodes_layout[e[1]], xycoords='data',
            xytext=nodes_layout[e[0]], textcoords='data',
            arrowprops=dict(
                arrowstyle="simple",
                color="r",
                alpha=0.5,
                shrinkA=10, shrinkB=10,
                connectionstyle="arc3,rad={}".format(-0.1*(1 + current_multiplicity))
            ),
        )


def demo_2d_classification(f, title=None, show_zero=False):
    # f(X, Y) is a 2d-function with the domain (-1, 1) for both X and Y
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import ma
    from matplotlib import ticker, cm

    N = 100
    x = np.linspace(-1.0, 1.0, N)
    y = np.linspace(-1.0, 1.0, N)

    X, Y = np.meshgrid(x, y)
    z = f(X, Y)

    # plot 3d
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, z, cmap=cm.coolwarm_r, alpha=0.7)
    if show_zero:
        ax.plot_wireframe(X, Y, 0*z, alpha=0.5, rcount=20, ccount=20)
    cset = ax.contourf(X, Y, z, levels=30, cmap=cm.coolwarm_r, zdir="z", offset=-8)
    cbar = fig.colorbar(cset)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-8, 2)
    if title:
        ax.set_title(title)

    plt.show()
