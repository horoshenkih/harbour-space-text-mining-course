def demo_computational_graph(
        connections,
        labels=None,
        title=None,
        figsize=(18, 10),
        font_size=20,
        node_size=None,
        forward_color="k",
        backward_color="r",
        static=False,
        forward_idx=0,
        backward_idx=0
):
    # each connection is a tuple (src, dst, forward, backward)
    # labels is a dict with mapping (node, label)
    import matplotlib.pyplot as plt
    from networkx import nx
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
