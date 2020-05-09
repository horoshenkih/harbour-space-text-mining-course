from collections import defaultdict
from datetime import timedelta, date, datetime
import typing as tp


def calendar_table(
        times: tp.List[tp.Union[date, datetime]],
        values: tp.List[tp.Any],
        weights: tp.List[float],
        from_datetime: tp.Optional[tp.Union[date, datetime]] = None,
        to_datetime: tp.Optional[tp.Union[date, datetime]] = None,
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
    th, td {
      padding: 8px;
      text-align: center;
      border-bottom: 1px solid #ddd;
    }
    </style>"""
    html = style + '<table><tr>{}</tr><tr>{}</tr></table>'.format(
        ''.join(['<th>{}</th>'.format(h) for h in header]),
        '</tr><tr>'.join('<td>{}</td>'.format('</td><td>'.join(str(_) for _ in row)) for row in table)
    )
    return html


def visualize_word_counter(word_counter, start, title, shift=10):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22})

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))

    ordered_word_count = word_counter.most_common()[start:start+shift]
    x_pos = range(len(ordered_word_count))
    words = [x[0] for x in reversed(ordered_word_count)]
    counts = [x[1] for x in reversed(ordered_word_count)]
    plt.barh(x_pos, counts, color="green")
    plt.yticks(x_pos, words)
    plt.ylabel("Lemma")
    plt.xlabel("Frequency")
    plt.title(title)
    plt.show()
