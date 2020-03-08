import arxiv
from collections import defaultdict
from pprint import pprint
from tqdm import tqdm
import random
random.seed(0)

PAPERS_TO_INCLUDE = [
    "cond-mat/9910332",  # Emergence of scaling in random networks
]
TARGET_CATEGORY = "cond-mat"
FRACTION = 0.01

def get_target(category):
    return int(category.startswith(TARGET_CATEGORY))

# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

with open("pscp-reversed-graph.txt") as rg, open("pscp-categories.txt") as c, open("train.tsv", "w") as train:
    # read the data
    arxiv_id2category = dict([_.strip().split(";") for _ in tqdm(c, desc="arXiv ID -> category")])
    graph = defaultdict(list)
    for line in tqdm(rg, desc="graph"):
        src, dst = line.strip().split(";")
        graph[src].append(dst)

    # sample the data
    arxiv_ids = []
    for arxiv_id in tqdm(arxiv_id2category, desc="sample", total=len(arxiv_id2category)):
        if arxiv_id in PAPERS_TO_INCLUDE or random.random() < FRACTION:
            arxiv_ids.append(arxiv_id)

    # extract texts
    texts = []
    batch_size = 100
    for batch_arxiv_ids in tqdm(batch(arxiv_ids, 100), desc="get texts", total=len(arxiv_ids) // batch_size):
        batch_texts = []
        for arxiv_id, item in zip(batch_arxiv_ids, arxiv.query(id_list=batch_arxiv_ids)):
            title = item["title"]
            summary = item["summary"]
            true_category = item["arxiv_primary_category"]["term"]
            category = arxiv_id2category[arxiv_id]
            if true_category != category:
                print("Wrong category for arXiv ID {}: expected {}, got {}".format(arxiv_id, true_category, category))
            batch_texts.append((title, summary))
        texts += batch_texts

    # extract targets
    for arxiv_id, paper_texts in zip(arxiv_ids, texts):
        title, summary = paper_texts
        category = arxiv_id2category[arxiv_id]
        t = get_target(category)
        # references may have the same category or be a child category
        ref_same_category = [int(arxiv_id2category[_] == category or arxiv_id2category[_].split(".")[0] == category) for _ in graph[arxiv_id]]
        train.write("\t".join(map(str, [arxiv_id, " ".join(title.split()), " ".join(summary.split()), t, len(ref_same_category), sum(ref_same_category)])) + "\n")
