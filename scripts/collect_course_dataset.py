from argparse import ArgumentParser
import glob
import json
import pandas as pd
from collections import defaultdict


def main():
    p = ArgumentParser()
    p.add_argument("dir")
    p.add_argument("out")
    args = p.parse_args()
    notebooks = sorted(glob.glob(args.dir + "/*.ipynb"))

    data = defaultdict(list)
    for notebook in notebooks:
        if "MISC" in notebook:
            continue
        fname = notebook.split("/")[-1]
        with open(notebook) as nb:
            nb_content = json.load(nb)

            for i, cell in enumerate(nb_content["cells"]):
                if not cell["source"]:
                    continue

                cell_source = cell["source"][:]
                if "@slideshow slide" in cell_source[0]:
                    cell_source = cell_source[1:]
                    slide_type = "slide"
                elif "@slideshow fragment" in cell_source[0]:
                    cell_source = cell_source[1:]
                    slide_type = "slide"
                else:
                    slide_type = "-"

                cell_type = cell["cell_type"]
                n_lines = len(cell_source)
                cell_text = "\t".join([s.strip() for s in cell_source])

                data["filename"].append(fname)
                data["cell_index"].append(i)
                data["cell_type"].append(cell_type)
                data["slide_type"].append(slide_type)
                data["n_lines"].append(n_lines)
                data["cell_text"].append(cell_text)

    df = pd.DataFrame(data)
    df.to_csv(args.out)


if __name__ == '__main__':
    main()
