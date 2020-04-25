from argparse import ArgumentParser
import json
import os
import sys
import shutil
import tempfile
from nbconvert.nbconvertapp import NbConvertApp
import re

PY_CODE_REGEX = re.compile(r'^#@slideshow (slide|subslide|fragment|skip|notes)\s*$')
MARKDOWN_CODE_REGEX = re.compile(r'<!--@slideshow (slide|subslide|fragment|skip|notes)-->')


class FakeSysArgv:
    def __enter__(self):
        self._argv = sys.argv[:]
        sys.argv = sys.argv[:1]

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.argv = self._argv


def main():
    p = ArgumentParser()
    p.add_argument("ipynb_notebook_file")
    p.add_argument("-o", "--out")
    p.add_argument("-d", "--default-slide-type", default="skip")
    args = p.parse_args()

    with open(args.ipynb_notebook_file) as nb, FakeSysArgv(), tempfile.NamedTemporaryFile(suffix=".ipynb", mode="w") as out_nb:
        nb_content = json.load(nb)
        for i, cell in enumerate(nb_content["cells"]):
            nb_content["cells"][i].setdefault("metadata", {})
            nb_content["cells"][i]["metadata"].setdefault("slideshow", {})
            if cell["cell_type"] == "markdown":
                regex = MARKDOWN_CODE_REGEX
            elif cell["cell_type"] == "code":
                regex = PY_CODE_REGEX
            slide_type = args.default_slide_type
            for line in cell["source"]:
                m = regex.match(line)
                if m:
                    slide_type = m.group(1)
                    break
            nb_content["cells"][i]["metadata"]["slideshow"]["slide_type"] = slide_type
        json.dump(nb_content, out_nb)
        out_nb.flush()
        converter = NbConvertApp()
        converter.notebooks = [out_nb.name]
        converter.export_format = "slides"
        if not args.out:
            converter.postprocessor_class = "serve"
        converter.initialize()
        converter.convert_notebooks()

        if args.out:
            base = os.path.splitext(out_nb.name)[0]
            shutil.copy(base + ".slides.html", args.out)


if __name__ == '__main__':
    main()
