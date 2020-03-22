from argparse import ArgumentParser
import json
import os
import sys
import tempfile
from nbconvert.nbconvertapp import NbConvertApp
import re

DEFAULT_SLIDE_TYPE = "slide"
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
    args = p.parse_args()
    with open(args.ipynb_notebook_file) as nb, tempfile.TemporaryDirectory() as tmpdir, FakeSysArgv():
        nb_content = json.load(nb)
        for i, cell in enumerate(nb_content["cells"]):
            nb_content["cells"][i].setdefault("metadata", {})
            nb_content["cells"][i]["metadata"].setdefault("slideshow", {})
            if cell["cell_type"] == "markdown":
                regex = MARKDOWN_CODE_REGEX
            elif cell["cell_type"] == "code":
                regex = PY_CODE_REGEX
            slide_type = DEFAULT_SLIDE_TYPE
            for line in cell["source"]:
                m = regex.match(line)
                if m:
                    slide_type = m.group(1)
                    break
            nb_content["cells"][i]["metadata"]["slideshow"]["slide_type"] = slide_type
        temp_nb = os.path.join(tmpdir, "slides.ipynb")
        with open(temp_nb, "w") as out:
            json.dump(nb_content, out)
        converter = NbConvertApp()
        converter.notebooks = [temp_nb]
        converter.export_format = "slides"
        converter.postprocessor_class = "serve"
        converter.initialize()
        converter.convert_notebooks()


if __name__ == '__main__':
    main()
