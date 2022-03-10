from argparse import ArgumentParser
import json
import os
import sys
import shutil
import tempfile
from nbconvert.nbconvertapp import NbConvertApp
from nbconvert.preprocessors import TagRemovePreprocessor
from nbconvert.exporters import TemplateExporter
import re

PY_CODE_REGEX = re.compile(r'^#@slideshow\s+(.+)$')
MARKDOWN_CODE_REGEX = re.compile(r'^<!--@slideshow\s+(.+)-->$')
ALLOWED_SLIDE_TYPES = "slide subslide fragment skip notes".split()


class FakeSysArgv:
    def __enter__(self):
        self._argv = sys.argv[:]
        sys.argv = sys.argv[:1]

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.argv = self._argv


TagRemovePreprocessor.enabled = True
TagRemovePreprocessor.remove_input_tags = {'remove_input'}
TagRemovePreprocessor.remove_all_outputs_tags = {'remove_output'}


def main():
    p = ArgumentParser()
    p.add_argument("ipynb_notebook_file")
    p.add_argument("-o", "--out")
    p.add_argument("-d", "--default-slide-type", default="skip")
    p.add_argument("--include-input-prompt", action="store_true")
    p.add_argument("--include-output-prompt", action="store_true")
    args = p.parse_args()
    # https://nbconvert.readthedocs.io/en/latest/config_options.html?highlight=TemplateExporter.exclude
    TemplateExporter.exclude_input_prompt = not args.include_input_prompt
    TemplateExporter.exclude_output_prompt = not args.include_output_prompt

    with open(args.ipynb_notebook_file) as nb, FakeSysArgv(), tempfile.NamedTemporaryFile(suffix=".ipynb", mode="w") as out_nb:
        nb_content = json.load(nb)
        for i, cell in enumerate(nb_content["cells"]):
            # breakpoint()
            nb_content["cells"][i].setdefault("metadata", {})
            nb_content["cells"][i]["metadata"].setdefault("slideshow", {})
            nb_content["cells"][i]["metadata"].setdefault("tags", [])
            if cell["cell_type"] == "markdown":
                regex = MARKDOWN_CODE_REGEX
            elif cell["cell_type"] == "code":
                regex = PY_CODE_REGEX
            slide_type = args.default_slide_type
            tags = []
            # check the first line
            if not cell["source"]:
                continue
            m = regex.match(cell["source"][0])
            if m:
                slideshow_config = m.group(1)
                slideshow_config_items = slideshow_config.split()
                slide_type = slideshow_config_items[0]
                if slide_type not in ALLOWED_SLIDE_TYPES:
                    raise ValueError("unknown slide type: {}".format(slide_type))
                # find tags in format "tags=tag1,tag2,..."
                for item in slideshow_config_items:
                    if not item.startswith("tags="):
                        continue
                    item_tags = item[len("tags="):].split(",")
                    for tag in item_tags:
                        # add only new tags, just in case
                        if tag not in cell["metadata"].get("tags", []):
                            tags.append(tag)
                nb_content["cells"][i]["source"] = cell["source"][1:]  # remove the first line

            nb_content["cells"][i]["metadata"]["slideshow"]["slide_type"] = slide_type
            nb_content["cells"][i]["metadata"]["tags"] += tags
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
