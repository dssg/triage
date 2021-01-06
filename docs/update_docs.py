import shutil

from md_autogen import MarkdownAPIGenerator
from md_autogen import to_md_file

from triage import experiments


def generate_api_docs():
    modules = [
        experiments.base,
        experiments.singlethreaded,
        experiments.multicore
    ]

    md_gen = MarkdownAPIGenerator("triage", "https://github.com/dssg/triage/tree/master")
    for m in modules:
        md_string = md_gen.module2md(m)
        to_md_file(md_string, m.__name__, "docs/sources")


def update_index_md():
    shutil.copyfile('README.md', 'docs/sources/index.md')


def copy_templates():
    shutil.rmtree('docs/sources', ignore_errors=True)
    shutil.copytree('docs/templates', 'docs/sources')


if __name__ == "__main__":
    #copy_templates()
    update_index_md()
    #generate_api_docs()
