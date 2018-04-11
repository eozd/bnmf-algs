#!/bin/bash

filepath=$1

python scripts/celero_to_latex.py ${filepath} "" "" | pandoc --from=latex --to=markdown_github
