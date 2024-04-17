#!/bin/bash

isort datum/*/*.py
autopep8 --in-place  datum/*/*.py
