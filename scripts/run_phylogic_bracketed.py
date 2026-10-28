"""Run public PhylogicNDT unchanged except for the numerical-only Stirling root bracket fallback (scripts/phylogic_root_bracket.py).

Valid original brackets are passed through untouched, so the model and priors are exactly PhylogicNDT's; the fallback only widens
a bracket that misses the positive root (the 'f(a) and f(b) must have different signs' crash at large mutation counts).
Usage (Python 2.7 runtime, from the work directory): run_phylogic_bracketed.py <PhylogicNDT arguments...>
"""
from __future__ import print_function
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'external', 'phylogicndt'))
from phylogic_root_bracket import bracketed_solver  # noqa: E402
from Cluster import DpEngine  # noqa: E402

DpEngine.root_scalar = bracketed_solver(DpEngine.root_scalar)
script = os.path.join(ROOT, 'external', 'phylogicndt', 'PhylogicNDT.py')
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name='__main__')
