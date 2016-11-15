#!/usr/bin/env python
# encoding: utf-8

"""
Stack-based abstract virtual machine
"""

# from __future__ import (absolute_import, print_function)

__version__ = '4.0.1'

from .machine import VM, BaseMachine
from .ebnf import EBNFParser as Parser
from .coder import Coder