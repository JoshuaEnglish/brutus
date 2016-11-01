# encoding: utf-8

"""
Stack-based abstract virtual machine
"""

from __future__ import (absolute_import, print_function)

__version__ = '3.0.1'

from machine import VM, BaseMachine
from library import (VMLibrary, FiveFunctionLibrary, 
                     ComparisonsLibrary, ControlOperationsLibrary)