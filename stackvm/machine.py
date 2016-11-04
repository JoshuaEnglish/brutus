"""vm.py
The Virtual Machine
"""
from __future__ import (absolute_import, print_function)

import collections
import logging
LOG = logging.getLogger('STACKVM')

from errors import (LibraryError, LibraryImportError,
                    MissingMethod, RunTimeError, RuleNameError,
                    FunctionNameError, CallerError)
from stack import Stack
from library import VMLibrary, ControlOperationsLibrary
from tokenizer import VMLexer


__version__ = "4.0"


class Namespace(collections.MutableMapping):
    """Namespace()
    Storage for variables.
    Allows access to variables through case-insenstive methods.
    All internal keys are in upper case
    """
    # ``__init__`` method required to create instance from class.
    def __init__(self, *args, **kwargs):
        '''Use the object dict'''
        self.__dict__.update(*args, **kwargs)
    # The next five methods are requirements of the ABC.
    def __setitem__(self, key, value):
        self.__dict__[key.upper()] = value
    def __getitem__(self, key):
        try:
            return self.__dict__[key.upper()]
        except AttributeError:
            return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key.upper()]
    def __iter__(self):
        return iter(self.__dict__)
    def __len__(self):
        return len(self.__dict__)
    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        '''returns simple dict representation of the mapping'''
        return str(self.__dict__)
    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return '{}, Namespace({})'.format(super(Namespace).__repr__(),
                                          self.__dict__)

### This is the underlying VM
class VM(object):
    """VM(name,version)

    Virtual Machine for Stack-Based languages.
    Big items here are the Rules list and the Reservedwords lists
    Rules is a list of tuples:
        (list of reserved words,function,caller)
        The first parameter is a list of strings that are reserved words
        The second parameter is the name of a function
        The third parameter is an optional string that is passed to the function

        Reserved words and functions may not have any white space.
        "Plus" and "  plus  "
        are treated as the same reserved words. Reserved words are case
        insensitive.

        Adding a Rule will add the list of reserved words to the class
        objects list of reserved words.

    """

    def __init__(self, name, version):
        self.name = str(name)
        self.version = float(version)
        self.rules = [(['END', '.'], 'terminate', None)]
        self.reservedwords = ['END', '.']
        self.stack = Stack()
        self.stack.clear()
        self.registers = Namespace()
        self.imported_methods = {}
        #self.allow_new_tokens = True
        self._lexerclass = VMLexer

        self._program = []
        self._continue = 1
        self._curline = 0
        self._cycles = 0
        self._exports = [] ### Used as a library
        self.history = [] ### history of the stack

    def set_lexer_class(self, lexerclass):
        """Changes the lexer class for this virtual machine"""
        self._lexerclass = lexerclass

    def __repr__(self):
        return "<VM '%s' version %0.1f>" % (self.name, self.version)

    def add_rule(self, matches, func, caller=None):
        """add_rule(matches, func, [string])
        The first item is a list of string objects which can be used to call the method,
        which is determined by the first string. The optional second string is passed
        to the method when called.
        """
        LOG.debug("Adding rules for: %s", matches)
        ### Make sure matches are valid
        if not isinstance(matches, (list, tuple)):
            raise RuleNameError("Matches must be list or tuple")

        if not all([isinstance(x, str) for x in matches]):
            raise RuleNameError("Non-String in matches list")

        for match in matches:
            if match.upper() in self.reservedwords:
                raise RuleNameError("Reserved Word %s reused" % (match))
            if match.split()[0] != match:
                raise RuleNameError("Match %s cannot contain spaces" % (match))

        matches = [s.upper() for s in matches]

        ### Make sure func is valid
        if not isinstance(func, str):
            raise FunctionNameError("Cannot convert function name to string")
        func = func.strip()
        if not func:
            raise FunctionNameError("Function must contain alphanumeric characters")
        if func.split()[0] != func:
            raise FunctionNameError("Function cannot contain white space")

        ### Make sure that caller is valid (if there)
        if caller and not isinstance(caller, collections.Callable):
            raise CallerError('Caller must be string or callable')
        ### Add the matches to
        self.reservedwords.extend(matches)
        self.rules.append((matches, func, caller))

    def set_register(self, **kw):
        """add arbitrary values to the register"""
        for key, val in kw.items():
            self.registers[key] = str(val)

    ### Basic functions
    def do_binary_op(self, op):
        """do_binary_op
        Basic function that pops two items from the stack, performs a binary
        operation, and pushes the result onto the stack.
        The op should be callable.
        The popped items are assumed to be integers
        """
        right = self.stack.pop()
        right = int(self.registers[right] if right in self.registers else right)

        left = self.stack.pop()
        left = int(self.registers[left] if left in self.registers else left)
        res = op(left, right)
        self.stack.push(res)

    def do_unary_op(self, op):
        """do_unary_op(op)
        Basic function that pops the top item, performs the operation,
        and pushes the result onto the stack.
        The op should be callable.
        The popped items are assumed to be integers.
        """
        right = self.stack.pop()
        res = int(self.registers[right] if right in self.registers else right)
        res = op(right)
        self.stack.push(res)

    def add_binary_rule(self, matches, op):
        """add_binary_rule(matches, op)
        Adds a binary rule to the virtual machine.
        Rule will pop the top two items of the stack, perform the operation
        on them, and push the result onto the stack.
        The operator must accept integer inputs.
        """
        if not isinstance(op, collections.Callable):
            raise ValueError("operation needs to be callable")
        self.add_rule(matches, 'do_binary_op', op)

    def add_unary_rule(self, matches, op):
        """add_unary_rule(matches, op)
        Adds a unary rule to the virtual machine. Rule will pop the top
        of the stack, perform the operation, and push the result onto the
        stack.
        This takes an operator that expects integers as parameters.
        """
        if not isinstance(op, collections.Callable):
            raise ValueError("operation must be callable")
        self.add_rule(matches, 'do_unary_op', op)

    def import_library(self, library):
        """Imports the rules and functions of a Library"""
        if not issubclass(library, VMLibrary):
            raise LibraryError("Cannot import %s" % library)

        lib = library()
        newmethods = lib.methods()

        for match, func, caller in lib.exports:
            try:
                handler = getattr(lib, func)
            except:
                raise LibraryError("Library missing %s method" % (func))

            if (func in self.imported_methods) and func not in newmethods:
                raise LibraryImportError("Language already has %s method" % (func))

            self.imported_methods[func] = handler
            self.add_rule(match, func, caller)

    ### Begin Compiler Portion ###
    @property
    def program(self):
        """Returns the compiled program"""
        return self._program


    def feed(self, text):
        """feed(text)
        load a program into the machine.
        replaces all references to labels with instruction numbers
        """
        registers = self.registers
        lexer = self._lexerclass(text, language=self)
        tokens = [token.lexeme for token in lexer]
        # tokens leaves the labels in place
        subroutines = {}

        idx = -1
        compiled = []
        for token in tokens:
            if token.endswith(':'):
                subroutines[token[:-1]] = idx + 1
            else:
                compiled.append(token)
                if token.endswith("'"):
                    registers[token[:-1]] = 0
                idx += 1

        labeld = []
        for token in compiled:
            if token in subroutines:
                labeld.append(str(subroutines[token]))
            else:
                labeld.append(token)

        self._program = tuple(labeld)
        self.reset()

    ### END Compiler Portion

    ### START Runner Portion

    def reset(self):
        """resets the program but leaves the registers alone"""
        self._continue = 1
        self._curline = 0
        self._cycles = 0
        self.stack.clear()
        self.history = []
        #self.registers.clear()

    def run(self, **registers):
        """run through the program"""
        if registers:
            self.set_register(**registers)

        while self._continue:
            self.step()

    def step(self):
        """Processes a single instruction"""
        if not self._continue:
            return None
        try:
            current_command = self._program[self._curline]
        except IndexError:
            raise RunTimeError("No instruction %d" % self._curline)

        LOG.debug("%s: %s, %s, %s",self._curline,
                  current_command, self.stack, self.registers)
        self.history.append((self._cycles, self._curline,
                                    current_command,
                                    ["%s" % item for item in self.stack],
                                    dict(self.registers)))
        self._cycles += 1

        if current_command:
            self._curline += 1

            handler = None
            passself = 0
            for match, func, caller in self.rules:
                if current_command.upper() in match:
                    try:
                        proc = getattr(self, func)
                    except AttributeError:
                        try:
                            proc = self.imported_methods[func]
                            passself = 1
                        except:
                            raise MissingMethod("Missing %s Method" % func)
                    handler = proc
                    break

            LOG.debug(" Handler: %s with pasself %s", handler, passself)

            if handler and passself:
                handler(caller, self)
            elif handler:
                handler(caller)

            elif current_command.endswith("'"):

                value = self.stack.pop()
                register = current_command[:-1]

                self.registers[register] = value

            elif current_command in self.registers:
                self.stack.push(self.registers[current_command])
            elif current_command.endswith(':'):
                pass
            elif current_command.isdigit():
                self.stack.push(int(current_command))
            else:
                self.stack.push(current_command)


    def terminate(self, caller):
        """terminates the program"""
        self._continue = 0

    def go_to_instruction(self, line):
        """sets the current instruction number"""
        self._curline = int(line)




class BaseMachine(VM):
    """Virtual Machine with several arithmetic operations already applied."""
    def __init__(self, name="base", version="1.0"):
        VM.__init__(self, name, version)
        import operator as OP
        self.add_binary_rule(['plus', '+'], OP.add)
        self.add_binary_rule(['minus', '-'], OP.sub)
        self.add_binary_rule(['mul', '*'], OP.mul)
        self.add_binary_rule(['div', '/'], OP.truediv)
        self.add_binary_rule(['floordiv', '//'], OP.floordiv)
        self.add_binary_rule(['and', '&'], OP.and_)
        self.add_binary_rule(['mod', '%'], OP.mod)
        self.add_binary_rule(['or', '|'], OP.or_)
        self.add_binary_rule(['pow', '^'], OP.pow)
        self.add_binary_rule(['xor',], OP.xor)

        self.add_binary_rule(['max',], max)
        self.add_binary_rule(['min',], min)

        #~ Comparisons
        self.add_binary_rule(['lessthan', 'lt', '<'], OP.lt)
        self.add_binary_rule(['lte', 'le', '<='], OP.le)
        self.add_binary_rule(['eq', '=='], OP.eq)
        self.add_binary_rule(['ne', '!='], OP.ne)
        self.add_binary_rule(['greaterthan', 'gt', '>'], OP.gt)
        self.add_binary_rule(['gte', '>='], OP.ge)

        #~ Unary ops
        self.add_unary_rule(['abs'], OP.abs)
        self.add_unary_rule(['neg', '~'], OP.neg)

        self.import_library(ControlOperationsLibrary)

if __name__ == '__main__':
    TL = BaseMachine('Test', '0.0')

    #TL.set_register(mylife=2)
    TL.feed(""" # very simple choice, attack or run if too weak

            mylife 5 < do_run if
            mylife 10 < do_ttyf if
                "fang and claw" attack' . # comment to ignore
                do_run: "run" attack' .
                do_ttyf: "ttyf" attack' .
                """)
    print(TL.program)
    TL.run(mylife=16)
    print(TL.registers)
    print(TL.stack)

