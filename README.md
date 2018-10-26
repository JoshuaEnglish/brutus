# stack-vm
Simple stack based virtual machine

There are 4 components

* A LIFO stack
* The registry (a collections.MutableMapping object)
* The program (a tuple of strings)
* the counter (instruction index)

The basic VM class only supports a single rule: *end*. A BaseMachine class
provides methods that can build simple programs.

New rules can be added with any callable object.

Brutus also adds a lexical scanner that builds concrete syntax trees that
can be translated into programs for the Stack VM. Small languages can 
be built.


