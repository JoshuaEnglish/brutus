import unittest
import stackvm


class VMTest(unittest.TestCase):

    def test(self):
        vm = stackvm.machine.VM('Simpleton', 0)
        vm.feed('1 2 plus end')
        vm.run()
        self.assertListEqual(vm.stack.items, [1, 2, 'plus'])

    def test_store(self):
        vm = stackvm.machine.VM('Simpleton', 0)
        vm.feed("1 a' end")
        vm.run()
        self.assertEqual(vm.registers['A'], 1)

    def test_recall(self):
        vm = stackvm.machine.VM('Simpleton', 0)
        vm.feed('a end')
        vm.run(a=1)
        self.assertEqual(vm.registers['A'], '1')
        self.assertListEqual(vm.stack.items, ['1',])

class BaseLanguageTest(unittest.TestCase):
    def test_addition(self):
        bm = stackvm.machine.BaseMachine('simpleton', 0)
        bm.feed('1 2 plus end')
        bm.run()
        self.assertListEqual(bm.stack.items, [3,])

    def test_subtraction(self):
        bm = stackvm.machine.BaseMachine('simpleton', 0)
        bm.feed('3 2 minus end')
        bm.run()
        self.assertListEqual(bm.stack.items, [1,])

if __name__ == '__main__':
    unittest.main()