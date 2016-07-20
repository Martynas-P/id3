import unittest

import id3


class EntropyCalculationTest(unittest.TestCase):

    def test_no_entropy(self):
        data_set = [
            ['A', 'X'],
            ['A', 'X'],
            ['A', 'X'],
        ]

        result = id3.calculate_entropy(data_set)

        self.assertEqual(result, 0)

    def test_maximum_entropy(self):
        data_set = [
            ['A', 'X'],
            ['B', 'X'],
            ['C', 'Y'],
            ['C', 'Y'],
        ]

        result = id3.calculate_entropy(data_set)

        self.assertEqual(result, 1)

    def test_entropy1(self):
        data_set = [
            ['B', 'X'],
            ['C', 'X'],
            ['D', 'Y'],
            ['E', 'Y'],
            ['F', 'Z'],
        ]

        result = id3.calculate_entropy(data_set)

        self.assertAlmostEqual(result, 1.521928094887)

    def test_entropy2(self):
        data_set = [
            ['A', 'B'],
            ['C', 'B'],
            ['E', 'D'],
            ['G', 'D'],
            ['I', 'E'],
            ['K', 'E'],
        ]

        result = id3.calculate_entropy(data_set)

        self.assertAlmostEqual(result, 1.584962500721)


class InformationGainTest(unittest.TestCase):

    def test_attribute_extremes(self):
        data_set = [
            ['A', 'B', 'X'],
            ['A', 'B', 'X'],
            ['A', 'C', 'Y'],
            ['A', 'C', 'Y'],
        ]

        # 1st attribute is the best one
        self.assertEqual(id3.calculate_information_gain(data_set, 1), 1)
        # 0th attribute is the worst
        self.assertEqual(id3.calculate_information_gain(data_set, 0), 0)

    def test_information_gain1(self):
        data_set = [
            ['A', 'B', 'E', 'X'],
            ['A', 'B', 'F', 'X'],
            ['A', 'B', 'G', 'Y'],
            ['A', 'C', 'H', 'Y'],
            ['A', 'C', 'I', 'Y'],
        ]

        self.assertEqual(id3.calculate_information_gain(data_set, 0), 0)
        self.assertAlmostEqual(id3.calculate_information_gain(data_set, 1), 0.419973094021)
        self.assertAlmostEqual(id3.calculate_information_gain(data_set, 2), 0.970950594455)


class BuildTreeTest(unittest.TestCase):

    def test_build_tree0(self):
        data_set = [
            ['A', 'X'],
            ['A', 'X'],
            ['A', 'X'],
        ]

        node = id3.Node()
        node.build_children(data_set)

        self.assertEqual(node._attribute, 0)
        self.assertEqual(node._paths['A'], 'X')
        self.assertEqual(node.make_decision(['A',]), 'X')

    def test_build_tree1(self):
        data_set = [
            ['A', 'X'],
            ['A', 'Y'],
            ['A', 'Y'],
        ]

        node = id3.Node()
        node.build_children(data_set)

        self.assertEqual(node._attribute, 0)
        self.assertEqual(node._paths['A'], 'Y')
        self.assertEqual(node.make_decision(['A', ]), 'Y')

    def test_builds_tree2(self):
        # Data set from http://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm
        data_set = [
            ['Sunny', 'Hot', 'High', 'Weak', 'No'],
            ['Sunny', 'Hot', 'High', 'Strong', 'No'],
            ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
            ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
            ['Sunny', 'Mild', 'High', 'Weak', 'No'],
            ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
            ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
            ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
            ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
            ['Rain', 'Mild', 'High', 'Strong', 'No'],
        ]

        node = id3.Node()
        node.build_children(data_set)

        # Did we pick the root attribute correctly?
        self.assertEqual(node._attribute, 0)

        # Did we pick the children correctly?
        self.assertIsInstance(node._paths['Sunny'], id3.Node)
        self.assertEqual(node._paths['Sunny']._attribute, 2)

        self.assertIsInstance(node._paths['Rain'], id3.Node)
        self.assertEqual(node._paths['Rain']._attribute, 3)

        self.assertEqual(node._paths['Overcast'], 'Yes')

        # Did we pick grandchildren correctly?
        self.assertEqual(node._paths['Sunny']._paths['High'], 'No')
        self.assertEqual(node._paths['Sunny']._paths['Normal'], 'Yes')

        self.assertEqual(node._paths['Rain']._paths['Strong'], 'No')
        self.assertEqual(node._paths['Rain']._paths['Weak'], 'Yes')

        # Do we get correct answers?
        self.assertEqual(node.make_decision(['Overcast', 'Hot', 'High', 'Strong']), 'Yes')
        self.assertEqual(node.make_decision(['Rain', 'Hot', 'High', 'Strong']), 'No')
        self.assertEqual(node.make_decision(['Rain', 'Hot', 'High', 'Weak']), 'Yes')
        self.assertEqual(node.make_decision(['Sunny', 'Hot', 'High', 'Weak']), 'No')
        self.assertEqual(node.make_decision(['Sunny', 'Hot', 'Normal', 'Weak']), 'Yes')
