from collections import Counter
import math

"""All calculations are based on vectors, where the last item is the target class"""


class Node:

    def __init__(self):
        self._attribute = None

        # Mapping a value of the attribute to another node or to the target class
        self._paths = {}

    def build_children(self, data_set):
        best_attribute, best_information_gain = None, -1

        for i in range(len(data_set[0]) - 1):
            information_gain = calculate_information_gain(data_set, i)

            if information_gain > best_information_gain:
                best_attribute = i
                best_information_gain = information_gain

        self._attribute = best_attribute

        attribute_values = set(data_point[best_attribute] for data_point in data_set)

        for attribute_value in attribute_values:
            children_subset = [i[0:best_attribute] + i[best_attribute:] for i in data_set if i[best_attribute] == attribute_value]
            target_classes = _get_attribute_count(children_subset, len(children_subset[0]) - 1)

            if len(target_classes) == 1:
                # No need to split any further as all instances in the set
                # have the same target class
                self._paths[attribute_value] = list(target_classes.keys())[0]
            elif len(data_set[0]) == 2:
                # we ran out of attributes to split on, so just picking
                # a class with the most occurrences
                self._paths[attribute_value] = max(target_classes, key=target_classes.get)
            else:
                # We can and should build the decision tree further
                child_node = Node()
                child_node.build_children(children_subset)

                self._paths[attribute_value] = child_node

    def make_decision(self, input):
        if isinstance(self._paths[input[self._attribute]], Node):
            return self._paths[input[self._attribute]].make_decision(input)
        else:
            return self._paths[input[self._attribute]]


def calculate_entropy(data_set):
    """Calculates the entropy of the data set"""

    attribute_counts = _get_attribute_count(data_set, len(data_set[0]) - 1)

    return sum(-attribute_count / len(data_set) * math.log(attribute_count / len(data_set), 2) for _, attribute_count in attribute_counts.items())

def calculate_information_gain(data_set, index):
    """Calculating the information gain of attribute given by `property_index`"""

    total_entropy = calculate_entropy(data_set)
    attribute_counts = _get_attribute_count(data_set, index)

    attribute_entropy = {}

    for attribute_value in attribute_counts:
        subset_of_attribute = list(filter(lambda x: x[index] == attribute_value, data_set))
        attribute_entropy[attribute_value] = calculate_entropy(subset_of_attribute)

    return total_entropy - sum(attribute_count / len(data_set) * attribute_entropy[attribute] for attribute, attribute_count in attribute_counts.items())

def _get_attribute_count(data_set, index):
    attribute_counts = {}

    for data_point in data_set:
        attribute_counts[data_point[index]] = attribute_counts.get(data_point[index], 0) + 1

    return attribute_counts