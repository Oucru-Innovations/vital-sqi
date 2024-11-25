"""
RuleSet Class for managing a set of SQI rules and building a decision flowchart.
"""

from vital_sqi.rule.rule_class import Rule
from pyflowchart import StartNode, EndNode, OperationNode, ConditionNode, Flowchart
import pandas as pd


class RuleSet:
    """
    A class to manage a set of rules for Signal Quality Indicators (SQI) and
    execute decision flow based on the provided rules.

    Attributes
    ----------
    rules : dict
        A dictionary where keys are rule order (int) and values are Rule instances.

    Methods
    -------
    export_rules():
        Exports the rules as a flowchart.
    execute(value_df):
        Executes the rules on a single-row DataFrame and returns a decision.
    """

    def __init__(self, rules):
        """
        Initializes RuleSet with a dictionary of rules.

        Parameters
        ----------
        rules : dict
            A dictionary of rules where the key is the rule's order (int)
            and the value is an instance of the Rule class.
        """
        self.rules = rules

    def __setattr__(self, name, value):
        if name == "rules":
            if not isinstance(value, dict):
                raise AttributeError("Rule set must be of dict type.")
            # Convert keys to integers to ensure consistent ordering
            try:
                value = {int(k): v for k, v in value.items()}
            except ValueError:
                raise ValueError("All rule keys must be convertible to integers.")
            # Validate rule order and types
            order = sorted(value.keys())
            if order != list(range(1, len(order) + 1)):
                raise ValueError("Rules must be ordered consecutively starting from 1.")
            for rule in value.values():
                if not isinstance(rule, Rule):
                    raise ValueError("All rules must be instances of the Rule class.")

        super().__setattr__(name, value)

    def export_rules(self):
        """
        Generates a flowchart representing the rule execution order.

        Returns
        -------
        str
            The generated flowchart in string format.
        """
        st = StartNode("")
        e = EndNode("")
        operations = []
        conditions = []

        for value in self.rules.values():
            operations.append(OperationNode(value.name))
            conditions.append(ConditionNode(value.write_rule()))

        # Connect nodes to create the flowchart
        st.connect(operations[0])
        for i, op in enumerate(operations):
            op.connect(conditions[i])
            conditions[i].connect_no(e)
            if i < len(operations) - 1:
                conditions[i].connect_yes(operations[i + 1])
        conditions[-1].connect_yes(e)

        fc = Flowchart(st)
        return fc.flowchart()

    def execute(self, value_df):
        """
        Executes the rule set on a given DataFrame and returns a decision.

        Parameters
        ----------
        value_df : pd.DataFrame
            A DataFrame containing one row with values for each rule.

        Returns
        -------
        str
            The decision ("accept" or "reject") based on the rule set.

        Raises
        ------
        KeyError
            If a rule's SQI is not found in the input DataFrame.
        """
        if not isinstance(value_df, pd.DataFrame):
            raise TypeError(f"Expected data frame, found {type(value_df)}")
        if len(value_df) != 1:
            raise ValueError(f"Expected a data frame of 1 row but got {len(value_df)}")

        for order, rule in sorted(self.rules.items()):
            try:
                value = value_df.iloc[0][rule.name]
            except KeyError:
                raise KeyError(f"SQI {rule.name} not found in input data frame")

            decision = rule.apply_rule(value)
            if decision == "reject":
                return "reject"
        return "accept"


# Example usage:
# r1 = Rule("sqi1")
# r2 = Rule("sqi2")
# r3 = Rule("sqi3")
# source = "/path/to/rule_dict_test.json"
# r1.load_def(source)
# r2.load_def(source)
# r3.load_def(source)
# rules = {3: r1, 2: r2, 1: r3}
# rule_set = RuleSet(rules)
# print(rule_set.export_rules())
# dat = pd.DataFrame([[6, 100, 0]], columns=['sqi1', 'sqi2', 'sqi3'])
# print(rule_set.execute(dat))
