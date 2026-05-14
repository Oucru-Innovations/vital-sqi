"""
Class Rule contains thresholds and corresponding labels of an SQI. Labels
are either 'accept' or 'reject'.
"""

import json
import re
import os
import numpy as np
import bisect
from vital_sqi.common.utils import parse_rule, update_rule


class Rule:
    """
    A class to represent and manage threshold-based rules for Signal Quality Indices (SQI).

    Attributes
    ----------
    name : str
        The name of the SQI rule.
    rule : dict or None
        The rule definition, containing thresholds, boundaries, and labels.

    Methods
    -------
    load_def(source=None):
        Loads rule definitions from a specified source.
    update_def(op_list, value_list, label_list):
        Updates rule definitions based on provided lists of operands, values, and labels.
    save_def(file_path, file_type="json", overwrite=False):
        Saves the current rule definition to a specified file path.
    apply_rule(x):
        Applies the rule to an input x, returning the appropriate label.
    write_rule():
        Returns a string representation of the rule for display purposes.
    """

    def __init__(self, name, rule=None):
        self.name = name
        self.rule = rule

    def __setattr__(self, name, value):
        if name == "name":
            if not isinstance(value, str) or not re.match(r"^[A-Za-z0-9_-]+$", value):
                raise ValueError(
                    "Name must contain only letters, numbers, hyphens, or underscores."
                )
        elif name == "rule" and value is not None and not isinstance(value, dict):
            raise ValueError("Rule definition must be a dictionary or None.")
        super().__setattr__(name, value)

    def load_def(self, source=None):
        """
        Loads rule definitions from a specified source.

        Parameters
        ----------
        source : str, optional
            The file path to load rule definitions from (default is None).
        """
        rule_def, boundaries, labels = parse_rule(self.name, source)
        self.rule = {"def": rule_def, "boundaries": boundaries, "labels": labels}

    def update_def(self, op_list, value_list, label_list):
        """
        Updates rule definitions with new thresholds, values, and labels.

        Parameters
        ----------
        op_list : list of str
            List of operators for the rule (e.g., ["<=", ">"]).
        value_list : list of float
            List of threshold values corresponding to each operator.
        label_list : list of str
            List of labels ("accept" or "reject") corresponding to each threshold.

        Raises
        ------
        ValueError
            If invalid operator, value, or label is provided.

        Returns
        -------

        Examples
        --------
        >>> rule = Rule("test_sqi")
        >>> rule.load_def("../resource/rule_dict.json")
        >>> rule.update_def(op_list=["<=", ">"],
                        value_list=[5, 5],
                        label_list=["accept", "reject"])
        >>> print(rule.rule['def'])
        [{'op': '>', 'value': '10', 'label': 'reject'},
        {'op': '>=', 'value': '3', 'label': 'accept'},
        {'op': '<', 'value': '3', 'label': 'reject'},
        {'op': '<=', 'value': 5, 'label': 'accept'},
        {'op': '>', 'value': 5, 'label': 'reject'}]
        """
        if any(op not in ["<", "<=", ">", ">=", "="] for op in op_list):
            raise ValueError("Operands must be one of '<', '<=', '>', '>=', '='.")
        if any(not np.isreal(value) for value in value_list):
            raise ValueError("Thresholds must be numeric.")
        if any(label not in ["accept", "reject", None] for label in label_list):
            raise ValueError("Labels must be 'accept', 'reject', or None.")

        threshold_list = []
        for idx in range(len(label_list)):
            threshold = {
                "op": op_list[idx],
                "value": value_list[idx],
                "label": label_list[idx],
            }
            threshold_list.append(threshold)

        if self.rule is None:
            self.rule = {"def": None, "boundaries": None, "labels": None}
        self.rule["def"], self.rule["boundaries"], self.rule["labels"] = update_rule(
            self.rule["def"], threshold_list
        )
        return

    def save_def(self, file_path, file_type="json", overwrite=False):
        """
        Saves the rule definition to a specified file.

        Parameters
        ----------
        file_path : str
            The path to save the rule definition.
        file_type : str, optional
            The format to save the file in (default is "json").
        overwrite : bool, optional
            If True, allows overwriting existing files (default is False).
        """
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("Invalid output file path.")

        if overwrite and not os.path.isfile(file_path):
            raise FileNotFoundError("File to overwrite does not exist.")

        # if overwrite:
        #     with open(file_path) as file_in:
        #         all_rules = json.load(file_in)
        #         if not isinstance(all_rules, dict):
        #             raise ValueError("Invalid file format.")
        #     all_rules[self.name] = {"name": self.name, "def": self.rule["def"]}
        # else:
        #     all_rules = {self.name: {"name": self.name, "def": self.rule["def"]}}

        # with open(file_path, "w") as file_out:
        #     json.dump(all_rules, file_out)

        if overwrite:
            with open(file_path) as file_in:
                all_rules = json.load(file_in)
                assert isinstance(all_rules, dict), "Invalid file format."
            if np.any(np.array(list(all_rules.keys())) == self.name):
                all_rules[self.name]["def"] = self.rule["def"]
            else:
                all_rules[self.name] = {"name": self.name, "def": self.rule["def"]}
            with open(file_path, "w") as file_out:
                json.dump(all_rules, file_out)
        else:
            with open(file_path, "w") as file_out:
                name = self.name
                rule = self.rule["def"]
                rule_def = {name: {"name": name, "def": rule}}
                json.dump(rule_def, file_out)
        return

    def apply_rule(self, x):
        """
        Apply the rule to an SQI value and return its quality label.

        The rule stores a sorted ``boundaries`` array and a parallel
        ``labels`` array built from the ``"def"`` entries.  Lookup is O(log n)
        via ``bisect.bisect_left``:

        - If *x* equals a boundary exactly, the label at the boundary position
          is returned (handles closed-interval endpoints).
        - Otherwise ``bisect_left`` locates the interval ``[boundaries[i-1],
          boundaries[i])`` containing *x* and returns ``labels[i*2]``.

        For the standard four-element calibrated rule encoding the open
        interval ``(lower, upper)``::

            boundaries = [lower, upper]
            labels     = ["reject", "accept", "reject", ...]

        so:  x <= lower → "reject", lower < x < upper → "accept",
             x >= upper → "reject".

        Parameters
        ----------
        x : float
            The SQI value to evaluate.

        Returns
        -------
        str or None
            ``"accept"``, ``"reject"``, or ``None`` if no label is defined
            for the interval containing *x*.
        """
        boundaries, labels = self.rule["boundaries"], self.rule["labels"]

        if not np.isfinite(x):
            return "reject"

        if x in boundaries:
            return labels[(np.where(boundaries == x)[0][0]) * 2 + 1]

        # Use bisect to locate the correct interval for the input value
        label_index = bisect.bisect_left(boundaries, x)
        return labels[label_index * 2] if label_index < len(labels) else "reject"

    def write_rule(self):
        """
        Returns a string representation of the rule.

        Returns
        -------
        str
            A string representation of the rule for display.
        """
        if not self.rule or not self.rule.get("def"):
            return ""
        return "\n".join(
            f"x {r['op']} {r['value']}: {r['label']}" for r in self.rule["def"]
        )


if __name__ == "__main__":
    out = Rule("test_sqi")

    # Test non-conflicting update with multiple thresholds
    out.update_def(
        op_list=["<=", ">", "<", ">="],
        value_list=[3, 3, 10, 10],
        label_list=["reject", "accept", "accept", "reject"],
    )
    # assert out.rule["labels"][0] == "reject"
    print(out.rule["labels"][0])
