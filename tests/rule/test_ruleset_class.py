import pytest
from vital_sqi.rule import *
import os


class TestRuleSet(object):
    r1 = Rule("perfusion")
    r2 = Rule("entropy")
    r3 = Rule("skewness_1")
    source = os.path.abspath("tests/test_data/rule_dict_test.json")
    r1.load_def(source)
    r2.load_def(source)
    r3.load_def(source)
    r = {3: r1, 2: r2, 1: r3}
    s = RuleSet(r)

    # def test_on_init(self):
    #     assert isinstance(self.s, RuleSet)

    def test_on_set(self):
        """Test setting rules in RuleSet."""
        r1 = Rule("sqi1")
        r2 = Rule("sqi2")
        r3 = Rule("sqi3")
        ruleset = RuleSet({1: r1, 2: r2, 3: r3})

        # Test invalid rule type
        with pytest.raises(AttributeError, match="Rule set must be of dict type."):
            ruleset.rules = []

        # Test non-consecutive keys
        with pytest.raises(
            ValueError, match="Rules must be ordered consecutively starting from 1."
        ):
            ruleset.rules = {1: r1, 3: r2}

        # Test mixed key types
        with pytest.raises(
            ValueError, match="All rule keys must be convertible to integers."
        ):
            ruleset.rules = {"1": r1, "two": r2}

        # Test invalid rule object
        with pytest.raises(
            ValueError, match="All rules must be instances of the Rule class."
        ):
            ruleset.rules = {1: r1, 2: "NotARule"}

        # Test valid reassignment
        valid_rules = {1: r1, 2: r2, 3: r3}
        ruleset.rules = valid_rules
        assert ruleset.rules == valid_rules

    def test_on_export(self):
        assert isinstance(self.s.export_rules(), str)

    def test_on_execute(self):
        dat = pd.DataFrame(
            [[6, 100, 1]], columns=["perfusion", "entropy", "skewness_1"]
        )
        assert self.s.execute(dat) == "accept"
        dat = pd.DataFrame(
            [[10, 100, 0]], columns=["perfusion", "entropy", "skewness_1"]
        )
        assert self.s.execute(dat) == "reject"
        with pytest.raises(ValueError) as exc_info:
            self.s.execute(pd.DataFrame([]))
        assert exc_info.match("Expected a data frame of 1 row but got 0")
        with pytest.raises(ValueError) as exc_info:
            dat = [[6, 1, 1], [1, 100, 3], [0, 0, 0]]
            dat = pd.DataFrame(dat, columns=["perfusion", "entropy", "skewness_1"])
            self.s.execute(dat)
        assert exc_info.match("Expected a data frame of 1 row but got 3")
        with pytest.raises(KeyError) as exc_info:
            dat = pd.DataFrame([[6, 100, 1]], columns=["perfusion", "entropy", "sqi4"])
            self.s.execute(dat)
        assert exc_info.match("not found in input data frame")

    @pytest.fixture
    def sample_rules(self):
        """Fixture to create sample Rule objects for testing."""
        r1 = Rule("sqi1")
        r2 = Rule("sqi2")
        r3 = Rule("sqi3")

        # Define rules for the tests
        r1.update_def(op_list=["<="], value_list=[5], label_list=["accept"])
        r2.update_def(op_list=[">"], value_list=[3], label_list=["reject"])
        r3.update_def(
            op_list=["<", ">="], value_list=[2, 10], label_list=["accept", "reject"]
        )
        return {1: r1, 2: r2, 3: r3}

    def test_on_init(self, sample_rules):
        """Test RuleSet initialization."""
        rule_set = RuleSet(sample_rules)
        assert isinstance(rule_set, RuleSet)
        assert len(rule_set.rules) == 3

    # def test_invalid_rule_set(self):
    #     """Test invalid RuleSet initialization."""
    #     with pytest.raises(AttributeError, match="Rule set must be of dict type."):
    #         RuleSet("not_a_dict")

    #     invalid_rules = {1: "NotARuleObject"}
    #     with pytest.raises(ValueError, match="All rules must be instances of the Rule class."):
    #         RuleSet(invalid_rules)

    #     unordered_rules = {2: Rule("sqi1"), 1: Rule("sqi2")}
    #     with pytest.raises(ValueError, match="Rules must be ordered consecutively starting from 1."):
    #         RuleSet(unordered_rules)

    # def test_export_rules(self, sample_rules):
    #     """Test exporting rules as a flowchart."""
    #     rule_set = RuleSet(sample_rules)
    #     flowchart_str = rule_set.export_rules()
    #     assert isinstance(flowchart_str, str)
    #     assert "Start" in flowchart_str
    #     assert "End" in flowchart_str

    def test_execute_valid(self, sample_rules):
        """Test executing rules with valid input."""
        rule_set = RuleSet(sample_rules)
        value_df = pd.DataFrame([[4, 3, 1]], columns=["sqi1", "sqi2", "sqi3"])
        decision = rule_set.execute(value_df)
        assert decision == "reject"

    # def test_execute_accept(self, sample_rules):
    #     """Test executing rules leading to acceptance."""
    #     rule_set = RuleSet(sample_rules)

    #     # Test with a valid DataFrame that passes all rules
    #     value_df = pd.DataFrame([[4, 3, 1]], columns=["sqi1", "sqi2", "sqi3"])
    #     decision = rule_set.execute(value_df)
    #     assert decision == "accept"

    # def test_execute_invalid_input(self, sample_rules):
    #     """Test invalid inputs for rule execution."""
    #     rule_set = RuleSet(sample_rules)

    #     with pytest.raises(TypeError, match="Expected data frame"):
    #         rule_set.execute("not_a_dataframe")

    #     invalid_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["sqi1", "sqi2", "sqi3"])
    #     with pytest.raises(ValueError, match="Expected a data frame of 1 row"):
    #         rule_set.execute(invalid_df)

    #     invalid_df = pd.DataFrame([[1, 2]], columns=["sqi1", "sqi2"])
    #     with pytest.raises(KeyError, match="SQI sqi3 not found in input data frame"):
    #         rule_set.execute(invalid_df)

    def test_empty_rule_set(self):
        """Test a RuleSet with no rules."""
        empty_rules = {}
        rule_set = RuleSet(empty_rules)
        value_df = pd.DataFrame([[1, 2, 3]], columns=["sqi1", "sqi2", "sqi3"])
        decision = rule_set.execute(value_df)
        assert decision == "accept"  # No rules to reject the input

    # def test_execute_with_edge_cases(self, sample_rules):
    #     """Test execution with edge-case values."""
    #     rule_set = RuleSet(sample_rules)

    #     value_df = pd.DataFrame([[5, 4, 10]], columns=["sqi1", "sqi2", "sqi3"])
    #     decision = rule_set.execute(value_df)
    #     assert decision == "reject"

    #     value_df = pd.DataFrame([[4.9, 3.9, 9.9]], columns=["sqi1", "sqi2", "sqi3"])
    #     decision = rule_set.execute(value_df)
    #     assert decision == "accept"

    # def test_export_rules_empty(self):
    #     """Test exporting rules when RuleSet is empty."""
    #     empty_rule_set = RuleSet({})
    #     with pytest.raises(ValueError, match="Cannot export flowchart for an empty RuleSet."):
    #         empty_rule_set.export_rules()
