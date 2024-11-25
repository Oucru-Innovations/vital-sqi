import pytest
from vital_sqi.rule.rule_class import Rule
import os
import tempfile


class TestRuleClass(object):

    def test_on_init(self):
        assert isinstance(Rule("test_sqi"), Rule) is True

    def test_on_set(self):
        out = Rule("test_sqi")
        with pytest.raises(ValueError) as exc_info:
            out.name = "a/b"
        assert (
            str(exc_info.value)
            == "Name must contain only letters, numbers, hyphens, or underscores."
        )
        with pytest.raises(ValueError) as exc_info:
            out.rule = []
        assert str(exc_info.value) == "Rule definition must be a dictionary or None."

    def test_on_update(self):
        out = Rule("test_sqi")

        # Test with non-conflicting rules (consistent labels)
        out.update_def(
            op_list=["<=", ">"], value_list=[5, 5], label_list=["reject", "reject"]
        )
        assert out.rule["labels"][0] == "reject"

        # Test non-conflicting update with multiple thresholds
        out.update_def(
            op_list=["<=", ">", "<", ">="],
            value_list=[3, 3, 10, 10],
            label_list=["reject", "accept", "accept", "reject"],
        )
        assert out.rule["labels"][0] == "reject"

        # Test update with a mix of None labels
        out.update_def(
            op_list=["<=", ">"],
            value_list=[7, 7],
            label_list=["accept", None],
        )
        assert out.rule["labels"][-1] is None

        # Test conflicting labels with the same threshold
        # with pytest.raises(ValueError, match="inconsistent labels"):
        #     out.update_def(
        #         op_list=["<=", ">="],
        #         value_list=[3, 3],
        #         label_list=["reject", "accept"],
        #     )

    def test_on_load(self):
        out = Rule("perfusion")
        source = os.path.abspath("tests/test_data/rule_dict_test.json")
        out.load_def(source)
        assert isinstance(out.rule["def"], list) is True
        with pytest.raises(Exception) as exc_info:
            out.name = "random_sqi"
            out.load_def(source)
        assert exc_info.match("not found")
        with pytest.raises(Exception) as exc_info:
            source = os.path.abspath("tests/test_data/file_not_exist.json")
            out.load_def(source)
        assert exc_info.match("Source file not found")

    def test_on_apply_rule(self):
        out = Rule("test_sqi")
        # Define non-conflicting rules
        out.update_def(
            op_list=["<=", "<", ">=", ">"],
            value_list=[3, 4, 10, 11],
            label_list=["reject", "accept", "reject", "accept"],
        )

        # Test applying the rule
        assert out.apply_rule(2) == "reject"  # Below the lowest boundary
        assert out.apply_rule(3.5) == "accept"  # Inside the range
        assert out.apply_rule(10) == "reject"  # On the boundary
        assert out.apply_rule(12) == "accept"  # Above the highest boundary

    def test_on_save(self):
        rule_obj = Rule("perfusion")
        source = os.path.abspath("tests/test_data/rule_dict_test.json")
        rule_obj.load_def(source)
        file_out = tempfile.gettempdir() + "/rule_dict.json"
        rule_obj.save_def(file_out)
        assert os.path.isfile(file_out)

        # Update with valid rules
        rule_obj.update_def(
            op_list=["<=", "<", ">=", ">"],
            value_list=[3, 4, 10, 11],
            label_list=["reject", "accept", "reject", "accept"],
        )
        rule_obj.save_def(file_out, overwrite=True)
        assert os.path.isfile(file_out)

    def test_on_update_invalid_labels(self):
        out = Rule("test_sqi")
        with pytest.raises(
            ValueError, match="Labels must be 'accept', 'reject', or None"
        ):
            out.update_def(
                op_list=["<=", ">", "<", ">="],
                value_list=[3, 3, 10, 10],
                label_list=["approve", "accept", "accept", "reject"],
            )

    def test_on_save_invalid_path(self):
        rule_obj = Rule("test_sqi")
        with pytest.raises(ValueError, match="Invalid output file path"):
            rule_obj.save_def("", overwrite=True)

    # def test_on_update_conflicting_boundaries(self):
    #     out = Rule("test_sqi")
    #     with pytest.raises(ValueError):
    #         out.update_def(
    #             op_list=["<=", ">="],
    #             value_list=[5, 5],
    #             label_list=["accept", "reject"],
    #         )

    def test_on_apply_rule_no_boundaries(self):
        out = Rule("test_sqi")
        out.rule = {"def": [], "boundaries": [], "labels": []}
        assert out.apply_rule(10) is None

    def test_on_apply_rule_outside_boundaries(self):
        out = Rule("test_sqi")
        out.update_def(
            op_list=["<=", ">"], value_list=[5, 10], label_list=["accept", "reject"]
        )
        assert out.apply_rule(11) == "reject"
        assert out.apply_rule(0) == "accept"

    def test_on_empty_rule_definition(self):
        out = Rule("test_sqi")
        assert out.rule is None
        with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
            out.apply_rule(5)

    def test_on_write_empty_rule_definition(self):
        out = Rule("test_sqi")
        assert out.write_rule() == ""

    def test_on_save_nonexistent_file_for_overwrite(self):
        rule_obj = Rule("test_sqi")
        with pytest.raises(FileNotFoundError, match="File to overwrite does not exist"):
            rule_obj.save_def("nonexistent_path.json", overwrite=True)
