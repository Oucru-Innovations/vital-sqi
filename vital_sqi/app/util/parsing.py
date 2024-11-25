import base64
import pandas as pd
import io
import json
import dash_html_components as html
import pathlib
from vital_sqi.rule.rule_class import Rule
from vital_sqi.rule.ruleset_class import RuleSet
from vital_sqi.common.utils import update_rule

# Define paths
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../temp").resolve()


def parse_data(contents, filename):
    """
    Parse uploaded file content into a DataFrame or JSON object.
    Supports CSV, Excel, TXT, and JSON files.
    """
    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        if filename.endswith(".csv"):
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")), header=0).to_dict()
        elif filename.endswith(".xls") or filename.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(decoded)).to_dict()
        elif filename.endswith(".txt"):
            return pd.read_csv(io.StringIO(decoded.decode("utf-8")), delimiter=r"\s+").to_dict()
        elif filename.endswith(".json"):
            return json.load(io.StringIO(decoded.decode("utf-8")))
        else:
            raise ValueError(f"Unsupported file type: {filename}")

    except Exception as e:
        print(f"Error parsing file: {e}")
        return html.Div(["Error processing file. Please check the format."])


def generate_rule(rule_name, rule_def):
    """
    Generate a Rule object from a rule definition.
    """
    try:
        rule_def, boundaries, label_list = update_rule(rule_def, is_update=False)
        rule_detail = {"def": rule_def, "boundaries": boundaries, "labels": label_list}
        return Rule(rule_name, rule_detail)
    except Exception as e:
        print(f"Error generating rule: {e}")
        return None


def generate_rule_set(rule_set_dict):
    """
    Generate a RuleSet object from a dictionary of rules.
    """
    try:
        rule_set = {}
        for rule_dict in rule_set_dict:
            rule_name = rule_dict["name"]
            rule_def = rule_dict["def"]
            rule = generate_rule(rule_name, rule_def)
            if rule:
                rule_set[rule_dict["order"]] = rule
        return RuleSet(rule_set)
    except Exception as e:
        print(f"Error generating rule set: {e}")
        return None


def parse_rule_list(rule_def):
    """
    Parse a rule definition into a list of dictionaries.
    """
    try:
        return [{"op": r["op"], "value": r["value"], "label": r["label"]} for r in rule_def]
    except Exception as e:
        print(f"Error parsing rule list: {e}")
        return []


def generate_boundaries(boundaries):
    """
    Generate boundary ranges for visualization.
    """
    try:
        bound_list = ["[-inf, " + str(boundaries[0]) + "]"]
        for i in range(1, len(boundaries)):
            bound_list.append(f"[{boundaries[i-1]}, {boundaries[i]}]")
        bound_list.append(f"[{boundaries[-1]}, inf]")
        return bound_list
    except Exception as e:
        print(f"Error generating boundaries: {e}")
        return []
