import ast
from dataclasses import dataclass
from typing import List, Optional

from onai.ml.peers.dp import get_retriable_session
from onai.ml.peers.dp_requests import HEADERS


@dataclass
class ComputationNode:
    mnemonic: str
    default_val: Optional[float]
    dependants: List
    concrete_val: Optional[float]

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.mnemonic) + " " + str(id(self))
        if self.default_val is not None:
            ret += " (O)"
        ret += "\n"
        for child in self.dependants:
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return "<Computational node representation>"


COA_QUERY = """
query($templateId: UUID) {
    chartOfAccountsTemplateV2(templateId: $templateId) {
      templateId
      name
      groups {
        members {
          dataItem {
            mnemonic
            formula {
              expression
            }
          }
        }
      }
  }
}
"""


class _InputFinder(ast.NodeVisitor):
    def __init__(self):
        self._call_stack = []
        self._inputs = []

    def visit_Call(self, node):
        func_name = node.func.id

        if func_name == "vals":
            # There must be a single node in node.args of type ast.Str
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Str)
            self._inputs.append((node.args[0].s, "default" in self._call_stack))

        self._call_stack.append(func_name)

        for arg_node in node.args:
            self.visit(arg_node)

        self._call_stack.pop()


def _find_inputs(expr: str):
    input_finder = _InputFinder()
    input_finder.visit(ast.parse(expr))
    return input_finder._inputs


class ComputationalTree:
    def __init__(self, root_node="EBITDA"):
        self.root = ComputationNode("EBITDA", None, [], None)
        self.formulae_by_mnemonic = self._get_all_coa_formula()
        assert root_node in self.formulae_by_mnemonic

        self._node_by_mnemonic = {}
        self.root = self._construct_node("EBITDA", None)
        self.loose_thres = 0.0

    def root_populatable(self):
        return self._node_populatable(self.root)

    def _node_populatable(self, node: ComputationNode):
        if node.concrete_val is not None:
            return True
        if len(node.dependants) == 0:
            return False
        available_dependants = sum(self._node_populatable(d) for d in node.dependants)
        if (available_dependants / len(node.dependants)) >= self.loose_thres:
            return True
        return False

    def __getitem__(self, item) -> ComputationNode:
        return self._node_by_mnemonic[item]

    def __contains__(self, item) -> bool:
        return item in self._node_by_mnemonic

    def reset_concrete_vals(self):
        self._reset_tree_concrete_vals(self.root)

    def _reset_tree_concrete_vals(self, node: ComputationNode):
        node.concrete_val = None
        for n in node.dependants:
            self._reset_tree_concrete_vals(n)

    def _construct_node(self, mnemonic, default_val: Optional[float]):
        if mnemonic in self._node_by_mnemonic:
            return self._node_by_mnemonic[mnemonic]
        formula = self.formulae_by_mnemonic.get(mnemonic, None)
        children = []
        if formula is not None:
            for dependant_mnemonic, optional_dependant in _find_inputs(
                formula["expression"]
            ):
                children.append(
                    self._construct_node(
                        dependant_mnemonic, 0.0 if optional_dependant else None
                    )
                )
        assert mnemonic not in self._node_by_mnemonic
        self._node_by_mnemonic[mnemonic] = ret = ComputationNode(
            mnemonic, default_val, children, None
        )
        return ret

    def _get_all_coa_formula(self, template_id="e29b84ce-fe16-51b2-9f3e-c12be25ca100"):

        with get_retriable_session() as s:
            groups = s.post(
                "https://data-services.onai.cloud/api/",
                json={"query": COA_QUERY, "variables": {"templateId": template_id}},
                headers=HEADERS,
            ).json()["data"]["chartOfAccountsTemplateV2"]["groups"]
        formulae_by_mnemonic = {}
        for g in groups:
            for m in g["members"]:
                di = m["dataItem"]

                formulae_by_mnemonic[di["mnemonic"]] = di["formula"]

        return formulae_by_mnemonic


def main():
    t = ComputationalTree()
    print(t.root)

    print("Before populating: %s" % t.root_populatable())

    t["EBITDA"].concrete_val = 0.0
    print("Direct populating: %s" % t.root_populatable())
    t.reset_concrete_vals()

    print("After reseting %s" % t.root_populatable())

    t["OPER_INC"].concrete_val = 0.0
    t["TOTAL_DEPR_AMORT"].concrete_val = 0.0
    print("Missing one %s" % t.root_populatable())

    t["COGS_DEPR_AMORT"].concrete_val = 0.0

    print("Everything is there %s" % t.root_populatable())
    t["OPER_INC"].concrete_val = None
    t["TOTAL_REVENUE"].concrete_val = 0.0
    t["TOTAL_OPER_EXP"].concrete_val = 0.0
    print("Second level populatable %s" % t.root_populatable())
    t["TOTAL_REVENUE"].concrete_val = None
    print("Second level missing %s" % t.root_populatable())
    t["NET_REVENUE"].concrete_val = 0.0
    t["REVENUE_SERVICE"].concrete_val = 0.0
    t["REVENUE_PRODUCT"].concrete_val = 0.0
    t["OWN_WORK_CAPITALISED"].concrete_val = 0.0
    print("Third level missing 1 %s" % t.root_populatable())
    t["TOTAL_OTHER_REVENUE"].concrete_val = 0.0

    print("Third level populatable  %s" % t.root_populatable())

    t.reset_concrete_vals()
    print("After reset %s " % t.root_populatable())


if __name__ == "__main__":
    main()
