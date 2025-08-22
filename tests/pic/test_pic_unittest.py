import random
import unittest

import numpy as np
import torch

from cirkit.backend.torch.parameters.pic import pc2qpc
from cirkit.pipeline import PipelineContext
from cirkit.templates import data_modalities, utils
from cirkit.templates.region_graph.algorithms.random import RandomBinaryTree

ctx = PipelineContext(backend="torch", semiring="lse-sum", fold=True, optimize=False)


def check_ll_shape(circuit, data):
    ll = circuit(data)
    assert ll.shape == (
        len(data),
        1,
        1,
    ), f"Expected log-likelihood shape {(len(data), 1, 1)}, got {ll.shape}"


class TestPic(unittest.TestCase):

    def test_image_data(self):
        symbolic_circuit = data_modalities.image_data(
            (1, 28, 28),
            region_graph="quad-graph",
            input_layer="categorical",
            num_input_units=64,
            sum_product_layer="cp",
            num_sum_units=64,
            sum_weight_param=utils.Parameterization(activation="none", initialization="normal"),
        )

        circuit = ctx.compile(symbolic_circuit)
        data = torch.randint(0, 256, (10, 1, 28, 28)).flatten(1)
        pc2qpc(circuit, integration_method="trapezoidal", net_dim=256)
        check_ll_shape(circuit, data)

        saved_model = circuit.state_dict()
        circuit.load_state_dict(saved_model)

    def test_continuous_data(self):

        n = 20
        features = 5
        data = torch.randn(n, features).float()
        rg = RandomBinaryTree(features)
        sum_weight_factory = utils.parameterization_to_factory(
            utils.Parameterization(activation="none", initialization="normal")
        )
        symbolic_circuit = rg.build_circuit(
            input_factory=utils.name_to_input_layer_factory("gaussian"),
            sum_product="cp",
            sum_weight_factory=sum_weight_factory,
            nary_sum_weight_factory=sum_weight_factory,
            num_input_units=10,
            num_sum_units=10,
        )
        circuit = ctx.compile(symbolic_circuit)

        pc2qpc(circuit, integration_method="trapezoidal", net_dim=4)
        check_ll_shape(circuit, data)

        saved_model = circuit.state_dict()
        circuit.load_state_dict(saved_model)

    def test_discrete_data(self):

        n = 20
        n_classes = 7
        features = 5
        data = torch.randint(0, n_classes, (n, features))
        rg = RandomBinaryTree(features)
        sum_weight_factory = utils.parameterization_to_factory(
            utils.Parameterization(activation="none", initialization="normal")
        )
        symbolic_circuit = rg.build_circuit(
            input_factory=utils.name_to_input_layer_factory(
                "categorical", **{"num_categories": n_classes}
            ),
            sum_product="cp",
            sum_weight_factory=sum_weight_factory,
            nary_sum_weight_factory=sum_weight_factory,
            num_input_units=4,
            num_sum_units=4,
        )
        circuit = ctx.compile(symbolic_circuit)

        pc2qpc(circuit, integration_method="trapezoidal", net_dim=2)
        check_ll_shape(circuit, data)

        saved_model = circuit.state_dict()
        circuit.load_state_dict(saved_model)

    def test_tabular_data(self):

        n_cat_features = 5
        n_num_features = 3
        region_graph = "chow-liu-tree"
        n = 20
        n_classes = 5
        cat_data = torch.randint(0, n_classes, (n, n_cat_features))
        num_data = torch.randn(n, n_num_features)
        data = torch.cat([cat_data, num_data], dim=1)
        num_features = data.shape[1]

        input_layers = [
            {"name": "categorical", "args": {"num_categories": n_classes + i}}
            for i in range(n_cat_features)
        ] + [{"name": "gaussian", "args": {}} for _ in range(n_num_features)]

        symbolic_circuit = data_modalities.tabular_data(
            region_graph=region_graph,
            data=data,
            input_layers=input_layers,
            num_input_units=10,
            sum_product_layer="cp",
            num_sum_units=10,
            sum_weight_param=utils.Parameterization(activation="none", initialization="normal"),
            use_mixing_weights=True,
        )

        # Check if the log-likelihood has the expected shape
        circuit = ctx.compile(symbolic_circuit)

        pc2qpc(circuit, integration_method="trapezoidal", net_dim=4)

        check_ll_shape(circuit, data)


if __name__ == "__main__":
    unittest.main()
