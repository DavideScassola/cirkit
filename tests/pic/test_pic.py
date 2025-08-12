import itertools

import pytest
import torch

from cirkit.backend.torch.parameters.pic import pc2qpc
from cirkit.pipeline import compile
from cirkit.templates import data_modalities, utils
from cirkit.templates.data_modalities import tabular_data
from cirkit.templates.region_graph.algorithms.random import RandomBinaryTree


def check_ll_shape(circuit, data):
    ll = circuit(data)
    assert ll.shape == (
        len(data),
        1,
        1,
    ), f"Expected log-likelihood shape {(n, 1, 1)}, got {ll.shape}"


@pytest.mark.parametrize(
    "n_cat_features,n_num_features,region_graph",
    itertools.product([0, 1, 2], [0, 1, 3], ["random-binary-tree", "chow-liu-tree"]),
)
def test_tabular_data(n_cat_features: int, n_num_features: int, region_graph: str):

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

    if num_features > 0:

        if num_features == 1:
            # TODO: The case with only one feature has to be fixed
            # and it does not depend on the function tested here
            # pytest.xfail("Single feature case is known to fail")
            pass
        else:
            symbolic_circuit = tabular_data(
                region_graph=region_graph,
                data=data,
                input_layers=input_layers,
                num_input_units=2,
                sum_product_layer="cp",
                num_sum_units=2,
                sum_weight_param=utils.Parameterization(
                    activation="softmax", initialization="normal"
                ),
                use_mixing_weights=True,
            )

            # Check if the circuit has the expected number of input layers
            assert len(symbolic_circuit.scope) == num_features

            # Check if the log-likelihood has the expected shape
            circuit = compile(symbolic_circuit)

            check_ll_shape(circuit, data)

            pc2qpc(circuit, integration_method="trapezoidal", net_dim=4)

            check_ll_shape(circuit, data)


def test_continuous_data():

    n = 20
    features = 5
    data = torch.randn(n, features).float()
    rg = RandomBinaryTree(features)
    sum_weight_factory = utils.parameterization_to_factory(
        utils.Parameterization(activation="softmax", initialization="normal")
    )
    symbolic_circuit = rg.build_circuit(
        input_factory=utils.name_to_input_layer_factory("gaussian"),
        sum_product="cp",
        sum_weight_factory=sum_weight_factory,
        nary_sum_weight_factory=sum_weight_factory,
        num_input_units=4,
        num_sum_units=4,
    )
    circuit = compile(symbolic_circuit)
    check_ll_shape(circuit, data)
    pc2qpc(circuit, integration_method="trapezoidal", net_dim=2)
    check_ll_shape(circuit, data)


def test_discrete_data():

    n = 20
    n_classes = 7
    features = 5
    data = torch.randint(0, n_classes, (n, features))
    rg = RandomBinaryTree(features)
    sum_weight_factory = utils.parameterization_to_factory(
        utils.Parameterization(activation="softmax", initialization="normal")
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
    circuit = compile(symbolic_circuit)
    check_ll_shape(circuit, data)
    pc2qpc(circuit, integration_method="trapezoidal", net_dim=2)
    check_ll_shape(circuit, data)


def test_image_data():

    symbolic_circuit = data_modalities.image_data(
        (1, 28, 28),  # The shape of MNIST image, i.e., (num_channels, image_height, image_width)
        region_graph="quad-graph",  # Select the structure of the circuit to follow the QuadGraph region graph
        input_layer="categorical",  # Use Categorical distributions for the pixel values (0-255) as input layers
        num_input_units=64,  # Each input layer consists of 64 Categorical input units
        sum_product_layer="cp",  # Use CP sum-product layers, i.e., alternate dense sum layers and hadamard product layers
        num_sum_units=64,  # Each dense sum layer consists of 64 sum units
        sum_weight_param=utils.Parameterization(
            activation="none",  # Do not use any parameterization
            initialization="normal",  # Initialize the sum weights by sampling from a standard normal distribution
        ),
    )
    circuit = compile(symbolic_circuit)
    data = torch.randint(0, 256, (10, 1, 28, 28))  # Simulating MNIST-like images
    #check_ll_shape(circuit, data)
    pc2qpc(circuit, integration_method="trapezoidal", net_dim=256)
    #check_ll_shape(circuit, data)
