import torch
import torch.nn as nn
import pytest

from vit_core.patch_embedding import ConvolutionalPatchEmbedding, ManualPatchEmbedding

@pytest.fixture(scope="module")
def params():
    """Test parameters"""
    return {
        "batch_size": 4,
        "in_channels": 3,
        "height": 32,
        "width": 32,
        "patch_size": 8,
        "embedding_dimension": 128,
    }


@pytest.fixture(scope="module")
def input_shape_tuple(params):
    """Provides image shape as a tuple (C, H, W)."""
    return (params["in_channels"], params["height"], params["width"])


@pytest.fixture(scope="module")
def conv_patch_embed_model(params, input_shape_tuple):
    """Provides an instance of the Convolutional model."""

    model = ConvolutionalPatchEmbedding(
        input_shape=input_shape_tuple,
        embedding_dimension=params["embedding_dimension"],
        patch_size=params["patch_size"],
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def manual_patch_embed_model(params, input_shape_tuple):
    """Provides an instance of the Manual model."""

    model = ManualPatchEmbedding(
        input_shape=input_shape_tuple,
        embedding_dimension=params["embedding_dimension"],
        patch_size=params["patch_size"],
    )
    model.eval()
    return model


@pytest.fixture
def sample_image_batch(params):
    """Provides a sample batch of image tensors."""
    return torch.rand(
        params["batch_size"], params["in_channels"], params["height"], params["width"]
    )


def calculate_num_patches(params):
    num_patches_h = params["height"] // params["patch_size"]
    num_patches_w = params["width"] // params["patch_size"]
    return num_patches_h * num_patches_w


def test_conv_patch_embed_forward_shape(
    conv_patch_embed_model, sample_image_batch, params
):
    """Tests the output shape of the forward pass for Convolutional."""
    output = conv_patch_embed_model(sample_image_batch)
    expected_num_patches = calculate_num_patches(params)

    expected_sequence_length = expected_num_patches + 1
    expected_shape = (
        params["batch_size"],
        expected_sequence_length,
        params["embedding_dimension"],
    )
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    assert output.dtype == sample_image_batch.dtype


def test_conv_patch_embed_invalid_dimensions_init(params):
    """Tests if Conv init raises error for non-divisible image dimensions."""
    invalid_height = params["height"] - 1
    invalid_shape = (params["in_channels"], invalid_height, params["width"])

    with pytest.raises(ValueError):
        ConvolutionalPatchEmbedding(
            input_shape=invalid_shape,
            embedding_dimension=params["embedding_dimension"],
            patch_size=params["patch_size"],
        )


def test_conv_patch_embed_batch_independence(params, input_shape_tuple):
    """Tests if batch processing matches individual for Conv."""
    model = ConvolutionalPatchEmbedding(
        input_shape=input_shape_tuple,
        embedding_dimension=params["embedding_dimension"],
        patch_size=params["patch_size"],
    )

    model.eval()
    batch_input = torch.rand(
        params["batch_size"], params["in_channels"], params["height"], params["width"]
    )
    output_batch = model(batch_input)
    outputs_individual = []
    for i in range(params["batch_size"]):
        individual_input = batch_input[i : i + 1]
        output_ind = model(individual_input)
        outputs_individual.append(output_ind)
    output_concat = torch.cat(outputs_individual, dim=0)
    assert output_concat.shape == output_batch.shape
    assert torch.allclose(
        output_batch, output_concat, atol=1e-6
    ), "Conv Batch processing result differs from individual processing."


def test_manual_patch_embed_forward_shape(
    manual_patch_embed_model, sample_image_batch, params
):
    """Tests the output shape of the forward pass for Manual."""
    output = manual_patch_embed_model(sample_image_batch)
    expected_num_patches = calculate_num_patches(params)

    expected_sequence_length = expected_num_patches + 1
    expected_shape = (
        params["batch_size"],
        expected_sequence_length,
        params["embedding_dimension"],
    )
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    assert output.dtype == sample_image_batch.dtype


def test_manual_patch_embed_invalid_dimensions_init(params):
    """Tests if Manual init raises error for non-divisible image dimensions."""
    invalid_width = params["width"] - 1
    invalid_shape = (params["in_channels"], params["height"], invalid_width)

    with pytest.raises(ValueError):
        ManualPatchEmbedding(
            input_shape=invalid_shape,
            embedding_dimension=params["embedding_dimension"],
            patch_size=params["patch_size"],
        )


def test_manual_patch_embed_batch_independence(params, input_shape_tuple):
    """Tests if batch processing matches individual for Manual."""
    model = ManualPatchEmbedding(
        input_shape=input_shape_tuple,
        embedding_dimension=params["embedding_dimension"],
        patch_size=params["patch_size"],
    )

    model.eval()
    batch_input = torch.rand(
        params["batch_size"], params["in_channels"], params["height"], params["width"]
    )
    output_batch = model(batch_input)
    outputs_individual = []
    for i in range(params["batch_size"]):
        individual_input = batch_input[i : i + 1]
        output_ind = model(individual_input)
        outputs_individual.append(output_ind)
    output_concat = torch.cat(outputs_individual, dim=0)
    assert output_concat.shape == output_batch.shape
    assert torch.allclose(
        output_batch, output_concat, atol=1e-6
    ), "Manual Batch processing result differs from individual processing."