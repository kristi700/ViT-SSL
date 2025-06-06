import torch
import pytest

from vit_core.vit import ViT


@pytest.fixture(scope="module")
def params():
    """Test parameters"""
    return {
        "batch_size": 4,
        "in_channels": 3,
        "height": 32,
        "width": 32,
        "patch_size": 8,
        "num_classes": 10,
        "embed_dim": 128,
        "num_blocks": 2,
        "num_heads": 4,
        "mlp_dim": 256,
        "dropout": 0.0,
    }


@pytest.fixture(scope="module")
def image_shape_tuple(params):
    """Provides image shape as a tuple (C, H, W)."""
    return (params["in_channels"], params["height"], params["width"])


@pytest.fixture(scope="module")
def vit_model(params, image_shape_tuple):
    """Provides an instance of the ViT model."""
    model = ViT(
        input_shape=image_shape_tuple,
        patch_size=params["patch_size"],
        num_classes=params["num_classes"],
        embed_dim=params["embed_dim"],
        num_blocks=params["num_blocks"],
        num_heads=params["num_heads"],
        mlp_dim=params["mlp_dim"],
        dropout=params["dropout"],
    )
    model.eval()
    return model


@pytest.fixture
def sample_image_batch(params):
    """Provides a sample batch of image tensors."""
    return torch.rand(
        params["batch_size"], params["in_channels"], params["height"], params["width"]
    )


def test_vit_forward_shape(vit_model, sample_image_batch, params):
    """Tests the output shape of the ViT forward pass."""
    output = vit_model(sample_image_batch)
    expected_shape = (params["batch_size"], params["num_classes"])
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"
    assert output.dtype == torch.float32


def test_vit_forward_value_change(vit_model, sample_image_batch):
    """Tests if the forward pass changes input representation."""
    input_clone = sample_image_batch.clone()
    output = vit_model(sample_image_batch)

    assert torch.equal(input_clone, sample_image_batch)
    assert torch.isfinite(output).all()
    assert output.abs().sum() > 1e-9


@pytest.mark.parametrize("batch_size", [1, 3])
def test_vit_batch_independence(params, image_shape_tuple, batch_size):
    """Tests if batch processing matches individual processing."""

    current_params = params.copy()
    current_params["batch_size"] = batch_size
    model = ViT(
        input_shape=image_shape_tuple,
        patch_size=current_params["patch_size"],
        num_classes=current_params["num_classes"],
        embed_dim=current_params["embed_dim"],
        num_blocks=current_params["num_blocks"],
        num_heads=current_params["num_heads"],
        mlp_dim=current_params["mlp_dim"],
        dropout=current_params["dropout"],
    )
    model.eval()

    batch_input = torch.rand(
        current_params["batch_size"],
        current_params["in_channels"],
        current_params["height"],
        current_params["width"],
    )

    output_batch = model(batch_input)

    outputs_individual = []
    for i in range(current_params["batch_size"]):
        individual_input = batch_input[i : i + 1]
        output_ind = model(individual_input)
        outputs_individual.append(output_ind)

    output_concat = torch.cat(outputs_individual, dim=0)

    assert output_concat.shape == output_batch.shape
    assert torch.allclose(
        output_batch, output_concat, atol=1e-6
    ), "ViT Batch processing result differs from individual processing."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vit_device_cuda(params, image_shape_tuple):
    """Tests if the ViT model works on CUDA device."""
    device = torch.device("cuda")
    model = ViT(
        input_shape=image_shape_tuple,
        patch_size=params["patch_size"],
        num_classes=params["num_classes"],
        embed_dim=params["embed_dim"],
        num_blocks=params["num_blocks"],
        num_heads=params["num_heads"],
        mlp_dim=params["mlp_dim"],
        dropout=params["dropout"],
    ).to(device)
    model.eval()

    sample_input_cuda = torch.rand(
        params["batch_size"], params["in_channels"], params["height"], params["width"]
    ).to(device)

    output = model(sample_input_cuda)

    expected_shape = (params["batch_size"], params["num_classes"])
    assert output.shape == expected_shape
    assert output.device.type == "cuda"


def test_vit_device_cpu(params, image_shape_tuple):
    """Tests if the ViT model works on CPU device."""
    device = torch.device("cpu")
    model = ViT(
        input_shape=image_shape_tuple,
        patch_size=params["patch_size"],
        num_classes=params["num_classes"],
        embed_dim=params["embed_dim"],
        num_blocks=params["num_blocks"],
        num_heads=params["num_heads"],
        mlp_dim=params["mlp_dim"],
        dropout=params["dropout"],
    ).to(device)
    model.eval()

    sample_input_cpu = torch.rand(
        params["batch_size"], params["in_channels"], params["height"], params["width"]
    ).to(device)

    output = model(sample_input_cpu)

    expected_shape = (params["batch_size"], params["num_classes"])
    assert output.shape == expected_shape
    assert output.device.type == "cpu"
