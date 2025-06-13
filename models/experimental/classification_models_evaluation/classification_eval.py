# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from loguru import logger

import torch
import torchvision
import pytest
import transformers
from transformers import AutoImageProcessor

import ttnn
from models.experimental.classification_models_evaluation.classification_eval_utils import get_data_loader


def evaluation(
    device=None,
    model=None,
    inputs=None,
    parameters=None,
    model_name=None,
    model_type=None,
    model_location_generator=None,
    image_processor=None,
    config=None,
    get_batch=None,
    imagenet_label_dict=None,
    batch_size=None,
    res=None,
):
    # Loading the dataset
    input_loc = str(model_location_generator("ImageNet_data"))
    iterations = 512
    # iteration dataset, preprocessing
    data_loader = get_data_loader(input_loc, batch_size, iterations // batch_size)
    gt_id = []
    pred_id = []
    for i in range(iterations // batch_size):
        if model_name in ["vit", "resnet50"]:
            inputs, labels = get_batch(data_loader, image_processor)
        elif model_name == "mobilenetv2":
            inputs, labels = get_batch(data_loader, res)
        else:
            inputs, labels = get_batch(data_loader)
            inputs = image_processor(inputs, return_tensors="pt")

        # preprocess
        if model_name == "Segformer":
            torch_input_tensor = inputs.pixel_values
            torch_input_tensor_permuted = torch.permute(torch_input_tensor, (0, 2, 3, 1))
            ttnn_input_tensor = ttnn.from_torch(
                torch_input_tensor_permuted,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
        elif model_name == "mobilenetv2":
            torch_input_tensor = inputs.reshape(batch_size, 3, res, res)
            tt_inputs = torch.permute(torch_input_tensor, (0, 2, 3, 1))
            ttnn_input_tensor = tt_inputs.reshape(
                1,
                1,
                tt_inputs.shape[0] * tt_inputs.shape[1] * tt_inputs.shape[2],
                tt_inputs.shape[3],
            )

            ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)
        elif model_name == "resnet50":
            torch_input_tensor = inputs
            if model_type == "tt_model":
                tt_inputs_host = ttnn.from_torch(
                    inputs,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=model.test_infra.inputs_mesh_mapper,
                )
        else:
            torch_input_tensor = inputs
            if model_type == "tt_model":
                tt_inputs = torch.permute(inputs, (0, 2, 3, 1))
                tt_inputs = torch.nn.functional.pad(tt_inputs, (0, 1, 0, 0, 0, 0, 0, 0))
                batch_size, img_h, img_w, img_c = tt_inputs.shape  # permuted input NHWC
                patch_size = config.patch_size
                tt_inputs = tt_inputs.reshape(batch_size, img_h, img_w // patch_size, 4 * patch_size)

                tt_inputs_host = ttnn.from_torch(
                    tt_inputs,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=model.test_infra.inputs_mesh_mapper,
                )

        # Inference
        if model_type == "tt_model":
            if model_name == "Segformer":
                output = model(
                    ttnn_input_tensor,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=None,
                    parameters=parameters,
                    model=None,
                )
            elif model_name == "mobilenetv2":
                output = model(ttnn_input_tensor)
            elif model_name == "vit":
                output = model.execute_vit_trace_2cqs_inference(tt_inputs_host)
            elif model_name == "resnet50":
                output = model.execute_resnet50_trace_2cqs_inference(tt_inputs_host)
        elif model_type == "torch_model":
            output = model(torch_input_tensor)

        # post_process
        if model_name == "Segformer":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output.logits)
            else:
                final_output = output.logits
            predicted_id = final_output.argmax(-1).item()
            predicted_label = imagenet_label_dict[predicted_id]

            pred_id.append(predicted_id)
            gt_id.append(labels[0])
        elif model_name == "mobilenetv2":
            if model_type == "tt_model":
                final_output = ttnn.from_device(output, blocking=True).to_torch().to(torch.float)
            else:
                final_output = output
            prediction = final_output.argmax(dim=-1)

            for i in range(batch_size):
                pred_id.append(prediction[i].item())
                gt_id.append(labels[i])

            del output, final_output

        elif model_name == "vit":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                predicted_id = final_output[:, 0, :1000].argmax(dim=-1)
            else:
                final_output = output.logits
                predicted_id = final_output.argmax(dim=-1)

            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                gt_id.append(labels[i])
        elif model_name == "resnet50":
            if model_type == "tt_model":
                final_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
                predicted_id = final_output[:, 0, 0, :].argmax(dim=-1)
            else:
                final_output = output
                predicted_id = final_output.argmax(dim=-1)

            for i in range(batch_size):
                pred_id.append(predicted_id[i].item())
                gt_id.append(labels[i])

    # Evaluation : Here we use correct_predection/total items
    correct_prediction = 0
    correct_prediction = sum(1 for a, b in zip(gt_id, pred_id) if a == b)
    accuracy = correct_prediction / len(pred_id)

    logger.info(f"accuracy: {accuracy:.2f}%")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
def test_segformer_classification(device, model_type, model_location_generator, imagenet_label_dict):
    from transformers import SegformerForImageClassification
    from models.demos.segformer.demo.classification_demo_utils import get_batch
    from ttnn.model_preprocessing import preprocess_model_parameters

    from models.demos.segformer.reference.segformer_for_image_classification import (
        SegformerForImageClassificationReference,
    )
    from models.demos.segformer.tt.ttnn_segformer_for_image_classification import TtSegformerForImageClassification
    from tests.ttnn.integration_tests.segformer.test_segformer_for_image_classification import (
        create_custom_preprocessor,
    )
    from tests.ttnn.integration_tests.segformer.test_segformer_model import move_to_device

    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0")
    torch_model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0").to(torch.bfloat16)
    reference_model = SegformerForImageClassificationReference(config=torch_model.config)

    reference_model.load_state_dict(torch_model.state_dict())
    reference_model.eval()
    if model_type == "tt_model":
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=None,
        )
        parameters = move_to_device(parameters, device)
        ttnn_model = TtSegformerForImageClassification(torch_model.config, parameters)

    evaluation(
        device=device,
        model=ttnn_model if model_type == "tt_model" else torch_model.to(torch.float),
        model_location_generator=model_location_generator,
        parameters=parameters if model_type == "tt_model" else None,
        model_type=model_type,
        model_name="Segformer",
        image_processor=image_processor,
        config=reference_model.config,
        get_batch=get_batch,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=1,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "num_command_queues": 2, "trace_region_size": 1700000}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    ((8),),
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
def test_vit_classification(
    mesh_device,
    model_type,
    use_program_cache,
    batch_size_per_device,
    imagenet_label_dict,
    model_location_generator,
):
    from models.demos.vit.tests.vit_performant_imagenet import VitTrace2CQ
    from transformers import ViTForImageClassification
    from models.demos.wormhole.vit.demo.vit_helper_funcs import get_batch

    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    if model_type == "torch_model":
        torch_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    else:
        vit_trace_2cq = VitTrace2CQ()

        vit_trace_2cq.initialize_vit_trace_2cqs_inference(
            mesh_device,
            batch_size_per_device,
        )

    model_version = "google/vit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(model_version)
    config = transformers.ViTConfig.from_pretrained(model_version)

    evaluation(
        device=mesh_device,
        model=vit_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        parameters=None,
        model_type=model_type,
        model_name="vit",
        image_processor=image_processor,
        config=config,
        get_batch=get_batch,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=batch_size,
    )


@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 1605632, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((16, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
def test_resnet50_classification(
    device,
    model_type,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    imagenet_label_dict,
    use_program_cache,
):
    from models.demos.ttnn_resnet.tests.resnet50_performant_imagenet import ResNet50Trace2CQ
    from models.demos.ttnn_resnet.tests.demo_utils import get_batch

    if model_type == "torch_model":
        torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        resnet50_trace_2cq = ResNet50Trace2CQ()

        resnet50_trace_2cq.initialize_resnet50_trace_2cqs_inference(
            device,
            batch_size,
            act_dtype,
            weight_dtype,
        )

    image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    evaluation(
        device=device,
        model=resnet50_trace_2cq if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        parameters=None,
        model_type=model_type,
        model_name="resnet50",
        image_processor=image_processor,
        config=None,
        get_batch=get_batch,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=batch_size,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize("batch_size, res", [[8, 224]])
def test_mobilenetv2_classification(
    device, model_type, batch_size, res, imagenet_label_dict, model_location_generator, use_program_cache
):
    from models.demos.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
    from models.demos.mobilenetv2.tt import ttnn_mobilenetv2
    import torchvision.models as models
    from models.demos.mobilenetv2.tt.model_preprocessing import create_mobilenetv2_model_parameters
    from models.demos.mobilenetv2.demo.demo_utils import get_batch

    weights_path = "models/demos/mobilenetv2/mobilenet_v2-b0353104.pth"

    if model_type == "torch_model":
        torch_model = models.mobilenet_v2(pretrained=True)
    else:
        if not os.path.exists(weights_path):
            os.system("bash models/demos/mobilenetv2/weights_download.sh")

        reference_model = Mobilenetv2()

        state_dict = torch.load(weights_path)
        ds_state_dict = {k: v for k, v in state_dict.items()}
        new_state_dict = {
            name1: parameter2
            for (name1, _), (_, parameter2) in zip(reference_model.state_dict().items(), ds_state_dict.items())
            if isinstance(parameter2, torch.FloatTensor)
        }
        reference_model.load_state_dict(new_state_dict)

        reference_model.eval()

        parameters = create_mobilenetv2_model_parameters(reference_model, device=device)

        ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(parameters, device, batchsize=batch_size)

    evaluation(
        device=device,
        model=ttnn_model if model_type == "tt_model" else torch_model,
        model_location_generator=model_location_generator,
        parameters=None,
        model_type=model_type,
        model_name="mobilenetv2",
        image_processor=None,
        config=None,
        get_batch=get_batch,
        imagenet_label_dict=imagenet_label_dict,
        batch_size=batch_size,
        res=res,
    )
