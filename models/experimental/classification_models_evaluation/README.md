# Classification Evaluation

- Using `imagenet-1k` validation dataset.

### The below observations are for ttnn_model vs dataset(ground truth data):

The following model is evaluated(Correct_predictions/Total_predictions) for 512 samples.:-
-   Segformer(batch_size=1)     - **30**
-   Vit(batch_size=8)           - **81**
-   Resnet50(batch_size=16)     - **79**
-   MobileNetV2(batch_size=8)   - **70**

Currently, The number of samples is set to 512.

To run the test of ttnn vs ground truth, please follow the following commands:

**Segformer:** <br>
**_For 512x512,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_segformer_classification[tt_model-device_params0]
 ```

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_vit_classification[wormhole_b0-tt_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_resnet50_classification[16-act_dtype0-weight_dtype0-device_params0-tt_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_mobilenetv2_classification[8-224-tt_model-device_params0]
 ```

### The below observations are for torch_model vs dataset(ground truth data):

The following model is evaluated(Correct_predictions/Total_predictions) for 512 samples.:-
-   Segformer(batch_size=1)     - **72**
-   Vit(batch_size=8)           - **82**
-   Resnet50(batch_size=16)     - **76**
-   MobileNetV2(batch_size=8)   - **65**

To run the test of ttnn vs ground truth, please follow the following commands:

**Segformer:** <br>
**_For 512x512,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_segformer_classification[torch_model-device_params0]
 ```

**Vit:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_vit_classification[wormhole_b0-torch_model-8-device_params0]
 ```

**Resnet50:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_resnet50_classification[16-act_dtype0-weight_dtype0-device_params0-torch_model]
 ```

**MobileNetV2:** <br>
**_For 224x224,_**<br>
 ```sh
 pytest models/experimental/classification_models_evaluation/classification_eval.py::test_mobilenetv2_classification[8-224-torch_model-device_params0]
 ```
