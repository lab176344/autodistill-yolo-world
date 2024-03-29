# Autodistill: YOLO-World Base Model

This repository contains the code implementing [YOLO-World](https://github.com/AILab-CVC/YOLO-World) as a Base Model for use with [`autodistill`](https://github.com/autodistill/autodistill).

YOLO-World combines [YOLO-World](https://github.com/AILab-CVC/YOLO-World) brings YOLO like efficiency for training and inferring open-vocabulary models

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).


## Installation

To use the YOLO-World, simply install it along with a Target Model supporting the `detection` task:

```bash
pip3 install autodistill-yolo-world
```

You can find a full list of `detection` Target Models on [the main autodistill repo](https://github.com/autodistill/autodistill).

## Quickstart

```python
from autodistill_yolo_world import YoloWorld
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

# define an ontology to map class names to our GroundedSAM prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = YoloWorld(
    ontology=CaptionOntology(
        {
            "person": "person",
            "car": "car",
        }
    ),
    model_type = "yolov8s-world.pt"
)

# run inference on a single image
results = base_model.predict("assets/test.jpg")

plot(
    image=cv2.imread("assets/test.jpg"),
    classes=base_model.ontology.classes(),
    detections=results
)
# label all images in a folder called `context_images`
base_model.label("./context_images", extension=".jpeg")
```

## License

The code in this repository is licensed under an [Apache 2.0 license](LICENSE).

## 🏆 Thanks

Thanks to [autodistill](https://github.com/autodistill/autodistill) and [ultralytics](https://github.com/ultralytics)
