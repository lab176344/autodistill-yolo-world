import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLOWorld
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class YoloWorld(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology, model_type: str = "yolov8s-world.pt"):
        self.ontology = ontology
        self.standard_model = YOLOWorld(model_type)
        labels = self.ontology.prompts()
        self.standard_model.set_classes(labels)
        # save the model for later use

    def predict(self, input: str, confidence=0.1) -> sv.Detections:
        labels = self.ontology.prompts()
        self.model.set_classes(labels)

        with torch.no_grad():

            outputs = self.model(input)

            for result in outputs:
                boxes = result.boxes
                scores = result.probs
                labels = result.labels

            # filter with score < confidence
            boxes = [box for box, score in zip(
                boxes, scores) if score > confidence]
            labels = [label for label, score in zip(
                labels, scores) if score > confidence]
            scores = [score for score in scores if score > confidence]

            if len(boxes) == 0:
                return sv.Detections.empty()

            detections = sv.Detections(
                xyxy=np.array(boxes),
                class_id=np.array(labels),
                confidence=np.array(scores),
            )

            return detections
