from autodistill_yolo_world.yolo_world import YoloWorld
from autodistill.detection import CaptionOntology
import os


def test_YoloWorld():
    caption_ontology = CaptionOntology({"person": "person", "car": "car"})
    yolo_world = YoloWorld(ontology=caption_ontology)
    input_image = os.join("assets", "test.jpg")
    yolo_world.predict(input_image)
