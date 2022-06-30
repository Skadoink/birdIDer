from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath("model_b.h5")
prediction.setJsonPath("model_class_b.json")
prediction.loadModel(num_objects=3)

predictions, probabilities = prediction.classifyImage("House-sparrow.jpeg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)