from imageai.Classification.Custom import ClassificationModelTrainer
import tensorflow as tf

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsResNet50()
model_trainer.setDataDirectory("birds")
with tf.device('/device:GPU:0'):
    model_trainer.trainModel(num_objects=3, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)
