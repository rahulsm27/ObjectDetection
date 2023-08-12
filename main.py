from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

video_path = 'traffic.mp4'

predictions = model.predict(video_path)
predictions.show()
predictions.save("pose_elephant_flip_prediction.mp4")