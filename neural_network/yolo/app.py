import supervision as sv
import cv2
from ultralytics import YOLO
import numpy as np

# Carga el modelo
model = YOLO("yolo11n.pt")
# Inicializa el anotador de cajas
bounding_box_annotator = sv.BoxAnnotator()  # ya no necesita argumentos
POLYGON = np.array(
    [
        [75.89, 1062.05],
        [864.61, 525.64],
        [1035.89, 416.92],
        [1139.48, 312.30],
        [1238.97, 216.92],
        [1334.35, 112.30],
        [1446.15, 114.35],
        [1432.82, 160.51],
        [1471.79, 209.74],
        [1583.58, 302.05],
        [1698.46, 425.12],
        [1919.99, 651.79],
        [1919.99, 1079.48],
    ],
    dtype=np.int32,
)
polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors=(sv.Position.CENTER,))


def main():
    video_file_path = "./video.mp4"

    frame_generator = sv.get_video_frames_generator(source_path=video_file_path)

    for i, frame in enumerate(frame_generator):
        # Procesa el frame con YOLO
        result = model(frame, device="cuda", verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        # Anota las detecciones en una copia del frame
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame, polygon=POLYGON, color=sv.Color.RED, thickness=2
        )
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
