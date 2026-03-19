import supervision as sv
import cv2
from supervision.draw.color import Color
from ultralytics import YOLO
import numpy as np

# Carga el modelo
model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack(minimum_consecutive_frames=3)
tracker.reset()
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

CLASSES = [2]  # coches
label_annotator = sv.LabelAnnotator(text_color=Color.BLUE)
trace_annotator = sv.TraceAnnotator(trace_length=10)
LINE_1_START = sv.Point(1128, 314)
LINE_1_END = sv.Point(1626, 314)
LINE_ZONE = sv.LineZone(
    start=LINE_1_START, end=LINE_1_END, triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)
line_zone_annotator = sv.LineZoneAnnotator(
    text_orient_to_line=True,
    text_scale=0.8,
    custom_in_text="in:)",
    display_out_count=False,
)
line_zone_annotator_multiclass = sv.LineZoneAnnotatorMulticlass(
    text_scale=0.8, text_thickness=2, table_margin=20
)


def main():
    video_file_path = "./video.mp4"

    frame_generator = sv.get_video_frames_generator(
        source_path=video_file_path, stride=1
    )

    for i, frame in enumerate(frame_generator):
        # Procesa el frame con YOLO
        # YOLO
        result = model(frame, device="cuda", verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(result)

        # filtro zona
        detections = detections[polygon_zone.trigger(detections)]

        # filtro clases
        detections = detections[(detections.class_id == 2) | (detections.class_id == 3)]

        # tracking
        detections = tracker.update_with_detections(detections)

        # labels (AHORA sí correcto)
        labels = [f"#{tid}" for tid in detections.tracker_id]
        # Anota las detecciones en una copia del frame
        LINE_ZONE.trigger(detections=detections)
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame, polygon=POLYGON, color=sv.Color.RED, thickness=2
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = line_zone_annotator.annotate(
            annotated_frame, line_counter=LINE_ZONE
        )
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


main()
