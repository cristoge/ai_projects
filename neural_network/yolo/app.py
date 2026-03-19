import supervision as sv
import cv2
from supervision.draw.color import Color
from ultralytics import YOLO
import numpy as np

# Carga el modelo
model = YOLO("yolo11n.pt")

CLASSES = [2]  # coches
POLYGON = np.array(
    [
        [2.7507163323782233, 916.8481375358166],
        [613.4097421203438, 600.5157593123208],
        [874.7277936962751, 438.2234957020057],
        [1309.3409742120343, 135.64469914040114],
        [1037.0200573065902, 146.64756446991404],
        [695.9312320916905, 253.92550143266473],
        [283.323782234957, 391.4613180515759],
        [11.002865329512893, 490.48710601719193],
    ],
    dtype=np.int32,
)

polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors=(sv.Position.CENTER,))
tracker = sv.ByteTrack(minimum_consecutive_frames=3)
tracker.reset()
LINE_1_START = sv.Point(22, 501)
LINE_1_END = sv.Point(322, 768)
LINE_ZONE = sv.LineZone(
    start=LINE_1_START, end=LINE_1_END, triggering_anchors=(sv.Position.BOTTOM_CENTER,)
)

bounding_box_annotator = sv.BoxAnnotator()  # ya no necesita argumentos
label_annotator = sv.LabelAnnotator(text_color=Color.BLUE)
trace_annotator = sv.TraceAnnotator(trace_length=10)
line_zone_annotator = sv.LineZoneAnnotator(
    text_orient_to_line=True,
    text_scale=0.8,
    display_in_count=False,
    custom_out_text="cars out",
)


def main():
    video_file_path = "./video.mp4"

    frame_generator = sv.get_video_frames_generator(
        source_path=video_file_path, stride=1
    )

    for i, frame in enumerate(frame_generator):
        result = model(frame, device="cuda", verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections[(detections.class_id == 2) | (detections.class_id == 3)]
        detections = tracker.update_with_detections(detections)
        labels = []
        for tid in detections.tracker_id:
            labels.append(f"#{tid}")
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
