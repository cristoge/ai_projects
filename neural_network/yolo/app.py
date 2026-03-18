import supervision as sv
import cv2
from ultralytics import YOLO

# Carga el modelo
model = YOLO("yolo11n.pt")

# Inicializa el anotador de cajas
bounding_box_annotator = sv.BoxAnnotator()  # ya no necesita argumentos


def main():
    video_file_path = "data/video.mp4"

    frame_generator = sv.get_video_frames_generator(source_path=video_file_path)

    for i, frame in enumerate(frame_generator):
        # Procesa el frame con YOLO
        result = model(frame, device="cuda", verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Anota las detecciones en una copia del frame
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
