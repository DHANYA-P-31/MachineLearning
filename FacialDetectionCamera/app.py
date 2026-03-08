import os
import time
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class FaceDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Detection Camera UI")
        self.root.geometry("980x700")

        self.output_dir = Path(__file__).resolve().parent / "captures"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.output_dir / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.output_dir / "ann_face_model.yml"
        self.labels_path = self.output_dir / "ann_labels.json"

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.cap = None
        self.running = False
        self.detect_live = tk.BooleanVar(value=True)
        self.recognize_live = tk.BooleanVar(value=False)
        self.current_frame = None
        self.current_photo = None
        self.last_saved_image = None
        self.person_name_var = tk.StringVar(value="")
        self.input_size = 100 * 100
        self.face_model = None
        self.labels = []

        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill="both", expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 10))

        ttk.Button(controls, text="Start Camera", command=self.start_camera).pack(side="left", padx=4)
        ttk.Button(controls, text="Stop Camera", command=self.stop_camera).pack(side="left", padx=4)
        ttk.Button(controls, text="Capture & Save", command=self.capture_image).pack(side="left", padx=4)
        ttk.Button(controls, text="Detect Last Capture", command=self.detect_last_capture).pack(side="left", padx=4)
        ttk.Button(controls, text="Detect From File", command=self.detect_from_file).pack(side="left", padx=4)
        ttk.Button(controls, text="Store Face Sample", command=self.store_face_sample).pack(side="left", padx=4)
        ttk.Button(controls, text="Train ANN", command=self.train_ann_model).pack(side="left", padx=4)
        ttk.Button(controls, text="Load ANN", command=self.load_ann_model).pack(side="left", padx=4)
        ttk.Button(controls, text="Recognize Current", command=self.recognize_current_frame).pack(side="left", padx=4)

        ttk.Checkbutton(
            controls,
            text="Live Face Detection",
            variable=self.detect_live,
        ).pack(side="right", padx=4)

        ttk.Checkbutton(
            controls,
            text="Live Recognition",
            variable=self.recognize_live,
        ).pack(side="right", padx=4)

        dataset_controls = ttk.Frame(container)
        dataset_controls.pack(fill="x", pady=(0, 8))
        ttk.Label(dataset_controls, text="Person Name:").pack(side="left", padx=(0, 6))
        ttk.Entry(dataset_controls, textvariable=self.person_name_var, width=24).pack(side="left")

        self.video_label = ttk.Label(container)
        self.video_label.pack(fill="both", expand=True)

        footer = ttk.Frame(container)
        footer.pack(fill="x", pady=(10, 0))

        self.path_var = tk.StringVar(value=f"Save folder: {self.output_dir}")
        ttk.Label(footer, textvariable=self.path_var).pack(side="left")
        ttk.Label(footer, textvariable=self.status_var).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_camera(self) -> None:
        if self.running:
            self.status_var.set("Camera already running")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_var.set("Could not open camera")
            messagebox.showerror("Camera Error", "Could not open the camera.")
            return

        self.running = True
        self.status_var.set("Camera started")
        self._update_frame()

    def stop_camera(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.running = False
        self.status_var.set("Camera stopped")

    def _update_frame(self) -> None:
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Failed to read frame")
            self.root.after(30, self._update_frame)
            return

        self.current_frame = frame.copy()
        display_frame = frame.copy()

        if self.detect_live.get() or self.recognize_live.get():
            display_frame = self._draw_faces(display_frame, recognize=self.recognize_live.get())

        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((950, 560), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.current_photo)

        self.root.after(15, self._update_frame)

    def _draw_faces(self, frame, recognize: bool = False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        recognized = 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if recognize and self.face_model is not None and self.labels:
                vector = self._face_vector_from_region(gray, (x, y, w, h))
                label, score = self._predict_label(vector)
                recognized += 1
                cv2.putText(
                    frame,
                    f"{label} ({score:.2f})",
                    (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        if recognize and self.face_model is not None and self.labels:
            self.status_var.set(f"Live recognition: {recognized} face(s)")
        elif self.detect_live.get():
            self.status_var.set(f"Live detection: {len(faces)} face(s)")
        return frame

    def _face_vector_from_region(self, gray: np.ndarray, rect) -> np.ndarray:
        x, y, w, h = rect
        face_roi = gray[y : y + h, x : x + w]
        face_resized = cv2.resize(face_roi, (100, 100), interpolation=cv2.INTER_AREA)
        return (face_resized.astype(np.float32).reshape(1, -1) / 255.0)

    def _detect_faces(self, frame) -> tuple[np.ndarray, list]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        return gray, list(faces)

    def capture_image(self) -> None:
        if self.current_frame is None:
            self.status_var.set("No frame to capture")
            messagebox.showwarning("Capture", "Start camera first.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = self.output_dir / f"capture_{timestamp}.jpg"
        cv2.imwrite(str(file_path), self.current_frame)
        self.last_saved_image = file_path
        self.status_var.set(f"Saved: {file_path.name}")

    def store_face_sample(self) -> None:
        if self.current_frame is None:
            self.status_var.set("No frame for sample")
            messagebox.showwarning("Store Sample", "Start camera first.")
            return

        person_name = self.person_name_var.get().strip()
        if not person_name:
            self.status_var.set("Enter person name")
            messagebox.showwarning("Store Sample", "Enter a person name before storing samples.")
            return

        gray, faces = self._detect_faces(self.current_frame)
        if not faces:
            self.status_var.set("No face found in frame")
            messagebox.showwarning("Store Sample", "No face detected in current frame.")
            return

        largest_face = max(faces, key=lambda face: face[2] * face[3])
        face_vector = self._face_vector_from_region(gray, largest_face)
        face_img = (face_vector.reshape(100, 100) * 255.0).astype(np.uint8)

        person_dir = self.dataset_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)

        sample_name = f"sample_{time.strftime('%Y%m%d_%H%M%S')}.png"
        sample_path = person_dir / sample_name
        cv2.imwrite(str(sample_path), face_img)

        self.status_var.set(f"Stored sample for {person_name}: {sample_name}")

    def train_ann_model(self) -> None:
        samples = []
        targets = []
        label_names = []

        for person_dir in sorted(self.dataset_dir.iterdir()) if self.dataset_dir.exists() else []:
            if not person_dir.is_dir():
                continue

            label_index = len(label_names)
            label_names.append(person_dir.name)

            for image_file in person_dir.glob("*.png"):
                img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
                vector = img.astype(np.float32).reshape(-1) / 255.0
                samples.append(vector)
                targets.append(label_index)

        if len(samples) < 2 or len(label_names) < 2:
            self.status_var.set("Need at least 2 people with samples")
            messagebox.showwarning(
                "Train ANN",
                "Add samples for at least 2 different people before training.",
            )
            return

        sample_matrix = np.array(samples, dtype=np.float32)
        response_matrix = np.zeros((len(targets), len(label_names)), dtype=np.float32)
        for index, target in enumerate(targets):
            response_matrix[index, target] = 1.0

        ann = cv2.ml.ANN_MLP_create()
        ann.setLayerSizes(np.array([self.input_size, 128, len(label_names)], dtype=np.int32))
        ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 1.0, 1.0)
        ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001, 0.1)
        ann.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 400, 1e-4))

        trained = ann.train(sample_matrix, cv2.ml.ROW_SAMPLE, response_matrix)
        if not trained:
            self.status_var.set("Training failed")
            messagebox.showerror("Train ANN", "ANN training failed.")
            return

        ann.save(str(self.model_path))
        with open(self.labels_path, "w", encoding="utf-8") as labels_file:
            json.dump(label_names, labels_file)

        self.face_model = ann
        self.labels = label_names
        self.status_var.set(f"ANN trained with {len(samples)} sample(s)")

    def load_ann_model(self) -> None:
        if not self.model_path.exists() or not self.labels_path.exists():
            self.status_var.set("Train model first")
            messagebox.showwarning("Load ANN", "Model files not found. Train ANN first.")
            return

        with open(self.labels_path, "r", encoding="utf-8") as labels_file:
            labels = json.load(labels_file)

        self.face_model = cv2.ml.ANN_MLP_load(str(self.model_path))
        self.labels = labels
        self.status_var.set(f"Loaded ANN model: {len(self.labels)} class(es)")

    def _predict_label(self, face_vector: np.ndarray) -> tuple[str, float]:
        if self.face_model is None or not self.labels:
            return "Unknown", 0.0

        _, outputs = self.face_model.predict(face_vector.astype(np.float32))
        best_index = int(np.argmax(outputs[0]))
        best_score = float(outputs[0][best_index])
        label = self.labels[best_index] if 0 <= best_index < len(self.labels) else "Unknown"
        return label, best_score

    def recognize_current_frame(self) -> None:
        if self.current_frame is None:
            self.status_var.set("No frame to recognize")
            messagebox.showwarning("Recognize", "Start camera first.")
            return

        if self.face_model is None or not self.labels:
            self.status_var.set("Load/train ANN first")
            messagebox.showwarning("Recognize", "Train or load ANN model first.")
            return

        preview = self._draw_faces(self.current_frame.copy(), recognize=True)

        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((950, 560), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.current_photo)

    def detect_last_capture(self) -> None:
        if self.last_saved_image is None or not self.last_saved_image.exists():
            self.status_var.set("No captured image found")
            messagebox.showwarning("Detect", "Capture an image first.")
            return
        self._detect_and_save(self.last_saved_image)

    def detect_from_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
        )
        if not path:
            return
        self._detect_and_save(Path(path))

    def _detect_and_save(self, image_path: Path) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            self.status_var.set("Could not read image")
            messagebox.showerror("Image Error", "Could not read the selected image.")
            return

        processed = self._draw_faces(image.copy())
        output_path = self.output_dir / f"detected_{image_path.stem}.jpg"
        cv2.imwrite(str(output_path), processed)

        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((950, 560), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.current_photo)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        self.status_var.set(f"Detected {len(faces)} face(s), saved: {output_path.name}")

    def on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
