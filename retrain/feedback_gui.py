import os
import csv
import time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

def open_feedback_gui(crops, predictions, track_ids, class_names):
    feedback_dir = "retrain/feedback_data"
    os.makedirs(feedback_dir, exist_ok=True)
    csv_path = "retrain/feedback_labels.csv"

    root = tk.Tk()
    root.title("Feedback GUI")
    root.geometry("600x900")

    canvas = tk.Canvas(root)
    scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    feedback_data = []
    for i, (track_id, pred_label) in enumerate(zip(track_ids, predictions)):
        try:
            crop = crops[i]
        except IndexError:
            continue

        obj_frame = tk.Frame(scroll_frame, bd=2, relief="groove", padx=10, pady=10)
        obj_frame.pack(pady=5, padx=10, fill="x")

        img = crop.resize((100, 100))
        img_tk = ImageTk.PhotoImage(img)
        img_label = tk.Label(obj_frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack(side="left")

        info_frame = tk.Frame(obj_frame)
        info_frame.pack(side="left", padx=10)
        tk.Label(info_frame, text=f"ID: {track_id} | Pred: {class_names[pred_label]}", font=("Arial", 10, "bold")).pack(anchor="w")

        var = tk.StringVar(value=class_names[pred_label])
        dropdown = ttk.Combobox(info_frame, textvariable=var, values=class_names, width=25)
        dropdown.pack(anchor="w")

        skip_var = tk.BooleanVar()
        tk.Checkbutton(info_frame, text="Don't save", variable=skip_var).pack(anchor="w")

        def on_change(event=None, v=var, d=dropdown, f=obj_frame, original=class_names[pred_label]):
            f.config(bg="#fff7e6" if v.get() != original else "SystemButtonFace")

        dropdown.bind("<<ComboboxSelected>>", on_change)

        feedback_data.append((crop, track_id, var, skip_var, i))

    def save_feedback():
        saved = 0
        timestamp = int(time.time())
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for crop, track_id, var, skip_var, i in feedback_data:
                if skip_var.get():
                    continue
                label = var.get()
                label_dir = os.path.join(feedback_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                filename = f"obj_{timestamp}_{i}.jpg"
                path = os.path.join(label_dir, filename)
                crop.save(path)
                writer.writerow([os.path.join(label, filename), label])
                saved += 1
        messagebox.showinfo("Saved", f"Saved {saved} samples")
        root.destroy()

    tk.Button(root, text="Submit Feedback", command=save_feedback, bg="#4CAF50", fg="white",
              font=("Arial", 11, "bold"), padx=15, pady=8).pack(pady=10)

    root.mainloop()
