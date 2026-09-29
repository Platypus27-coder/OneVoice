"""Local VI->EN microphone GUI using the real streaming pipeline, not a mock."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import queue
import sys
import threading
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@dataclass
class LiveTextState:
    raw_asr: str = ""
    final_asr: str = ""
    source: str = ""
    translation: str = ""
    suppressed: bool = False

    def apply(self, event: dict) -> None:
        if event["kind"] == "asr":
            self.raw_asr = event.get("text", "")
            if event.get("endpoint"):
                self.final_asr = event.get("hypothesis", self.raw_asr)
        elif event["kind"] == "translation":
            # Keep the exact source paired with this translation. A newer
            # microphone hypothesis must not relabel an older MT result.
            self.source = event["source"]
            self.translation = event["text"]
            self.suppressed = event.get("suppressed", False)


class DesktopLiveWindow:
    def __init__(self, root, args):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.root, self.args = root, args
        self.events = queue.SimpleQueue()
        self.state = LiveTextState()
        self.pipeline = None
        self.worker = None
        self.stop_requested = threading.Event()
        self.closing = False
        self.failed = False
        root.title("OneVoice — LIVE VI → EN · Offline")
        root.geometry("1040x790")
        root.minsize(760, 620)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="VI → EN · Micro laptop → Loa laptop",
                  font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(frame, text="ASR tạm cập nhật khi nói; dịch theo cụm khi ngắt khoảng 0,5 giây. "
                  "Micro tạm ngưng nhận lúc loa phát (half-duplex, không phải AEC).",
                  wraplength=960).pack(anchor="w", pady=(4, 10))
        controls = ttk.Frame(frame)
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="Bắt đầu", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Dừng", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        ttk.Label(controls, text="Ngưỡng VAD:").pack(side="left", padx=(12, 4))
        self.vad = tk.StringVar(value=str(args.vad_energy_threshold))
        self.vad_entry = ttk.Entry(controls, textvariable=self.vad, width=9)
        self.vad_entry.pack(side="left")
        self.status = tk.StringVar(value="Chưa bật micro. Nhấn Bắt đầu khi sẵn sàng.")
        ttk.Label(frame, textvariable=self.status, wraplength=960).pack(anchor="w", pady=8)
        self.raw = self._text_box(frame, "VI · ASR gốc đang nhận diện — tạm, có thể thay đổi", 3)
        self.final = self._text_box(frame, "VI · ASR cuối cụm — chưa phải nhãn đúng/sai", 3)
        self.source = self._text_box(frame, "VI · Nguồn của bản dịch bên dưới", 2)
        self.translated = self._text_box(frame, "EN · Bản dịch thật từ pipeline", 3)
        ttk.Label(frame, text="Lịch sử cụm dịch / lỗi (không sửa text nhận diện bằng câu mẫu)").pack(anchor="w")
        self.history = ScrolledText(frame, height=7, wrap="word", font=("Segoe UI", 10), state="disabled")
        self.history.pack(fill="both", expand=True, pady=(3, 0))
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def _text_box(self, parent, label, height):
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        ttk.Label(parent, text=label).pack(anchor="w")
        widget = ScrolledText(parent, height=height, wrap="word", font=("Segoe UI", 12), state="disabled")
        widget.pack(fill="x", pady=(3, 8))
        return widget

    @staticmethod
    def _replace(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def _append(self, text):
        self.history.configure(state="normal")
        self.history.insert("end", text + "\n\n")
        if int(self.history.index("end-1c").split(".")[0]) > 400:
            self.history.delete("1.0", "201.0")
        self.history.see("end")
        self.history.configure(state="disabled")

    def start(self):
        from tkinter import messagebox
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            threshold = float(self.vad.get())
            if not 0 < threshold <= 1:
                raise ValueError("Ngưỡng VAD phải lớn hơn 0 và không vượt quá 1.")
        except ValueError as exc:
            messagebox.showerror("Ngưỡng VAD không hợp lệ", str(exc))
            return
        self.stop_requested.clear()
        self.failed = False
        self.pipeline = None
        self.start_button.configure(state="disabled")
        self.vad_entry.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.set("Đang nạp model local; chưa bật micro...")
        self.worker = threading.Thread(target=self.run_pipeline, args=(threshold,), daemon=True)
        self.worker.start()

    def run_pipeline(self, threshold):
        import os
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            # Bundle paths are relative to the bundle, never the shell's cwd.
            os.chdir(self.args.bundle_dir)
            import onnxruntime  # Prime ORT before sherpa-onnx on Windows.
            from pipeline import OneVoicePipeline
            import sounddevice as sd
            sd.query_devices(self.args.input_device, kind="input")
            sd.query_devices(self.args.output_device, kind="output")
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            report_dir = self.args.report_dir / stamp
            pipeline = OneVoicePipeline(
                config_path=str(self.args.bundle_dir / "runtime_config.yaml"),
                direction="vi2en", profile="edge", offline=True,
                report_dir=str(report_dir), vad_energy_threshold=threshold,
                input_device=self.args.input_device, output_device=self.args.output_device,
                tts_backend="sapi", windows_en_voice=self.args.windows_en_voice,
                windows_en_rate=self.args.windows_en_rate,
                event_callback=self.events.put, pause_mic_during_playback=True,
            )
            # Match the listening setup already heard clearly, per run only.
            pipeline.tts.playback_options.update(gain=0.7, stereo=True, latency="high")
            self.pipeline = pipeline
            self.events.put({"kind": "report", "path": str(report_dir)})
            if self.stop_requested.is_set():
                pipeline.cancel()
                return
            pipeline.start()
        except Exception as exc:
            traceback.print_exc()
            self.events.put({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.events.put({"kind": "finished"})

    def stop(self):
        self.stop_requested.set()
        self.status.set("Đang dừng; chờ tác vụ đang xử lý kết thúc...")
        self.stop_button.configure(state="disabled")
        if self.pipeline is not None:
            self.pipeline.cancel()
            import sounddevice as sd
            sd.stop()  # Stop this process's convenience playback, not system audio.

    def close(self):
        self.closing = True
        self.stop()

    def poll(self):
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event["kind"]
            self.state.apply(event)
            if kind == "asr":
                self._replace(self.raw, self.state.raw_asr or "(ASR chưa nhận được chữ)")
                if event.get("endpoint"):
                    self._replace(self.final, self.state.final_asr)
            elif kind == "translation":
                self._replace(self.source, self.state.source)
                self._replace(self.translated, self.state.translation)
                suffix = " [KHÔNG PHÁT: validator chặn]" if self.state.suppressed else ""
                self._append(f"VI: {self.state.source}\nEN: {self.state.translation}\nRoute: {event['route']}{suffix}")
            elif kind == "status" and not self.failed and not self.stop_requested.is_set():
                self.status.set(event["message"])
            elif kind == "playback" and not self.failed and not self.stop_requested.is_set():
                self.status.set("Đang phát EN — tạm ngưng nhận micro; chờ nói tiếp."
                                if event["state"] == "started" else "Đang nghe — bạn có thể nói tiếp.")
            elif kind == "error":
                self.failed = True
                self.status.set("LỖI: " + event["message"])
                self._append("LỖI: " + event["message"])
            elif kind == "report":
                self._append("Báo cáo local: " + event["path"])
            elif kind == "finished":
                self.start_button.configure(state="normal")
                self.vad_entry.configure(state="normal")
                self.stop_button.configure(state="disabled")
                if not self.failed:
                    self.status.set("Đã dừng. Báo cáo được lưu local; nhấn Bắt đầu để thử lại.")
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(100, self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path(r"D:\OneVoiceDesktop\onevoice-v2-rc1\vi2en"))
    parser.add_argument("--report-dir", type=Path, default=REPO / "reports" / "desktop_live_vi2en")
    parser.add_argument("--input-device", default="Microphone Array MME")
    parser.add_argument("--output-device", default="Speakers Realtek WASAPI")
    parser.add_argument("--windows-en-voice", default="Microsoft Zira Desktop")
    parser.add_argument("--windows-en-rate", type=int, default=-2)
    parser.add_argument("--vad-energy-threshold", type=float, default=0.015)
    args = parser.parse_args()
    args.bundle_dir = args.bundle_dir.resolve()
    args.report_dir = args.report_dir.resolve()
    if not (args.bundle_dir / "runtime_config.yaml").is_file():
        parser.error(f"Không tìm thấy runtime_config.yaml: {args.bundle_dir}")
    if not -10 <= args.windows_en_rate <= 10:
        parser.error("--windows-en-rate must be between -10 and 10")
    if not 0 < args.vad_energy_threshold <= 1:
        parser.error("--vad-energy-threshold must be greater than 0 and at most 1")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    import tkinter as tk
    root = tk.Tk()
    DesktopLiveWindow(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
