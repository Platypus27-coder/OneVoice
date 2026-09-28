"""Publish a compact, current-PC report only after all benchmark stages finish."""

from __future__ import annotations

import argparse
import csv
import html
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_complete_run(root: Path) -> dict:
    directions = {}
    for direction in ("vi2en", "en2vi"):
        folder = root / direction
        paths = {
            "pipeline": folder / "pipeline_full_noisy/summary.json",
            "tts": folder / "tts_full_test/aggregate.json",
            "profile": folder / "profile/profile.json",
        }
        for name, path in paths.items():
            if not path.is_file():
                raise RuntimeError(f"Benchmark is not complete: missing {direction}/{name}: {path}")
        reports = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
        pipeline, tts, profile = (reports[name] for name in ("pipeline", "tts", "profile"))
        expected = sum(pipeline["available_cases"].values())
        if not pipeline.get("full_selected_corpus") or not pipeline.get("measurement_completed") or pipeline["completed_cases"] != expected:
            raise RuntimeError(f"{direction}: incomplete/sampled pipeline run cannot be published as full")
        if pipeline["requested_cases"] != expected or pipeline["passed_cases"] + pipeline["failed_cases"] != expected:
            raise RuntimeError(f"{direction}: inconsistent pipeline case counts")
        if sum(row["samples"] for row in pipeline["suites"].values()) != expected:
            raise RuntimeError(f"{direction}: pipeline suite counts do not cover the corpus")
        if len(pipeline["failures"]) != pipeline["failed_cases"]:
            raise RuntimeError(f"{direction}: pipeline failure list is incomplete")
        if not tts.get("full_prompt_csv") or tts.get("target_language") != ("en" if direction == "vi2en" else "vi"):
            raise RuntimeError(f"{direction}: TTS must cover the full CSV in the correct output language")
        if tts["passed_samples"] + tts["failed_samples"] != tts["samples"]:
            raise RuntimeError(f"{direction}: incomplete TTS ledger")
        tts_contract = folder / "tts_full_test/benchmark_contract.json"
        if not tts_contract.is_file() or len(json.loads(tts_contract.read_text(encoding="utf-8"))["prompts"]) != tts["samples"]:
            raise RuntimeError(f"{direction}: TTS does not cover every contracted prompt")
        validate_tts_ledger(folder / "tts_full_test", tts)
        if any(report.get("direction") != direction or not report.get("offline") for report in reports.values()):
            raise RuntimeError(f"{direction}: stage direction/offline metadata is inconsistent")
        if not profile.get("measurement_completed"):
            raise RuntimeError(f"{direction}: P5 measurement did not finish")
        contract_path = folder / "pipeline_full_noisy/benchmark_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        reports["diagnostics"] = case_diagnostics(folder / "pipeline_full_noisy", pipeline, contract)
        reports["corpus"] = {}
        for suite in pipeline["suites"]:
            cases = [row for row in contract["cases"] if row["suite"] == suite]
            reports["corpus"][suite] = {
                "entries": len(cases),
                "unique_audio_sha256": len({row["input_sha256"] for row in cases if row.get("input_sha256")}),
                "unique_source_texts": len({row["reference_text"] for row in cases if row.get("reference_text")}),
                "unique_source_ids": len({row["source_id"] for row in cases if row.get("source_id")}),
            }
        reports["provenance"] = {
            "pipeline_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            **{key: contract.get(key) for key in (
                "runtime_code_sha256", "config_sha256", "manifest_sha256", "artifact_manifest_sha256",
                "context_data_sha256", "safety_manifest_sha256", "platform", "python", "dependencies")},
            "tts_contract_sha256": hashlib.sha256(tts_contract.read_bytes()).hexdigest(),
        }
        directions[direction] = reports
    if len({reports["pipeline"]["git_revision"] for reports in directions.values()}) != 1 or len({reports["provenance"]["runtime_code_sha256"] for reports in directions.values()}) != 1:
        raise RuntimeError("Directions measured different runtime revisions; do not publish as one current run")
    return {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "directions": directions,
        "scope": "PC Windows offline, replay tăng tốc WAV tổng hợp và TTS hệ thống cục bộ. Thời gian tạo buffer audio không gồm driver hoặc truyền Bluetooth. Trường trọng yếu được đối chiếu với văn bản gốc bằng rule; chưa phải đánh giá ngữ nghĩa của người nghe.",
    }


def validate_tts_ledger(folder: Path, aggregate: dict) -> None:
    path = folder / "predictions.csv"
    if not path.is_file():
        raise RuntimeError("Missing per-prompt TTS evidence")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    prompts = json.loads((folder / "benchmark_contract.json").read_text(encoding="utf-8"))["prompts"]
    expected = {str(row["prompt_id"]): row["text"] for row in prompts}
    actual = {str(row["prompt_id"]): row["text"] for row in rows}
    if len(rows) != len(actual) or len(prompts) != len(expected) or actual != expected:
        raise RuntimeError("TTS prompt evidence does not match the contracted corpus")
    if any(row["status"] not in {"pass", "fail"} for row in rows) or sum(row["status"] == "pass" for row in rows) != aggregate["passed_samples"]:
        raise RuntimeError("TTS prompt statuses do not match aggregate counts")


def case_diagnostics(folder: Path, pipeline: dict, contract: dict) -> dict:
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((folder / "cases").glob("*/result.json"))]
    if len(rows) != pipeline["completed_cases"] or sum(row["status"] != "pass" for row in rows) != pipeline["failed_cases"]:
        raise RuntimeError("Per-case evidence does not match the completed pipeline summary")
    expected = {row["case_id"]: row for row in contract["cases"]}
    actual = {row["case_id"]: row for row in rows}
    if len(expected) != len(contract["cases"]) or len(actual) != len(rows) or actual.keys() != expected.keys():
        raise RuntimeError("Per-case identities do not match the contracted corpus")
    for case_id, row in actual.items():
        if row["status"] not in {"pass", "fail"} or any(row.get(key) != expected[case_id].get(key) for key in ("input_sha256", "reference_text", "reference_translation", "suite")):
            raise RuntimeError(f"Per-case evidence differs from contract: {case_id}")
    failure_evidence = {row["case_id"]: row.get("error") for row in rows if row["status"] != "pass"}
    if failure_evidence != {row["case_id"]: row.get("error") for row in pipeline["failures"]}:
        raise RuntimeError("Per-case failures do not match the published failure list")
    if Counter(row["suite"] for row in rows) != {name: row["samples"] for name, row in pipeline["suites"].items()}:
        raise RuntimeError("Per-case suite counts do not match the summary")
    functional = Counter(row.get("error", "unspecified") for row in rows if row["status"] != "pass")
    critical = Counter(error.split(":", 1)[0] for row in rows for error in row.get("reference_validation_errors", []))
    safety = [row for row in rows if row.get("suite") == "approved_safety"]
    agreements = 0
    safety_review = []
    for row in safety:
        emitted = [event.get("safety_id") for event in row.get("hypothesis_trace", [])
                   if event.get("decision") == "COMMIT_SAFETY"]
        agreements += emitted == [row.get("safety_id")]
        if row["status"] != "pass" or emitted != [row.get("safety_id")]:
            safety_review.append({
                **{key: row.get(key) for key in ("case_id", "status", "safety_id", "reference_text", "asr_text", "translation", "error")},
                "emitted_safety_ids": emitted,
            })
    examples = [{key: row.get(key) for key in (
        "case_id", "status", "reference_text", "asr_text", "translation", "error", "reference_validation_errors")}
        for row in rows if row.get("reference_validation_errors") or row["status"] != "pass"][:12]
    return {
        "functional_error_counts": dict(functional.most_common()),
        "reference_error_occurrences": dict(critical.most_common()),
        "functional_pass_but_reference_invalid": sum(row["status"] == "pass" and not row.get("reference_critical_fields_valid", False) for row in rows),
        "canonical_safety_id_agreement": {"samples": len(safety), "exact_id_agreements": agreements},
        "safety_review": safety_review,
        "examples": examples,
        "notes": "Reference errors are rule-based findings, not human semantic labels. Exact safety-ID agreement is stricter than route checks; equivalent approved phrases may have different IDs and require manual review.",
    }


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def milliseconds(value: float | None) -> str:
    return "—" if value is None else f"{value:,.1f} ms"


def overview_svg(data: dict) -> str:
    """Use separate scales for ratios and latency; never turn WER into accuracy."""
    counts = {reports["pipeline"]["suites"].get("approved_safety", {}).get("samples")
              for reports in data["directions"].values()}
    safety_caption = f"{next(iter(counts))} mục safety canonical mỗi chiều" if len(counts) == 1 and None not in counts else "safety canonical"
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="810" viewBox="0 0 1120 810">',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#e5e7eb}.title{font-size:23px;font-weight:650}.label{font-size:14px}.value{font-size:14px;font-weight:600}.note{font-size:12px;fill:#aeb7c6}.axis{stroke:#9aa4b2;stroke-width:1}</style>',
        '<rect width="1120" height="810" rx="14" fill="#0d1117"/>',
        '<text x="32" y="40" class="title">OneVoice · Benchmark pipeline PC sau sửa lỗi</text>',
        f'<text x="32" y="65" class="note">Full noisy test + {safety_caption} · Offline · Replay tăng tốc</text>',
    ]
    panels = [
        ("Noisy test: qua gate streaming (audio, route, rule, commit/frame)", "functional", 100.0, "%", 115, "Tỷ lệ lượt (%)"),
        ("Safety canonical: qua gate streaming (đúng safety route, không phát lặp)", "safety", 100.0, "%", 285, "Tỷ lệ entry (%)"),
        ("Noisy test: bảo toàn trường trọng yếu so với lời gốc", "critical", 100.0, "%", 455, "Tỷ lệ câu (%)"),
        ("Noisy normal: commit → buffer audio p95", "latency", None, "ms", 625, "Độ trễ p95 (ms)"),
    ]
    values_by_key = {}
    for direction, reports in data["directions"].items():
        noisy = reports["pipeline"]["suites"]["test_noisy"]
        normal = reports["pipeline"]["routes"].get("normal", {})
        safety = reports["pipeline"]["suites"].get("approved_safety", {})
        values_by_key[direction] = {
            "functional": noisy["passed"] / noisy["samples"] * 100,
            "critical": noisy["reference_critical_field_preservation"] * 100,
            "safety": safety["passed"] / safety["samples"] * 100 if safety.get("samples") else None,
            "latency": normal.get("commit_to_first_audio_p95_ms"),
            "samples": noisy["samples"], "normal_samples": normal.get("passed", 0),
            "safety_samples": safety.get("samples", 0),
        }
    for title, key, maximum, suffix, y, axis_title in panels:
        actual_values = [row[key] for row in values_by_key.values() if row[key] is not None]
        if not actual_values:
            continue
        maximum = maximum or max(actual_values) * 1.2 or 1
        lines.append(f'<text x="32" y="{y}" class="label">{html.escape(title)}</text>')
        chart_x, chart_width = 225, 800
        for index, (direction, row) in enumerate(values_by_key.items()):
            value = row[key]
            if value is None:
                continue
            row_y = y + 20 + index * 34
            label = "VI → EN" if direction == "vi2en" else "EN → VI"
            count = row["normal_samples"] if key == "latency" else row["safety_samples"] if key == "safety" else row["samples"]
            width = chart_width * min(value / maximum, 1)
            color = "#7dd3c2" if index == 0 else "#f19aaa"
            lines.extend([
                f'<text x="32" y="{row_y + 15}" class="label">{label} · n={count}</text>',
                f'<rect x="{chart_x}" y="{row_y}" width="{width:.2f}" height="22" rx="3" fill="{color}"/>',
                f'<text x="{min(chart_x + width + 10, 1030):.2f}" y="{row_y + 16}" class="value">{value:.1f} {suffix}</text>',
            ])
        axis_y = y + 94
        lines.append(f'<path d="M{chart_x} {axis_y}H{chart_x + chart_width}" class="axis"/>')
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            x = chart_x + chart_width * fraction
            lines.append(f'<text x="{x}" y="{axis_y + 18}" text-anchor="middle" class="note">{maximum * fraction:.0f}</text>')
        lines.append(f'<text x="{chart_x + chart_width / 2}" y="{axis_y + 36}" text-anchor="middle" class="note">{axis_title}</text>')
    date = data.get("generated_at", "")[:10]
    date_caption = f" · Tổng hợp {html.escape(date)}" if date else ""
    lines.append(f'<text x="32" y="794" class="note">Nguồn: summary.json · p95 chỉ tính ca qua invariant · Trường trọng yếu kiểm bằng rule{date_caption}</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def markdown_report(data: dict) -> str:
    total = sum(reports["pipeline"]["completed_cases"] for reports in data["directions"].values())
    failed = sum(reports["pipeline"]["failed_cases"] for reports in data["directions"].values())
    lines = ["# OneVoice — benchmark PC sau sửa lỗi", "", data["scope"], "",
             f"Đã hoàn tất {total:,} lượt streaming; còn {failed} lượt không qua gate chức năng. Hoàn tất phép đo không có nghĩa đã đạt nghiệm thu sản phẩm.", "",
             "![Pipeline PC](overview.svg)", ""]
    for direction, reports in data["directions"].items():
        pipeline, tts, profile = (reports[name] for name in ("pipeline", "tts", "profile"))
        lines.extend([
            f"## {'VI → EN' if direction == 'vi2en' else 'EN → VI'}", "",
            f"Runtime commit `{pipeline['git_revision']}`; policy `{pipeline['normal_commit_policy']}`; giọng `{pipeline['tts_engine']}` ({pipeline.get('sapi_voice') or 'giọng VI local'}).",
            f"Đã đo {pipeline['completed_cases']}/{pipeline['requested_cases']} lượt: {pipeline['passed_cases']} qua invariant chức năng, {pipeline['failed_cases']} lỗi. RAM process Python peak {pipeline['peak_process_rss_mb'] / 1024:.2f} GiB; không gồm process con (PowerShell SAPI/eSpeak).", "",
        ])
        for suite, row in pipeline["suites"].items():
            corpus = reports["corpus"][suite]
            lines.extend([
                f"### {suite} · {row['samples']} mẫu", "",
                f"- Corpus: {corpus['entries']} entry, {corpus['unique_audio_sha256']} hash WAV riêng, {corpus['unique_source_texts']} văn bản nguồn riêng. Entry không đồng nghĩa câu/người nói độc lập.",
                f"- Qua invariant: {row['passed']}/{row['samples']}; output trống: {row['empty_outputs']}.",
                f"- WER/CER hypothesis ASR endpoint đã chuẩn hóa: {percentage(row['asr_wer'])} / {percentage(row['asr_cer'])}.",
                f"- Bảo toàn trường trọng yếu theo reference: {percentage(row['reference_critical_field_preservation'])}.",
                f"- WER văn bản dịch so với reference: {percentage(row['translation_reference_wer'])}; exact match: {percentage(row['translation_exact_match'])}.",
                f"- Commit→audio p50/p95: {milliseconds(row['commit_to_first_audio_p50_ms'])} / {milliseconds(row['commit_to_first_audio_p95_ms'])}.", "",
            ])
            if suite == "approved_safety" and row["reference_critical_field_preservation"] == 1 and row["failed"]:
                lines.extend([f"Lưu ý: 100% qua rule reference nhưng {row['failed']} lượt không qua safety gate. Rule chưa bao phủ đủ nghĩa cảnh báo; con số 100% này không phải độ chính xác safety.", ""])
        lines.extend([
            f"TTS đúng ngôn ngữ đầu ra ({tts['target_language']}): {tts['passed_samples']}/{tts['samples']} câu; p50/p95 {milliseconds(tts['latency_p50_ms'])} / {milliseconds(tts['latency_p95_ms'])}.",
            f"P5 input cố định, {profile['repeats']} lượt mỗi route: gate budget {'PASS' if profile['passed'] else 'FAIL'}; blockers: {profile.get('blockers', [])}.", "",
        ])
        if pipeline["failures"]:
            lines.extend(["Các lỗi chức năng đầu tiên (danh sách đầy đủ trong summary.json):", ""])
            lines.extend(f"- `{row['case_id']}`: {row['error']}" for row in pipeline["failures"][:10])
            lines.append("")
        diagnostics = reports["diagnostics"]
        safety_ids = diagnostics["canonical_safety_id_agreement"]
        lines.extend([
            f"Lượt qua invariant nhưng không qua rule reference: {diagnostics['functional_pass_but_reference_invalid']}.",
            f"Nhóm lỗi chức năng: {diagnostics['functional_error_counts']}.",
            f"Số lần rule reference báo thiếu: {diagnostics['reference_error_occurrences']} (một câu có thể có nhiều lỗi).", "",
            f"Khớp nguyên safety ID theo reference: {safety_ids['exact_id_agreements']}/{safety_ids['samples']}. Đây là kiểm tra ID, không phải độ chính xác nghĩa; các biến thể tương đương cần review. Chi tiết nằm trong `diagnostics.safety_review` của summary.json.", "",
        ])
    lines.extend([
        "## Diễn giải", "",
        "Qua invariant nghĩa là có audio hợp lệ, đúng route, không rơi/đảo frame/commit và không có cảnh báo validator runtime theo ASR. Không đồng nghĩa nội dung dịch đúng hoàn toàn so với lời gốc.",
        "p50/p95 pipeline chỉ tính ca qua invariant chức năng. Ca đã tạo audio nhưng bị rule/route đánh FAIL không góp latency vào các percentile này; số lượng pass/fail được công bố riêng.",
        "Ca lỗi vẫn nằm trong mẫu số chất lượng. Nếu một ca lỗi đã có bản dịch/audio, phép đo giữ văn bản thực tế; chỉ ca không có output mới dùng chuỗi rỗng, không loại ca khó ra khỏi báo cáo.",
        "WER văn bản dịch đo khác biệt bề mặt với một reference. Cách diễn đạt khác có thể vẫn đúng nghĩa; không dùng 1 − WER làm độ chính xác dịch.",
        "Bảo toàn trường trọng yếu dùng validator thuật ngữ/entity/phủ định so với lời gốc. Đây là phép kiểm tự động, chưa thay thế người nghe hoặc đánh giá ngữ nghĩa.",
        "Chỉ số này là tỷ lệ câu qua toàn bộ rule phát hiện được, không phải critical-term recall theo số thuật ngữ của benchmark ASR. Câu không có trường được rule phát hiện vẫn có thể qua kiểm tra dù sai nghĩa; không so trực tiếp hai metric hoặc gọi chúng là độ chính xác dịch.",
        "Rule hiện tại có thể báo nhầm khi dùng từ tương đương hoặc viết số thành chữ, chẳng hạn material store thay material storage area, hoặc five hundred thay 500. Không diễn giải mọi cảnh báo thành lỗi model; ví dụ raw được giữ trong summary.json để review.",
        "Gate safety kiểm tra đường audio đã duyệt, chưa đủ chứng minh chọn đúng mệnh lệnh. summary.json bổ sung mức khớp safety ID và ví dụ để review; ID khác có thể là câu tương đương, không tự kết luận sai nghĩa.",
        "Latency ASR trong ledger là lần ASR tạo commit, không phải tổng mọi lần suy luận trên rolling window. Commit→audio và toàn lượt mới mô tả phần pipeline được đo.",
        "Load time là lần nạp model quan sát được trong mỗi process, không phải kiểm tra cold boot/cache trống. Peak RSS là của Python, không phải tổng RAM của máy hoặc toàn bộ process con.",
        "WER/CER ASR ở báo cáo này lấy hypothesis endpoint sau normalize/rolling assembly của runtime, không phải raw ASR benchmark đơn lẻ. Không dùng chênh lệch này để kết luận model/ONNX bị giảm độ chính xác.",
        "Full ở đây là full noisy test local và toàn bộ safety canonical. Full clean corpus, mic live, độ rõ của giọng, tai nghe Bluetooth và WAV công trường chưa được nghiệm thu bởi lượt này.", "",
        "Mỗi mẫu noisy là một WAV/biến thể nhiễu, không phải một người nói độc lập. VI và EN dùng hai manifest có kích thước/split riêng; không so tỷ lệ hai chiều như thể chúng được đo trên cùng một tập câu.", "",
    ])
    return "\n".join(lines)


def html_report(data: dict) -> str:
    failed = sum(reports["pipeline"].get("failed_cases", 0) for reports in data["directions"].values())
    blocks = []
    for direction, reports in data["directions"].items():
        pipeline, tts, profile = (reports[name] for name in ("pipeline", "tts", "profile"))
        rows = []
        for suite, row in pipeline["suites"].items():
            rows.append(f"<tr><td>{html.escape(suite)}</td><td>{row['samples']}</td><td>{row['passed']}/{row['samples']}</td><td>{percentage(row['asr_wer'])}</td><td>{percentage(row['reference_critical_field_preservation'])}</td><td>{milliseconds(row['commit_to_first_audio_p95_ms'])}</td></tr>")
        failures = ''.join(f"<li><code>{html.escape(row['case_id'])}</code> · {html.escape(str(row['error']))}</li>" for row in pipeline["failures"][:10])
        tts_failures = ''.join(f"<li><code>{html.escape(row['prompt_id'])}</code> · {html.escape(str(row['error']))}</li>" for row in tts.get("failures", []))
        safety = pipeline["suites"].get("approved_safety", {})
        safety_warning = ("<p>EN safety qua rule reference 100% nhưng còn lượt FAIL safety gate. Rule chưa bao phủ đủ nghĩa cảnh báo; 100% này không phải độ chính xác safety.</p>" if safety.get("reference_critical_field_preservation") == 1 and safety.get("failed") else "")
        blocks.append(f"<section><h2>{'VI → EN' if direction == 'vi2en' else 'EN → VI'}</h2><p>Commit <code>{pipeline['git_revision'][:12]}</code> · {html.escape(pipeline['tts_engine'])} · {html.escape(str(pipeline.get('sapi_voice') or 'giọng VI local'))} · policy {html.escape(pipeline['normal_commit_policy'])}</p><table><thead><tr><th>Corpus</th><th>Mẫu</th><th>Qua invariant</th><th>ASR WER</th><th>Trường trọng yếu</th><th>Commit→audio p95</th></tr></thead><tbody>{''.join(rows)}</tbody></table><p>TTS {tts['passed_samples']}/{tts['samples']} câu đúng ngôn ngữ ({tts['target_language']}) · p95 {milliseconds(tts['latency_p95_ms'])}. P5 input cố định: {'PASS' if profile['passed'] else 'FAIL'} budget desktop.</p>{'<details><summary>Lỗi chức năng đầu tiên</summary><ul>' + failures + '</ul></details>' if failures else ''}</section>")
        blocks.append(safety_warning)
        if tts_failures:
            blocks.append(f"<details><summary>Các lỗi TTS</summary><ul>{tts_failures}</ul></details>")
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OneVoice · Benchmark PC sau sửa lỗi</title><style>
body{{margin:0;background:#0d1117;color:#e5e7eb;font:16px/1.6 Segoe UI,Arial,sans-serif}}
main{{max-width:1120px;margin:auto;padding:36px 24px}}h1{{font-size:28px}}
h2{{font-size:23px;margin-bottom:6px}}p{{color:#b8c2cf}}section{{margin:38px 0}}
img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{padding:12px 10px;text-align:left;border-bottom:1px solid #30363d}}
th{{color:#9db0c2}}code{{font-size:13px}}a{{color:#7dd3c2}}.source{{font-size:13px}}
@media(max-width:700px){{table{{display:block;overflow:auto}}}}
</style></head><body><main><h1>OneVoice · PC offline sau sửa lỗi</h1>
<p>Full noisy test, safety canonical và TTS đúng ngôn ngữ đầu ra.
Đây là số đo của runtime đang dùng trên Windows.</p>
<p class="verdict">Đo đã hoàn tất; còn {failed} lượt không qua gate chức năng.
Đây chưa phải nghiệm thu chất lượng sản phẩm.</p>
<img src="overview.svg" alt="Tỷ lệ qua invariant, bảo toàn trường trọng yếu và latency pipeline PC">
{''.join(blocks)}<section><h2>Phạm vi đo</h2><p>{html.escape(data['scope'])}</p>
<p>Qua invariant kiểm tra audio/route/frame/commit và cảnh báo rule runtime theo ASR. Chất lượng theo lời gốc được ghi riêng,
gồm cả lỗi và output bị chặn. p95 chỉ tính các ca qua invariant chức năng;
không bao gồm driver/tai nghe. Giọng VI và mic live cần kiểm thử người nghe riêng.</p>
<p>WER ASR ở đây lấy hypothesis endpoint sau chuẩn hóa của pipeline,
không phải raw ASR benchmark. Trường trọng yếu được kiểm bằng rule, không phải độ chính xác ngữ nghĩa.</p>
<p class="source">Nguồn: <a href="summary.json">summary.json đầy đủ</a> ·
<a href="report.md">Markdown chi tiết</a> · <a href="review_findings.md">Lỗi còn lại và hướng xử lý</a> · Tạo lúc {data['generated_at']}</p>
</section></main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = read_complete_run(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in {
        "summary.json": json.dumps(data, ensure_ascii=False, indent=2),
        "overview.svg": overview_svg(data), "report.md": markdown_report(data),
        "report.html": html_report(data),
    }.items():
        (args.output_dir / name).write_text(text, encoding="utf-8")
    print(f"Report: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
