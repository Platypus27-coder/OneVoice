"""
Synthetic Data Generator for OneVoice Edge
==========================================
Tự động sinh 10.000+ câu đối song Tiếng Việt - Tiếng Anh
dựa trên thuật toán Tổ hợp ngẫu nhiên các thực thể công trường.
Dữ liệu sinh ra được dùng để fine-tune mô hình Seq2Seq (envit5)
trong trường hợp chưa có dataset chính thức.
"""

import os
import sys
import csv
import random

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Định nghĩa các tập hợp thực thể (Entities)
SUBJECTS = [
    ("cai thầu", "foreman"),
    ("đội trưởng", "crew chief"),
    ("kỹ sư", "engineer"),
    ("giám sát", "supervisor"),
    ("anh thợ máy", "mechanic"),
    ("tổ đội", "the crew"),
    ("chỉ huy trưởng", "site manager"),
    ("công nhân hàn", "welder")
]

MACHINES = [
    ("máy xúc số ba", "excavator number three"),
    ("xe cẩu", "the crane truck"),
    ("máy trộn bê tông", "the concrete mixer"),
    ("xe ủi", "the bulldozer"),
    ("cẩu tháp", "the tower crane"),
    ("bơm chìm", "the submersible pump"),
    ("máy hàn", "the welding machine"),
    ("máy rải nhựa", "the asphalt paver"),
    ("xe lu", "the road roller"),
    ("máy đóng cọc", "the pile driver")
]

ISSUES = [
    ("bị xì nhớt thủy lực", "is leaking hydraulic fluid"),
    ("đang kêu to lắm", "is making a loud noise"),
    ("bị kẹt cứng rồi", "is jammed"),
    ("vừa bốc khói", "just emitted smoke"),
    ("bị rò điện", "has an electrical leakage"),
    ("hư motor rồi", "has motor failure"),
    ("bị tuột sên", "slipped its chain"),
    ("bị nổ lốp", "has a tire blowout"),
    ("không khởi động được", "won't start")
]

ACTIONS = [
    ("dừng máy ngay", "stop the machine immediately"),
    ("cúp điện khẩn cấp", "cut the power urgently"),
    ("kiểm tra lẹ giùm mình", "please check it immediately"),
    ("gọi thợ sửa đi", "call the repairman"),
    ("tránh xa ra", "stay away"),
    ("báo cáo giám sát", "report to the supervisor"),
    ("đổi máy khác", "switch to another machine"),
    ("chụp hình lại", "take a picture"),
    ("lập biên bản", "make a report"),
    ("gọi đội bảo trì", "call the maintenance team")
]

URGENT_PHRASES = [
    ("Ê bạn ơi,", "Hey buddy,"),
    ("Khẩn cấp,", "Urgent,"),
    ("Trời ơi,", "Oh my god,"),
    ("Anh em ơi,", "Guys,"),
    ("Nhanh lên,", "Hurry up,"),
    ("Cảnh báo,", "Warning,"),
    ("Lưu ý,", "Attention,")
]

CONSTRUCTION_TASKS = [
    ("đổ bê tông", "pouring concrete"),
    ("kéo cáp", "cable pulling"),
    ("cắt thép", "cutting steel"),
    ("buộc thép", "rebar tying"),
    ("đóng cốp pha", "formwork installation"),
    ("lát gạch", "tiling"),
    ("trát tường", "plastering")
]

TEMPLATES = [
    # Template 1: Sự cố máy móc khẩn cấp
    ("{urgent} {machine} {issue}, {action}!", 
     "{urgent_en} {machine_en} {issue_en}, {action_en}!"),
    
    # Template 2: Báo cáo kỹ thuật
    ("Báo cáo {subject}, {machine} {issue}.", 
     "Report to {subject_en}, {machine_en} {issue_en}."),
    
    # Template 3: Yêu cầu thực hiện công việc
    ("Hôm nay tổ chúng ta sẽ làm phần {task}.", 
     "Today our crew will do the {task_en}."),
    
    # Template 4: Cảnh báo an toàn
    ("Cẩn thận lúc {task}, coi chừng nguy hiểm.", 
     "Be careful during {task_en}, watch out for danger."),
     
    # Template 5: Giao tiếp thường ngày
    ("Cho mình mượn {machine} được không?",
     "Can I borrow {machine_en}?")
]


def generate_dataset(num_samples=10000, output_path="data/synthetic_dataset_10k.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    generated_set = set()
    data = []
    
    print(f"Bắt đầu sinh dữ liệu...")
    
    iterations = 0
    max_iterations = num_samples * 10
    
    # Sinh dữ liệu ngẫu nhiên
    while len(data) < num_samples and iterations < max_iterations:
        iterations += 1
        template_vi, template_en = random.choice(TEMPLATES)
        
        # Randomize entities
        subject_vi, subject_en = random.choice(SUBJECTS)
        machine_vi, machine_en = random.choice(MACHINES)
        issue_vi, issue_en = random.choice(ISSUES)
        action_vi, action_en = random.choice(ACTIONS)
        urgent_vi, urgent_en = random.choice(URGENT_PHRASES)
        task_vi, task_en = random.choice(CONSTRUCTION_TASKS)
        
        # Populate template
        vi_sent = template_vi.format(
            subject=subject_vi, machine=machine_vi, issue=issue_vi, 
            action=action_vi, urgent=urgent_vi, task=task_vi
        )
        en_sent = template_en.format(
            subject_en=subject_en, machine_en=machine_en, issue_en=issue_en, 
            action_en=action_en, urgent_en=urgent_en, task_en=task_en
        )
        
        # Normalize Capitalization
        vi_sent = vi_sent.capitalize()
        en_sent = en_sent.capitalize()
        
        # Chống trùng lặp tuyệt đối
        if vi_sent not in generated_set:
            generated_set.add(vi_sent)
            data.append({"vi": vi_sent, "en": en_sent})
            
    # Ghi ra file CSV chuẩn HuggingFace
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vi", "en"])
        writer.writeheader()
        writer.writerows(data)
        
    print(f"✅ Đã lưu {len(data)} câu hội thoại vào: {output_path}")

if __name__ == "__main__":
    generate_dataset(10000)
