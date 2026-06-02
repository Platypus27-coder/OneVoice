import sys
import os

sys.path.insert(0, os.path.join(os.path.abspath('src')))

# 1. Test Text Normalization
from utils.text_normalizer import normalize
text = 'CẬU ĐÃ LÀM GÌ VỚI NÓ VẬY THÊM NĂNG LƯỢNG HẢ NÓ HOẠT ĐỘNG NHƯ THẾ NÀO VẬY CHO MÌNH MƯỢN CHÚT ĐỪNG CÓ KEO KIỆT VẬY CHỨ HÔM NAY LỚP MÌNH CÓ BÀI KIỂM TRA MÔN THỂ DỤC NÊN MÌNH RẤT LÀ CẦN NÓ LUÔN XÀI XONG MÌNH TRẢ LẠI LIỀN'
normalized = normalize(text, lang='vi')
print('=== NORMALIZED TEXT ===')
print(normalized)

# 2. Test Translation
import yaml
from translation.mt_engine import Translator

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

mt = Translator(cfg)
mt.load()

print('=== TRANSLATION ===')
result = mt.translate(text, direction='vi2en')
print(result)
