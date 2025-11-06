"""
Test nhanh OCR với ảnh có sẵn: tests/test1.jpg
Chạy: python tests/test_test1.py
"""

import os
import sys
import json

# Fix encoding cho Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Thêm thư mục gốc vào path để import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from ocr.ocr_infer import extract_id_fields, validate_ocr_result


def main():
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test2.jpg')
    print(f"Đang đọc ảnh: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Lỗi: Không thể đọc ảnh test1.jpg. Hãy kiểm tra đường dẫn hoặc định dạng file.")
        sys.exit(1)

    print(f"✅ Đã đọc ảnh thành công. Kích thước: {img.shape}")
    print("\n" + "=" * 50)
    print("Đang trích xuất thông tin từ ảnh CCCD...")
    print("=" * 50 + "\n")

    data = extract_id_fields(img)
    is_valid, error_msg = validate_ocr_result(data)

    print("\n📋 KẾT QUẢ TRÍCH XUẤT:")
    print("-" * 50)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    print("-" * 50)

    if not is_valid:
        print(f"\n⚠️  CẢNH BÁO: {error_msg}")
    else:
        print("\n✅ Kết quả OCR hợp lệ!")


if __name__ == "__main__":
    main()


