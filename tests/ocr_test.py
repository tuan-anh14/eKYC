"""
Test script cho OCR module
Sử dụng: python tests/ocr_test.py --image <đường_dẫn_ảnh_CCCD>
"""

import sys
import os
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

def test_ocr(image_path):
    """Test OCR trên ảnh CCCD"""
    print(f"Đang đọc ảnh: {image_path}")
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Lỗi: Không thể đọc ảnh từ {image_path}")
        print("   Vui lòng kiểm tra đường dẫn ảnh có đúng không.")
        return
    
    print(f"✅ Đã đọc ảnh thành công. Kích thước: {img.shape}")
    print("\n" + "="*50)
    print("Đang trích xuất thông tin từ ảnh CCCD...")
    print("="*50 + "\n")
    
    try:
        # Trích xuất thông tin
        data = extract_id_fields(img)
        
        # Validate kết quả
        is_valid, error_msg = validate_ocr_result(data)
        
        # Hiển thị kết quả
        print("\n📋 KẾT QUẢ TRÍCH XUẤT:")
        print("-" * 50)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("-" * 50)
        
        if not is_valid:
            print(f"\n⚠️  CẢNH BÁO: {error_msg}")
            print("   Vui lòng chụp lại ảnh rõ ràng, thẳng góc.")
        else:
            print("\n✅ Kết quả OCR hợp lệ!")
            
    except Exception as e:
        print(f"\n❌ Lỗi khi chạy OCR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test OCR module để trích xuất thông tin từ ảnh CCCD"
    )
    parser.add_argument(
        "--image", 
        required=True, 
        help="Đường dẫn đến ảnh CCCD (ví dụ: path/to/cccd.jpg)"
    )
    
    args = parser.parse_args()
    test_ocr(args.image)
