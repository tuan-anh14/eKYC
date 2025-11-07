from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, QThread, QObject, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout,
                             QLabel, QMainWindow, QPushButton, QStackedWidget,
                             QVBoxLayout, QWidget, QLineEdit, QTextEdit, QMessageBox)

import cv2 as cv
import json
import time
import os
from .utils import *
from ocr.ocr_infer import extract_id_fields, validate_ocr_result


class IDCardPhoto(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.window_heigt = 800
        self.window_width = 1600
                
        # Thiết lập tiêu đề và kích thước của cửa sổ
        self.setWindowTitle('Choose ID Card Photo')
        self.setGeometry(100, 100, self.window_width, self.window_heigt)
        self.setFixedSize(self.window_width, self.window_heigt)
        
        self.font = QFont()
        self.font.setPointSize(13)
        self.font.setFamily("Times New Roman")

        self.label = QLabel(self)
        self.label.setText('Vui lòng chọn ảnh mặt trước của thẻ căn cước công dân.')
        self.label.move(550, 50)
        self.label.setFont(self.font)
        
        self.exit_button = add_button(self, "Exit", 800, 750, 150, 50, exit)
    
        self.select_image_button = add_button(self, "Chọn ảnh CCCD", 320, 750, 150, 50, self.selectImage)
        self.extract_button = add_button(self, "Trích xuất thông tin", 500, 750, 180, 50, self.extractInfo, disabled=True)
        self.next = add_button(self, "Next", 1280, 750, 150, 50, self.switch_page, disabled = True)
        self.in_image = QLabel(self)
        
        self.img_path = None
        self.ocr_fields = {}
        
        # Tạo các text fields để hiển thị và chỉnh sửa kết quả OCR
        self._create_ocr_fields()
        
        # Label thông báo lỗi
        self.error_label = QLabel(self)
        self.error_label.setFont(self.font)
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: red;")
        self.error_label.hide()
    
    def switch_page(self):
        # Validate trước khi chuyển trang
        fields = self._get_ocr_fields_from_ui()
        if not fields.get('id_number') or not fields.get('dob'):
            QMessageBox.warning(
                self, 
                "Thông tin không đầy đủ",
                "Vui lòng nhập đầy đủ số CCCD và ngày sinh trước khi tiếp tục."
            )
            return
        
        # Lưu kết quả OCR ra JSON
        saved_path = self._save_ocr_result()
        if saved_path:
            print(f"Đã lưu kết quả OCR vào: {saved_path}")
        
        # Lưu fields vào main_window để có thể dùng ở các trang khác
        self.main_window.ocr_fields = fields
        
        self.main_window.switch_page(1)     
        
    def rescale_image(self, width, height):
        return int(width * 400 / height), 400

    def selectImage(self):
        # Hiển thị hộp thoại chọn tệp ảnh và lấy tên tệp ảnh được chọn
        file_name, _ = QFileDialog.getOpenFileName(self, 'Chọn ảnh CCCD', '', 'Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)')
        
        if file_name:
            self.img_path = file_name
            # Tải ảnh từ tệp và hiển thị nó trên QLabel
            pixmap = QPixmap(file_name)
            img = cv.imread(file_name)
            width, height = self.rescale_image(img.shape[1], img.shape[0])
            # Đặt ảnh bên trái, cách lề trái 50px
            self.in_image.setGeometry(QRect(50, 150, width, height))
            self.in_image.setPixmap(pixmap.scaled(width, height))  
            self.in_image.show()
            
            # Reset các field OCR
            for field_data in self.ocr_text_fields.values():
                field_data['label'].hide()
                field_data['field'].hide()
            self.error_label.hide()
            self.ocr_fields = {}
            
            # Enable nút extract
            self.extract_button.setDisabled(False)
            # Disable nút Next cho đến khi có OCR result hợp lệ
            self.next.setDisabled(True)

    def _create_ocr_fields(self):
        """Tạo các text field để hiển thị và chỉnh sửa kết quả OCR"""
        # Đặt các trường OCR bên phải, bắt đầu từ x=900 để tránh đè lên ảnh
        field_labels = {
            'id_number': ('Số CCCD:', 900, 150),
            'name': ('Họ và tên:', 900, 200),
            'dob': ('Ngày sinh:', 900, 250),
            'gender': ('Giới tính:', 900, 300),
            'nationality': ('Quốc tịch:', 900, 350),
            'place_of_origin': ('Quê quán:', 900, 400),
            'residence': ('Nơi thường trú:', 900, 450),
            'expiry_date': ('Có giá trị đến:', 900, 550),
        }
        
        self.ocr_text_fields = {}
        small_font = QFont()
        small_font.setPointSize(10)
        small_font.setFamily("Times New Roman")
        
        for field_name, (label_text, x, y) in field_labels.items():
            # Label
            label = QLabel(self)
            label.setText(label_text)
            label.move(x, y)
            label.setFont(small_font)
            label.hide()
            
            # Text field
            if field_name in ['residence']:
                # Địa chỉ dùng QTextEdit vì có thể dài
                text_field = QTextEdit(self)
                text_field.setGeometry(QRect(x + 120, y, 400, 80))
            else:
                text_field = QLineEdit(self)
                text_field.setGeometry(QRect(x + 120, y, 400, 30))
            
            text_field.setFont(small_font)
            text_field.hide()
            
            self.ocr_text_fields[field_name] = {
                'label': label,
                'field': text_field
            }
    
    def extractInfo(self):
        """Chạy OCR trên background thread để UI không đơ"""
        if not self.img_path:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn ảnh CCCD trước.")
            return

        class OCRWorker(QObject):
            finished = pyqtSignal(dict, tuple)
            failed = pyqtSignal(str)
            def __init__(self, path):
                super().__init__()
                self.path = path
            def run(self):
                try:
                    img = cv.imread(self.path)
                    if img is None:
                        raise ValueError("Không thể đọc ảnh")
                    fields = extract_id_fields(img)
                    valid = validate_ocr_result(fields)
                    self.finished.emit(fields, valid)
                except Exception as e:
                    self.failed.emit(str(e))

        # Thông báo đang xử lý
        self.error_label.setText("Đang trích xuất thông tin từ ảnh...")
        self.error_label.setStyleSheet("color: blue;")
        self.error_label.setGeometry(QRect(900, 100, 500, 30))
        self.error_label.show()

        # Disable buttons trong lúc xử lý
        self.extract_button.setDisabled(True)
        self.next.setDisabled(True)

        # Khởi tạo và chạy thread
        self._ocr_thread = QThread(self)
        self._ocr_worker = OCRWorker(self.img_path)
        self._ocr_worker.moveToThread(self._ocr_thread)
        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.failed.connect(self._on_ocr_failed)
        self._ocr_worker.finished.connect(self._ocr_thread.quit)
        self._ocr_worker.failed.connect(self._ocr_thread.quit)
        self._ocr_thread.start()

    def _on_ocr_finished(self, fields, valid_tuple):
        self.ocr_fields = fields
        is_valid, error_msg = valid_tuple
        self.error_label.hide()
        self._display_ocr_fields()
        if not is_valid:
            self.error_label.setText(error_msg)
            self.error_label.setStyleSheet("color: red;")
            self.error_label.setGeometry(QRect(900, 100, 500, 50))
            self.error_label.show()
        else:
            self.next.setDisabled(False)
        self.extract_button.setDisabled(False)

    def _on_ocr_failed(self, msg):
        QMessageBox.critical(self, "Lỗi", f"Lỗi khi trích xuất thông tin: {msg}")
        self.error_label.hide()
        self.extract_button.setDisabled(False)
    
    def _display_ocr_fields(self):
        """Hiển thị các trường OCR trên UI"""
        field_mapping = {
            'id_number': 'id_number',
            'name': 'name',
            'dob': 'dob',
            'gender': 'gender',
            'nationality': 'nationality',
            'place_of_origin': 'place_of_origin',
            'residence': 'residence',
            'expiry_date': 'expiry_date',
        }
        
        for ui_field, data_field in field_mapping.items():
            if ui_field in self.ocr_text_fields:
                label = self.ocr_text_fields[ui_field]['label']
                field = self.ocr_text_fields[ui_field]['field']
                
                value = self.ocr_fields.get(data_field, '')
                if isinstance(field, QTextEdit):
                    field.setPlainText(value)
                else:
                    field.setText(value)
                
                label.show()
                field.show()
    
    def _get_ocr_fields_from_ui(self) -> dict:
        """Lấy giá trị từ các text field trên UI"""
        fields = {}
        field_mapping = {
            'id_number': 'id_number',
            'name': 'name',
            'dob': 'dob',
            'gender': 'gender',
            'nationality': 'nationality',
            'place_of_origin': 'place_of_origin',
            'residence': 'residence',
            'expiry_date': 'expiry_date',
        }
        
        for ui_field, data_field in field_mapping.items():
            if ui_field in self.ocr_text_fields:
                field = self.ocr_text_fields[ui_field]['field']
                if isinstance(field, QTextEdit):
                    value = field.toPlainText().strip()
                else:
                    value = field.text().strip()
                if value:
                    fields[data_field] = value
        
        return fields
    
    def _save_ocr_result(self):
        """Lưu kết quả OCR ra file JSON"""
        fields = self._get_ocr_fields_from_ui()
        if not fields:
            return
        
        # Tạo tên file với timestamp
        timestamp = int(time.time())
        filename = f"ekyc_{timestamp}.json"
        
        # Lưu vào thư mục hiện tại hoặc thư mục results nếu có
        save_dir = "results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filepath = os.path.join(save_dir, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(fields, f, ensure_ascii=False, indent=2)
            return filepath
        except Exception as e:
            print(f"Lỗi khi lưu file JSON: {str(e)}")
            return None
    
    def clear_window(self):
        self.in_image.hide()
        # Ẩn các field OCR
        for field_data in self.ocr_text_fields.values():
            field_data['label'].hide()
            field_data['field'].hide()
        self.error_label.hide()
        self.ocr_fields = {}
        self.img_path = None