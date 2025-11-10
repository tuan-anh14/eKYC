# OCR module for ID card processing
from .ocr_infer import extract_id_fields, validate_ocr_result, DEFAULT_ID_FIELDS

__all__ = ['extract_id_fields', 'validate_ocr_result', 'DEFAULT_ID_FIELDS']
