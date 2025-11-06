# eKYC 

## Update
- 7/12/2024: The source code has been rewritten and tested to be compatible with the VGGFace2 model (InceptionResnetV1). However, I have only tested it using Norm L2 for face matching.

----------------------------

eKYC (Electronic Know Your Customer) is a project designed to electronically verify the identity of customers. This is an essential system to ensure authenticity and security in online transactions.

![](resources/ekyc.jpg)

eKYC (Electronic Know Your Customer) is an electronic customer identification and verification solution that enables banks to identify customers 100% online, relying on biometric information and artificial intelligence (AI) for customer recognition, without the need for face-to-face interactions as in the current process.

## eKYC flow 
This README provides an overview of the eKYC (Electronic Know Your Customer) flow, which comprises three main components: Upload Document (ID Card), Face Recognition (Verification), and Liveness Detection.

![](resources/flow.jpg)

#### 1. Upload Document (ID Card) & OCR Extraction

Initially, users are required to upload an image of their ID card. The system will automatically extract information from the ID card using OCR (Optical Character Recognition) technology:

- **OCR Technology**: The system uses PaddleOCR to extract text from Vietnamese ID cards (CCCD)
- **Extracted Fields**: 
  - Số CCCD (ID Number)
  - Họ và tên (Full Name)
  - Ngày sinh (Date of Birth)
  - Giới tính (Gender)
  - Quốc tịch (Nationality)
  - Địa chỉ (Address)
  - Ngày cấp (Issue Date)
  - Nơi cấp (Issuing Authority)
- **Manual Editing**: Users can manually edit any OCR results if there are recognition errors
- **Validation**: The system validates that essential fields (ID number and date of birth) are present before proceeding
- **Data Export**: OCR results are automatically saved to JSON files in the `results/` directory

#### 2. Face Verification

Following the document upload, we proceed to verify whether the user matches the individual pictured on the ID card. Here's how we do it:

- **Step 1 - Still Face Capture**: Users are prompted to maintain a steady face in front of the camera.

- **Step 2 - Face Matching (Face Verification)**: Our system utilizes advanced facial recognition technology to compare the live image of the user's face with the photo on the ID card.

#### 3. Liveness Detection

To ensure the user's physical presence during the eKYC process and to prevent the use of static images or videos, we implement Liveness Detection. This step involves the following challenges to validate the user's authenticity:

- **Step 3 - Liveness Challenges**: Users are required to perform specific actions or challenges, which may include blinking, smiling, or turning their head.

- **Step 4 - Successful Liveness Verification**: Successful completion of the liveness challenges indicates the user's authenticity, confirming a successful eKYC process.

These combined steps—ID card upload, Face Verification, and Liveness Detection—comprehensively verify the user's identity, enhancing security and reducing the risk of fraudulent attempts.

## Installation
1. Clone the repository
```bash
git clone https://github.com/manhcuong02/eKYC
cd eKYC
```
2. Install the required dependencies
```bash
# Nâng cấp pip và wheel trước
python -m pip install --upgrade pip wheel

# Cài đặt dependencies
pip install -r requirements.txt
```

**Note**: The OCR functionality requires PaddleOCR and PaddlePaddle. On Windows (AMD64), PaddlePaddle 2.5.2 will be automatically installed. For other platforms, you may need to adjust the requirements.

## Usage

### 1. Download Model Weights
Download weights of the [pretrained VGGFace models](https://drive.google.com/drive/folders/1-pEMok04-UqpeCi_yscUcIA6ytvxhvkG?usp=drive_link) from ggdrive, and then add them to the 'verification_models/weights' directory. Download weights and landmarks of the [pretrained liveness detection models](https://drive.google.com/drive/folders/1S6zLU8_Cgode7B7mfJWs9oforfAODaGB?usp=drive_link) from ggdrive, and then add them to the 'liveness_detection/landmarks' directory.

### 2. Run eKYC Application (GUI)
Chạy ứng dụng eKYC với giao diện PyQt5:
```bash
python main.py
```

**Hướng dẫn sử dụng OCR trong GUI:**
1. Trên trang đầu tiên, nhấn nút **"Chọn ảnh CCCD"** để chọn ảnh thẻ căn cước
2. Nhấn nút **"Trích xuất thông tin"** để chạy OCR và tự động trích xuất thông tin từ ảnh
3. Kiểm tra và chỉnh sửa các thông tin được trích xuất nếu cần (các trường có thể chỉnh sửa: Số CCCD, Họ tên, Ngày sinh, Giới tính, Quốc tịch, Địa chỉ, Ngày cấp, Nơi cấp)
4. Nhấn **"Next"** để tiếp tục sang bước xác thực khuôn mặt
5. Kết quả OCR sẽ tự động được lưu vào thư mục `results/` dưới dạng file JSON

### 3. Test OCR Module (Command Line)
Test OCR module độc lập từ command line:
```bash
# Cách 1: Sử dụng script test
python tests/ocr_test.py --image <đường_dẫn_ảnh_CCCD>

# Cách 2: Sử dụng module trực tiếp
python -m ocr.ocr_infer --image <đường_dẫn_ảnh_CCCD>
```

**Ví dụ:**
```bash
python tests/ocr_test.py --image path/to/cccd_image.jpg
```

Kết quả sẽ hiển thị các thông tin được trích xuất dưới dạng JSON.

## Results

> [!Note]
> Due to concerns about my personal information, I have deleted the video result from my repo
