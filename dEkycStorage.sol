// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title dEkycStorage
 * @dev Hệ thống lưu trữ định danh phi tập trung (Decentralized Identity Storage)
 * Tính năng nâng cao: Phân quyền (RBAC), Truy vết (Audit Trail), Quản lý Ngân hàng, CHỐNG GIAN LẬN (Anti-Fraud).
 */
contract dEkycStorage {
    
    // Enum trạng thái xác thực để quản lý vòng đời định danh
    enum IdentityStatus {
        NOT_FOUND,  // Chưa có trên hệ thống
        VERIFIED,   // Đã xác thực thành công (Hợp lệ)
        REVOKED     // Đã bị thu hồi/Hủy (Do gian lận hoặc hết hạn)
    }

    // Admin là Ngân hàng Trung ương hoặc Tổ chức quản lý
    address public admin;

    struct Identity {
        string fullName;     // Họ tên
        string idNumber;     // Số CCCD
        string dateOfBirth;  // Ngày sinh
        string homeTown;     // Quê quán
        IdentityStatus status; // Trạng thái (Thay vì bool true/false đơn giản)
        uint256 timestamp;   // Thời gian xác thực
        address verifier;    // Địa chỉ ví của Ngân hàng đã xác thực
    }

    // Mapping: Số CCCD -> Thông tin định danh
    mapping(string => Identity) public identities;
    
    // Danh sách các Ngân hàng được cấp phép (Allowed Banks)
    mapping(address => bool) public approvedBanks;

    // Sự kiện lưu log (giúp tra cứu lịch sử trên blockchain explorer)
    event BankAdded(address indexed bankAddress);
    event BankRemoved(address indexed bankAddress);
    event IdentityVerified(string indexed idNumber, string fullName, address indexed verifier);
    event IdentityRevoked(string indexed idNumber, address indexed revoker, string reason);

    // Modifier: Chỉ Admin mới được gọi hàm
    modifier onlyAdmin() {
        require(msg.sender == admin, "Chi Admin moi co quyen nay");
        _;
    }

    // Modifier: Chỉ Ngân hàng được cấp phép mới được gọi hàm
    modifier onlyBank() {
        require(approvedBanks[msg.sender], "Ngan hang cua ban chua duoc cap phep tham gia he thong");
        _;
    }

    constructor() {
        // Người deploy contract sẽ là Admin
        admin = msg.sender;
        // Tự cấp quyền cho chính mình để test (Vừa là Admin vừa là Bank A)
        approvedBanks[msg.sender] = true; 
    }

    // --- QUẢN TRỊ HỆ THỐNG (ADMIN) ---

    // Admin cấp phép cho một Ngân hàng mới tham gia mạng lưới
    function registerBank(address _bankAddress) public onlyAdmin {
        approvedBanks[_bankAddress] = true;
        emit BankAdded(_bankAddress);
    }

    // Admin thu hồi quyền của một Ngân hàng (nếu phát hiện gian lận)
    function removeBank(address _bankAddress) public onlyAdmin {
        approvedBanks[_bankAddress] = false;
        emit BankRemoved(_bankAddress);
    }

    // --- NGHIỆP VỤ NGÂN HÀNG (BANK) ---

    // 1. Xác thực mới: Ngân hàng đẩy dữ liệu khách hàng ok lên Blockchain
    function addCustomer(
        string memory _idNumber, 
        string memory _fullName, 
        string memory _dateOfBirth, 
        string memory _homeTown
    ) public onlyBank {
        require(bytes(_idNumber).length > 0, "So CCCD khong hop le");
        
        // Anti-Overwrite: Không cho phép ghi đè nếu khách hàng đã tồn tại và chưa bị Revoke
        if (identities[_idNumber].status == IdentityStatus.VERIFIED) {
            revert("Khach hang nay da ton tai tren he thong");
        }

        // Lưu dữ liệu
        identities[_idNumber] = Identity({
            fullName: _fullName,
            idNumber: _idNumber,
            dateOfBirth: _dateOfBirth,
            homeTown: _homeTown,
            status: IdentityStatus.VERIFIED, // Mặc định là VERIFIED
            timestamp: block.timestamp,
            verifier: msg.sender
        });

        emit IdentityVerified(_idNumber, _fullName, msg.sender);
    }

    // 2. Chống gian lận: Ngân hàng báo cáo hủy xác thực (Tính năng Nâng cao)
    function revokeCustomer(string memory _idNumber, string memory _reason) public onlyBank {
        require(identities[_idNumber].status != IdentityStatus.NOT_FOUND, "Khach hang chua ton tai");
        
        // Cập nhật trạng thái sang REVOKED
        identities[_idNumber].status = IdentityStatus.REVOKED;
        
        // Cập nhật người báo cáo (người hủy) là người ghi đè cuối cùng
        identities[_idNumber].verifier = msg.sender; 
        identities[_idNumber].timestamp = block.timestamp;

        emit IdentityRevoked(_idNumber, msg.sender, _reason);
    }

    // --- TRA CỨU CÔNG KHAI (PUBLIC) ---

    // Bất kỳ ai cũng có thể kiểm tra trạng thái
    function checkCustomer(string memory _idNumber) public view returns (
        string memory fullName,
        string memory dateOfBirth,
        string memory homeTown,
        string memory statusString, // Trả về text cho dễ đọc trên giao diện
        uint256 verifiedAt,
        address verifiedBy
    ) {
        Identity memory i = identities[_idNumber];
        
        require(i.status != IdentityStatus.NOT_FOUND, "Khach hang nay chua co du lieu tren Blockchain");

        string memory stt = "UNKNOWN";
        if (i.status == IdentityStatus.VERIFIED) {
            stt = "VERIFIED (HOP LE)";
        } else if (i.status == IdentityStatus.REVOKED) {
            stt = "REVOKED (DA BI HUY/GIAN LAN)";
        }

        return (
            i.fullName,
            i.dateOfBirth,
            i.homeTown,
            stt,
            i.timestamp,
            i.verifier
        );
    }
}
