from web3 import Web3
import json
import os
from dotenv import load_dotenv

# Load biến môi trường từ file .env
load_dotenv()

class BlockchainConnector:
    def __init__(self):
        # --- CẤU HÌNH TỪ .ENV ---
        self.GANACHE_URL = os.getenv("GANACHE_URL", "http://127.0.0.1:7545")
        self.PRIVATE_KEY = os.getenv("PRIVATE_KEY")
        self.CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
        self.MY_ADDRESS = os.getenv("MY_ADDRESS")

        if not self.PRIVATE_KEY or not self.CONTRACT_ADDRESS:
            print("CẢNH BÁO: Chưa cấu hình file .env đầy đủ!")

        self.web3 = Web3(Web3.HTTPProvider(self.GANACHE_URL))
        
        # ABI tối thiểu của hàm addCustomer trong dEkycStorage.sol
        # Nếu bạn sửa Smart Contract, hãy cập nhật ABI này (Lấy từ Remix -> Compilation Details -> ABI)
        self.contract_abi = [
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "_idNumber",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "_fullName",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "_dateOfBirth",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "_homeTown",
				"type": "string"
			}
		],
		"name": "addCustomer",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	}
]
        
        if self.web3.is_connected():
            print("Kết nối Blockchain thành công!")
            self.contract = self.web3.eth.contract(address=self.CONTRACT_ADDRESS, abi=self.contract_abi)
        else:
            print("Lỗi: Không thể kết nối tới Ganache.")
            self.contract = None

    def save_to_blockchain(self, id_number, full_name, dob, hometown):
        if not self.contract:
            return "Lỗi: Chưa kết nối Blockchain"

        try:
            # Xây dựng giao dịch
            txn = self.contract.functions.addCustomer(
                id_number, full_name, dob, hometown
            ).build_transaction({
                'chainId': 1337, # Ganache thường dùng ID 1337 hoặc 5777
                'gas': 3000000,
                'gasPrice': self.web3.to_wei('20', 'gwei'),
                'nonce': self.web3.eth.get_transaction_count(self.MY_ADDRESS),
            })

            # Ký giao dịch
            signed_txn = self.web3.eth.account.sign_transaction(txn, private_key=self.PRIVATE_KEY)

            # Gửi giao dịch
            # FIX: Một số bản Web3 trả về thuộc tính rawTransaction, một số bản cũ/mới trả về raw_transaction
            raw_tx = getattr(signed_txn, 'rawTransaction', None)
            if raw_tx is None:
                 raw_tx = getattr(signed_txn, 'raw_transaction', None)
            
            if raw_tx is None:
                # Fallback cuối cùng: coi nó như dictionary hoặc tuple
                 raw_tx = signed_txn[0] if isinstance(signed_txn, (list, tuple)) else signed_txn

            tx_hash = self.web3.eth.send_raw_transaction(raw_tx)
            
            # Chờ xác nhận
            tx_receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return f"Thành công! Hash: {self.web3.to_hex(tx_hash)}"
        except Exception as e:
            return f"Lỗi Gửi Giao Dịch: {str(e)}"
