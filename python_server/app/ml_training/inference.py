import os
import torch

from python_server.app.config.category_config import CategoryConfig
from python_server.app.ml_training.train_gru import BiGRUTextClassifier


MAX_LEN = 20


def encode_chars(text, char_to_idx):
    if not text or str(text).lower() == "nan":
        text = "<EMPTY>"

    ids = [char_to_idx.get(ch, 0) for ch in text[:MAX_LEN]]
    while len(ids) < MAX_LEN:
        ids.append(0)

    return torch.tensor([ids], dtype=torch.long)  # batch size = 1


def predict(price, merchant, memo=""):

    # 1️⃣ 룰 기반 먼저 처리
    rule = CategoryConfig.rule_based(merchant, memo)
    if rule is not None:
        return rule

    # 2️⃣ 모델 파일 불러오기
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "char_gru_classifier.pth")
    ENC_PATH = os.path.join(BASE_DIR, "model", "char_gru_encoders.pth")

    enc = torch.load(ENC_PATH, map_location="cpu", weights_only=False)
    char_to_idx = enc["char_to_idx"]
    category_encoder = enc["category_encoder"]

    # 텐서 변환
    price_tensor = torch.tensor([[price / 100000]], dtype=torch.float32)
    merchant_tensor = encode_chars(merchant, char_to_idx)
    memo_tensor = encode_chars(memo, char_to_idx)

    # 모델 준비
    vocab_size = len(char_to_idx)
    num_classes = len(category_encoder.classes_)

    model = BiGRUTextClassifier(vocab_size, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 예측
    with torch.no_grad():
        output = model(price_tensor, merchant_tensor, memo_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = category_encoder.inverse_transform([pred_idx])[0]

    return pred_label


if __name__ == "__main__":
    print(predict(9000, "을지서적"))   # → 교육
    print(predict(5500, "카카오T"))    # → 교통
    print(predict(4800, "YES24", "도서"))  # → 교육
    print("룰 기반 결과 =", CategoryConfig.rule_based("영동서적", ""))
