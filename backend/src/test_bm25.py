from bm25.bm25 import Bm25
from bm25.sparse_text_embedding import SparseTextEmbedding
import re
from bm25.sparse_embedding_base import (
    SparseEmbedding,)


bm25 = Bm25("davicn81/bm25")
docs = ["Điều 1. Phạm vi điều chỉnh Chương I QUY ĐỊNH CHUNG 36/2024/TT-BGTVT. Thông tư này quy định về tổ chức, quản lý hoạt động vận tải hành khách, hàng hóa bằng xe ô tô và hoạt động của bến xe, bãi đỗ xe, trạm dừng nghỉ, điểm dừng xe; trình tự, thủ tục đưa bến xe, trạm dừng nghỉ vào khai thác."]

# a = bm25._stem(docs)
embeddings = []
def remove_non_alphanumeric(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
document = remove_non_alphanumeric(docs[0])
tokens = bm25.tokenizer.tokenize(document)
stemmed_tokens = bm25._stem(tokens)
token_id2value = bm25._term_frequency(stemmed_tokens)
# embeddings.append(SparseEmbedding.from_dict(token_id2value))
print(f"\nVăn bản gốc: {document}")
print(f"Tokens sau khi tách: {tokens}")
print(f"Tokens sau stemming: {stemmed_tokens}")
print(f"Chỉ số và giá trị: {token_id2value}")

print(len(tokens))
print(len(stemmed_tokens))
print(len(token_id2value))

# {"indices":[25735538,397376024,455241967,536867153,631755095,724779215,725511616,786953846,812180802,847782532,863080113,873203361,875535473,877198705,1103508420,1143935419,1370014596,1526851980,1555540962,1601947068,1745044521,1749866956,1810453357,1964716150,2109066402,2128973781,2142926160],"values":[1.5421686,1.5421686,1.8132646,1.5421686,1.9261286,1.5421686,1.5421686,1.8132646,1.5421686,1.9261286,1.5421686,1.8132646,1.8132646,1.5421686,1.5421686,1.5421686,1.5421686,1.5421686,1.5421686,1.5421686,1.5421686,1.5421686,1.5421686,1.5421686,1.8132646,1.5421686,1.5421686]}