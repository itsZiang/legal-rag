train reranker áp dụng phương pháp LLM pairwise, tức là sử dụng LLM để đánh giá các cặp câu hỏi và chunks

xây dựng data train:
1. retrieval top-5 chunk dùng hybrid search
2. dùng LLM để đánh giá (question, chunk1, chunk2) -> output là chunk nào có độ chính xác cao hơn
3. cuối cùng làm thế nào để có được (question, pos, neg)

train model dùng thư viện flag embedding