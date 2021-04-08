class Single_Tokenizer():
    def __init__(self):
        self.name = "single_tokenizer"

    def cut(self,words):
        # 如果有英文的话，如何把英文拿出来，这里可以优化
        return list(words)