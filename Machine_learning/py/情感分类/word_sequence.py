# 实现的是,构建词典,实现方法,把句子转换为数字序列和 把数字序列转换为句子
# 词嵌入

class Word2Sequence:
    UNK_TAG = "UNK" # 用于 测试集中的未知字符指代
    PAD_TAG = "PAD" # 用于 填充
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        } # 按序排列词语和序号的字典
        self.count = {} # 统计词频的字典

    def fit(self, sentence):
        """
        把单个句子保存到dict中
        sentence: [word1, word2, word3...]
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, min=5, max=None, max_features=None):
        """
        生成词典
        清洗词典
        :param min:最小出现的次数
        :param max:最大的次数
        :param max_features:一共保留多少个词语
        :return:
        """
        # 删除count中词频小于min的word
        if min is not None:
            self.count = {word:value for word,value in self.count.items() if value>min}
        # 删除count中词频大于max的word
        if max is not None:
            self.count = {word:value for word,value in self.count.items() if value<max}
        # 限制保留的词语数
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x:x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)

        # 得到一个反转的字典 把key和value 互换,方便直接用单词找序号
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))
        # 到此为止,第二步, 将词语存入词典,根据次数对词语进行过滤,并统计次数的任务完成

    def transform(self, sentence, max_len=None):
        """
        把句子转换为数字序列
        :param sentence: [word1, word2...]
        :param max_len: int 对句子进行填充或者裁剪
        :return:
        """
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len] #裁剪

        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        """
        把序列转换为句子
        :param indices: [1,2,3,4,...]
        :return:
        """
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)

if __name__ == "__main__":
    ws = Word2Sequence()
    ws.fit(["我","是","谁"])
    ws.fit(["我","是","我"])
    ws.build_vocab(min=0)
    print(ws.dict)
    ret = ws.transform(["我","爱","北京"],max_len=10)
    ret = ws.inverse_transform(ret)
    print(ret)