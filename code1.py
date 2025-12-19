# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 14:10:05 2025

@author: 张士涵
"""
# 中文文本分词与词频统计实验
import os
from collections import Counter

class ChineseSegmenter:
    def __init__(self, dict_path, stopwords_path):
        #初始化
        self.dictionary = self.load_dictionary(dict_path)
        self.stopwords = self.load_stopwords(stopwords_path)
        self.max_word_len = max(len(word) for word in self.dictionary) if self.dictionary else 0
    
    def load_dictionary(self, dict_path):
        #分词词典
        dictionary = set()
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        dictionary.add(word)
            return dictionary
        except FileNotFoundError:
            return set()
    
    def load_stopwords(self, stopwords_path):
        """加载停用词词典"""
        stopwords = set()
        try:
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        stopwords.add(word)
            return stopwords
        except FileNotFoundError:
            return set()
    
    def forward_max_match(self, text, max_len=None):
        
        #正向最大匹配算法
        #param text: 待分词文本
        #param max_len: 最大词长，默认为词典中的最大词长
        
        if max_len is None:
            max_len = self.max_word_len
        
        words = []  # 存储分词结果
        index = 0   # 当前处理位置指针
        text_len = len(text)
        
        # 遍历整个文本
        while index < text_len:
            matched = False  # 标记是否匹配到词
            
            # 从最大长度开始尝试匹配，逐渐减小长度
            for length in range(min(max_len, text_len - index), 0, -1):
                word = text[index:index + length]  # 截取候选词
                
                # 如果候选词在词典中，则匹配成功
                if word in self.dictionary:
                    words.append(word)    # 将词加入结果列表
                    index += length       # 移动指针到下一个位置
                    matched = True        # 标记已匹配
                    break                 # 跳出当前循环，处理下一个位置
            
            # 如果没有匹配到任何词，按单字切分
            if not matched:
                words.append(text[index])  # 将单字作为词加入结果
                index += 1                 # 指针前进一位
        
        return words
    
    def backward_max_match(self, text, max_len=None):
        
        #逆向最大匹配算法
        #param text: 待分词文本
        #param max_len: 最大词长，默认为词典中的最大词长
        
        if max_len is None:
            max_len = self.max_word_len
        
        words = []  # 存储分词结果
        index = len(text)  # 从文本末尾开始处理
        
        # 从后往前遍历整个文本
        while index > 0:
            matched = False  # 标记是否匹配到词
            
            # 从最大长度开始尝试匹配，逐渐减小长度
            for length in range(min(max_len, index), 0, -1):
                word = text[index - length:index]  # 截取候选词（从后往前）
                
                # 如果候选词在词典中，则匹配成功
                if word in self.dictionary:
                    words.insert(0, word)  # 将词插入到结果列表开头（保持顺序）
                    index -= length        # 移动指针到前一个位置
                    matched = True         # 标记已匹配
                    break                  # 跳出当前循环，处理下一个位置
            
            # 如果没有匹配到任何词，按单字切分
            if not matched:
                words.insert(0, text[index - 1])  # 将单字插入到结果列表开头
                index -= 1                        # 指针后退一位
        
        return words
    
    def remove_stopwords(self, words):
        #去除停用词
        return [word for word in words if word not in self.stopwords]
    
    def calculate_word_frequency(self, words, remove_stopwords=True):
        
        #计算词频
        #param words: 分词结果列表
        #param remove_stopwords: 是否去除停用词
        
        if remove_stopwords:
            words = self.remove_stopwords(words)
        
        word_freq = Counter(words)
        return word_freq

def main():
    # 文件路径配置
    dict_path = r"E:\nlp\experiment1_data\experiment1_data\data\dictionary.txt"
    stopwords_path = r"E:\nlp\experiment1_data\experiment1_data\data\stoplist.txt"
    input_file = r"E:\nlp\experiment1_data\experiment1_data\data\input.txt" 
    
    # 读取输入文本
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            sample_text = f.read().strip()
    except FileNotFoundError:
        return
    
    # 初始化分词器
    segmenter = ChineseSegmenter(dict_path, stopwords_path)
    if not segmenter.dictionary:
        return
    # 分词
    fmm_words = segmenter.forward_max_match(sample_text)
    bmm_words = segmenter.backward_max_match(sample_text)
    # 词频统计
    fmm_freq = segmenter.calculate_word_frequency(fmm_words)
    bmm_freq = segmenter.calculate_word_frequency(bmm_words)
    
    output_dir = r"E:\nlp\experiment1_data\experiment1_data\data\output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存分词结果
    with open(os.path.join(output_dir, "segmentation_results.txt"), 'w', encoding='utf-8') as f:
        f.write("正向最大匹配分词结果 \n")
        f.write(" ".join(fmm_words))
        f.write("\n\n逆向最大匹配分词结果 \n")
        f.write(" ".join(bmm_words))
    
    
    # 词频从高到低排序
    with open(os.path.join(output_dir, "sorted_results.txt"), 'w', encoding='utf-8') as f:
        f.write("正向最大匹配词频排序 \n")
        f.write("排名\t词语\t频次\n")
        for i, (word, freq) in enumerate(fmm_freq.most_common(), 1):
            f.write(f"{i}\t{word}\t{freq}\n")
        
        f.write("\n逆向最大匹配词频排序\n")
        f.write("排名\t词语\t频次\n")
        for i, (word, freq) in enumerate(bmm_freq.most_common(), 1):
            f.write(f"{i}\t{word}\t{freq}\n")

if __name__ == "__main__":
    main()