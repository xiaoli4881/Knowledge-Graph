# coding:utf-8 基于bert+lstm+crf模型的医疗领域实体抽取
import codecs
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import json
from utils import load_vocab
from dcc.code.knowledge_graph.model_ner.ner_constant import *
from model_ner import BERT_LSTM_CRF


class medical_ner(object):
    def __init__(self):
        self.NEWPATH = '/keeson/code/dcc/code/knowledge_graph/model/medical_ner/medical_ner/model.pkl' #实体抽取模型权重
        self.vocab = load_vocab('/keeson/code/dcc/code/knowledge_graph/model/medical_ner/medical_ner/vocab.txt') #分词表
        self.vocab_reverse = {v: k for k, v in self.vocab.items()}#分词表反向映射
        self.model = BERT_LSTM_CRF('/keeson/code/dcc/code/knowledge_graph/model/medical_ner/medical_ner', tagset_size, 768, 200, 2,
                              dropout_ratio=0.5, dropout1=0.5, use_cuda=use_cuda)#初始化bert+lstm+crf模型 bert语义编码+lstm捕捉序列特征+crf优化标签序列
        if use_cuda:
            self.model.to(device)
    def from_input(self, input_str):
        # 处理单句文本
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        text = ['[CLS]'] + [x for x in input_str] + ['[SEP]']# 加bert特殊标记
        raw_text.append(text)
        cur_len = len(text)
        # raw_textid = [self.vocab[x] for x in text] + [0] * (max_length - cur_len) 数值化 （将字转换成分词表id，长度不足用0补全）
        raw_textid = [self.vocab[x] for x in text if self.vocab.__contains__(x)] + [0] * (max_length - cur_len)
        textid.append(raw_textid)
        raw_textmask = [1] * cur_len + [0] * (max_length - cur_len)# 生成掩码，有效文本是1，补全文本是0
        textmask.append(raw_textmask)
        textlength.append([cur_len])
        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength # 原始文本、ID、掩码、长度
    def from_txt(self, input_path):
        # 处理txt文本文件
        raw_text = []
        textid = []
        textmask = []
        textlength = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip())==0:
                    continue
                if len(line) > 448:
                    line = line[:448]
                temptext = ['[CLS]'] + [x for x in line[:-1]] + ['[SEP]']
                cur_len = len(temptext)
                raw_text.append(temptext)

                tempid = [self.vocab[x] for x in temptext[:cur_len]] + [0] * (max_length - cur_len)
                textid.append(tempid)
                textmask.append([1] * cur_len + [0] * (max_length - cur_len))
                textlength.append([cur_len])

        textid = torch.LongTensor(textid)
        textmask = torch.LongTensor(textmask)
        textlength = torch.LongTensor(textlength)
        return raw_text, textid, textmask, textlength
    def split_entity_input(self,label_seq):
        entity_mark = dict()
        entity_pointer = None
        for index, label in enumerate(label_seq):
            #print(f"before: {label_seq}")
            if label.split('-')[-1]=='B':
                category = label.split('-')[0]
                entity_pointer = (index, category)
                entity_mark.setdefault(entity_pointer, [label])
            elif label.split('-')[-1]=='M':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            elif label.split('-')[-1]=='E':
                if entity_pointer is None: continue
                if entity_pointer[1] != label.split('-')[0]: continue
                entity_mark[entity_pointer].append(label)
            else:
                entity_pointer = None
           # print(entity_mark)
        return entity_mark
    def predict_sentence(self, sentence):
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        if sentence == '':
            print("输入为空！请重新输入")
            return
        if len(sentence) > 448:
            print("输入句子过长，请输入小于448的长度字符！")
            sentence = sentence[:448]
        raw_text, test_ids, test_masks, test_lengths = self.from_input(sentence)# 文本预处理
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=device))
        self.model.eval()

        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)

            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]
            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    # print(item, ent)
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)

                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            # print(entity_list)
        return entity_list
    def predict_file(self, input_file, output_file):
        tag_dic = {"d": "疾病", "b": "身体", "s": "症状", "p": "医疗程序", "e": "医疗设备", "y": "药物", "k": "科室",
                   "m": "微生物类", "i": "医学检验项目"}
        raw_text, test_ids, test_masks, test_lengths = self.from_txt(input_file)
        test_dataset = TensorDataset(test_ids, test_masks, test_lengths)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1)
        self.model.load_state_dict(torch.load(self.NEWPATH, map_location=device))
        self.model.eval()
        op_file = codecs.open(output_file, 'w', 'utf-8')
        for i, dev_batch in enumerate(test_loader):
            sentence, masks, lengths = dev_batch
            batch_raw_text = raw_text[i]
            sentence, masks, lengths = Variable(sentence), Variable(masks), Variable(lengths)
            if use_cuda:
                sentence = sentence.to(device)
                masks = masks.to(device)
            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = self.model(sentence, masks)
            predict_tags.tolist()
            predict_tags = [i2l_dic[t.item()] for t in predict_tags[0]]
            predict_tags = predict_tags[:len(batch_raw_text)]
            pred = predict_tags[1:-1]
            raw_text = batch_raw_text[1:-1]

            entity_mark = self.split_entity_input(pred)
            entity_list = {}
            if entity_mark is not None:
                for item, ent in entity_mark.items():
                    entity = ''
                    index, tag = item[0], item[1]
                    len_entity = len(ent)
                    for i in range(index, index + len_entity):
                        entity = entity + raw_text[i]
                    entity_list[tag_dic[tag]] = entity
            op_file.write("".join(raw_text))
            op_file.write("\n")
            op_file.write(json.dumps(entity_list, ensure_ascii=False))
            op_file.write("\n")

        op_file.close()
        print('处理完成！')
        print("结果保存至 {}".format(output_file))

def read_markdown_file(file_path):
    """读取Markdown文件并按行返回文本列表"""
    with open(file_path, 'r', encoding='utf-8') as file:
        # 逐行读取并存储到列表
        lines = file.readlines()
        return lines

def split_long_line(line, max_len=256):
    """
    将超过max_len的长行分割为多个短行
    分割原则：
    1. 优先在标点符号（，。！？；）后分割
    2. 其次在空格后分割
    3. 最后强制截断（避免单词被截断）
    """
    if len(line) <= max_len:
        return [line]
    segments = []
    current_start = 0
    while current_start < len(line):
        # 尝试在max_len附近寻找合适的分割点
        end = current_start + max_len
        if end >= len(line):
            # 剩余部分不足max_len，直接作为最后一段
            segments.append(line[current_start:])
            break
        # 优先在标点符号后分割
        split_points = []
        for punct in ['，', '。', '！', '？', '；', ',', '.', '!', '?', ';']:
            idx = line.rfind(punct, current_start, end)
            if idx != -1:
                split_points.append(idx + 1)  # 分割点在标点后
        # 其次在空格后分割
        if not split_points:
            space_idx = line.rfind(' ', current_start, end)
            if space_idx != -1:
                split_points.append(space_idx + 1)
        # 如果没有找到合适的分割点，强制截断
        if not split_points:
            split_idx = end
        else:
            split_idx = max(split_points)  # 选择最接近max_len的分割点
        segments.append(line[current_start:split_idx])
        current_start = split_idx
    return segments

def process_markdown_lines(lines):
    """处理Markdown行列表，分割超过256字符的行"""
    processed_lines = []
    for line in lines:
        # 去除行末换行符
        line = line.rstrip('\n')
        # 分割长行
        segments = split_long_line(line, max_len=256)
        processed_lines.extend(segments)
    return processed_lines

if __name__ == "__main__":
    my_pred = medical_ner()
    file_path = "/keeson/code/dcc/code/markdown文献/中国儿童阻塞性睡眠呼吸暂停诊断与治疗指南2020.md"
    # 读取文件
    lines = read_markdown_file(file_path)
    if lines:
        # 处理并输出每一行
        processed_lines = process_markdown_lines(lines)
        for line in processed_lines:
            if line !='':
                sentence = line
                res = my_pred.predict_sentence(sentence)
                print("---")
                print(res)
    