import json
import re

def triple_deduplicate(original_data):
    # 去重处理
    unique_tuples = set(tuple(item) for item in original_data)
    unique_data = [list(tpl) for tpl in unique_tuples]
    return unique_data

def remove_null_data(original_data):
    # 过滤掉包含空值的三元组
    filtered_data = []
    for triple in original_data:
        has_empty = any(s == "" for s in triple)
        if not has_empty:
            filtered_data.append(triple)
    return filtered_data

def remove_triples_with_short_elements(original_data):
    # 三元组清洗:清洗客体只有一个字符的异常数据
    clean_triples = []
    for triple in original_data:
        obj = triple[-1]
        # 检查客体是否为单个字符
        if len(obj.strip()) != 1:
            clean_triples.append(triple)
    return clean_triples

def remove_triples_with_long_elements(original_data, length_threshold=10):
    # 三元组清洗:清洗主体或客体超过十个字符的异常数据
    cleaned_triples = []
    for triple in original_data: 
        subject = triple[0]
        obj = triple[-1]
        if len(subject.strip())<=length_threshold and len(obj.strip())<=length_threshold:
            cleaned_triples.append(triple)
    return cleaned_triples

def replace_abbreviations_in_triples(file_path, abbreviation_map, output_path):
    abbr_prefix_map = {}
    for abbr, full in abbreviation_map.items():
        # 提取全称的所有可能前缀（从1个字符到全称长度-1）
        prefixes = [full[:i] for i in range(1, len(full))]
        abbr_prefix_map[abbr] = {
            'full': full,
            'prefixes': prefixes  # 存储所有可能的前缀字符组合
        }
    # 读取三元组数据
    with open(file_path, 'r', encoding='utf-8') as f:
        triples = json.load(f)
    replaced_triples = []
    replacement_records = []
    for triple in triples:
        subject, predicate, obj = triple
        original_triple = (subject, predicate, obj)
        # 替换主体和客体中的缩写
        subject = replace_abbreviation_with_context(subject, abbr_prefix_map)
        obj = replace_abbreviation_with_context(obj, abbr_prefix_map)
        replaced_triple = (subject, predicate, obj)
        replaced_triples.append(replaced_triple)
        
        if original_triple != replaced_triple:
            replacement_records.append((original_triple, replaced_triple))
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(replaced_triples, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"处理完成！共处理 {len(triples)} 个三元组")
    print(f"其中 {len(replacement_records)} 个三元组包含缩写并已替换")
    
    # 显示替换示例
    print("\n替换示例：")
    for i, (original, replaced) in enumerate(replacement_records[:]):
        print(f"{i+1}. 原始: {original}")
        print(f"   替换后: {replaced}\n")
    
    return replaced_triples, replacement_records

def replace_abbreviation_with_context(text, abbr_prefix_map):
    # 处理替换重复文本问题
    if not isinstance(text, str):
        return text
    # 遍历所有缩写进行处理
    for abbr, info in abbr_prefix_map.items():
        full_name = info['full']
        prefixes = info['prefixes']
        abbr_len = len(abbr)
        
        # 从左到右查找所有缩写位置
        start = 0
        while start <= len(text) - abbr_len:
            # 精确匹配缩写（避免部分匹配）
            if text[start:start+abbr_len] == abbr:
                # 获取缩写前面的文本
                prefix_text = text[:start].strip()
                
                # 检查前面的文本是否包含全称的任何前缀
                redundant_prefix = None
                for prefix in prefixes:
                    if prefix_text.endswith(prefix):
                        redundant_prefix = prefix
                        break
                
                if redundant_prefix:
                    # 计算需要保留的部分（去除重复前缀）
                    remaining = full_name[len(redundant_prefix):]
                    # 替换缩写为剩余部分
                    text = text[:start] + remaining + text[start+abbr_len:]
                    # 移动指针，跳过已处理部分
                    start += len(remaining)
                else:
                    # 无重复前缀，直接替换
                    text = text[:start] + full_name + text[start+abbr_len:]
                    # 移动指针，跳过已处理部分
                    start += len(full_name)
            else:
                start += 1
    
    return text

def remove_suffix_from_subject(file_path, output_path, pattern=r'(c2)'):
    """
    去除三元组主体中的特定后缀
    """
    # 编译正则表达式，用于匹配并移除指定模式
    # 添加\s*处理可能的空格，如"( c2 )"
    regex = re.compile(re.escape(pattern) + r'\s*')
    
    # 读取三元组数据
    with open(file_path, 'r', encoding='utf-8') as f:
        triples = json.load(f)
    
    cleaned_triples = []
    modified_records = []  # 记录修改前后的三元组
    
    for triple in triples:
        subject, predicate, obj = triple
        original_subject = subject
        
        # 去除主体中的(c2)标记（支持带空格的情况，如"( c2 )"）
        cleaned_subject = regex.sub('', subject).strip()
        
        # 构建清洗后的三元组
        cleaned_triple = (cleaned_subject, predicate, obj)
        cleaned_triples.append(cleaned_triple)
        
        # 记录修改前后的差异
        if cleaned_subject != original_subject:
            modified_records.append({
                'original': (original_subject, predicate, obj),
                'cleaned': cleaned_triple
            })
    
    # 保存清洗后的结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_triples, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    print(f"处理完成！共处理 {len(triples)} 个三元组")
    print(f"其中 {len(modified_records)} 个三元组的主体包含'{pattern}'并已移除")
    
    # 显示部分修改示例
    print("\n修改示例：")
    for i, record in enumerate(modified_records[:]):
        print(f"{i+1}. 原始: {record['original']}")
        print(f"   清洗后: {record['cleaned']}\n")
    
    return cleaned_triples, modified_records
    
if __name__ == "__main__":
    # 文件路径
    file_path = 'original_triples.json'
    # 保存路径
    out_path = 'triples_without_c2.json'
    # 读取JSON数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))
    # 缩写对应全称
    abbr_map = {
        'osahs':'阻塞性睡眠呼吸暂停低通气综合征',
        "osa": "阻塞性睡眠呼吸暂停",
        "cbti": "认知行为疗法",
        "rls": "不宁腿综合征",
        'sdb':'睡眠呼吸障碍',
        'eds':'日间过度思睡',
        'mslt':'多次睡眠潜伏期试验',
        'comisa':'失眠和睡眠呼吸暂停共病',
    }
    clean_data = triple_deduplicate(data)
    print(len(clean_data))
    # 覆盖写入原文件
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)
    print(f"已将处理后的数据覆盖写入 {out_path}")
    