import json
import re

def clean_triples_with_alphanumeric(file_path):
    # 编译正则表达式，用于检测是否-9、a-z、A-Z的字符
    alnum_pattern = re.compile(r'[a-zA-Z0-9]')
    
    # 读取JSON数据
    with open(file_path, 'r', encoding='utf-8') as f:
        triples = json.load(f)
    
    cleaned_triples = []
    removed_triples = []
    
    for triple in triples:
        subject = triple[0]
        obj = triple[-1]
        
        # 检查主体是否包含数字或字母
        subject_has_alnum = isinstance(subject, str) and alnum_pattern.search(subject) is not None
        # 检查客体是否包含数字或字母
        obj_has_alnum = isinstance(obj, str) and alnum_pattern.search(obj) is not None
        
        if subject_has_alnum:
            removed_triples.append(triple)   
        elif obj_has_alnum:
            removed_triples.append(triple)
        else:
            cleaned_triples.append(triple)
    
    # 输出清洗结果统计
    print(f"原始三元组数量: {len(triples)}")
    print(f"清洗后三元组数量: {len(cleaned_triples)}")
    print(f"移除的三元组数量: {len(removed_triples)}")
    
    # 打印部分被移除的三元组示例
    print("\n被移除的三元组示例:")
    for i, item in enumerate(removed_triples[:]):  
        print(f"{i+1}. {item}")
    
    return cleaned_triples, removed_triples

# 使用示例
if __name__ == "__main__":
    file_path = 'all_triples.json'  # 替换为你的文件路径
    cleaned, removed = clean_triples_with_alphanumeric(file_path)
    
    # 保存清洗后的结果
    output_path = 'cleaned_triples_no_alnum.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"\n清洗后的三元组已保存到: {output_path}")
    