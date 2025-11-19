import markdown
from bs4 import BeautifulSoup


def md_to_text(md_content):
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, features='html.parser')
    text = ''.join(soup.findAll(text=True))
    return text.strip()


# 读取Markdown文件内容
with open('/keeson/code/dcc/code/markdown文献/中国儿童阻塞性睡眠呼吸暂停诊断与治疗指南2020.md', 'r', encoding='utf - 8') as f:
    md_content = f.read()
txt_content = md_to_text(md_content)
# 写入TXT文件
with open('/keeson/code/dcc/code/txt文献/中国儿童阻塞性睡眠呼吸暂停诊断与治疗指南2020.txt', 'w', encoding='utf - 8') as f:
    f.write(txt_content)