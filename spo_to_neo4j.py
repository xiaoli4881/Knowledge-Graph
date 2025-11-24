from neo4j import GraphDatabase
import json

# Neo4j数据库连接信息
uri = ""  # Neo4j地址
username = ""  # 用户名
password = ""  # 密码
JSON_FILE_PATH = ""  # 三元组JSON文件路径

class TripleImporter:
    def __init__(self, uri, user, password):
        self.driver = None
        self.uri = uri
        self.user = user
        self.password = password

    def connect(self):
        """建立数据库连接"""
        self.driver = GraphDatabase.driver(self.uri,auth=(self.user, self.password))
        # 测试连接
        with self.driver.session() as session:
            session.run("RETURN 1")
        print("成功连接到Neo4j数据库")
        
    def create_constraints(self):
        """创建唯一约束，避免重复节点"""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
        print("成功创建节点约束")

    def import_triple(self, session, head, relation, tail):
        """导入单个三元组"""
        query = """
        MERGE (h:Entity {name: $head})
        MERGE (t:Entity {name: $tail})
        MERGE (h)-[r:RELATION {name: $relation}]->(t)
        RETURN h.name, r.name, t.name
        """
        result = session.run(query, head=head, relation=relation, tail=tail)
        record = result.single()
        if record:
            print(f"导入: {record[0]} -[{record[1]}]-> {record[2]}")
            return True
        else:
            return False
      

    def import_from_json(self, json_path):
        """从JSON文件导入所有三元组"""
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        print(f"成功读取JSON文件，共发现 {len(triples)} 条三元组")

        # 批量导入
        with self.driver.session() as session:
            success_count = 0
            for i, triple in enumerate(triples, 1):
                head, relation, tail = triple
                # 去除可能的空白字符
                head = head.strip()
                relation = relation.strip()
                tail = tail.strip()
                if self.import_triple(session, head, relation, tail):
                    success_count += 1
                print(f"导入完成: 成功 {success_count}/{len(triples)} 条")


    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("数据库连接已关闭")

if __name__ == "__main__":
    # 创建导入器实例
    importer = TripleImporter(uri, username, password)
    # 连接数据库
    importer.connect()
    importer.create_constraints()
    # 从JSON导入数据
    importer.import_from_json(JSON_FILE_PATH)
    # 确保连接关闭
    importer.close()
