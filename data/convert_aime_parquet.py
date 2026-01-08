import pandas as pd
import json
import os
# 1. 设定路径
# 请确保你怎么下载的 parquet 文件放在这里
input_path = '/online1/ycsc_lijt1/lijt1/tyr/Search-o1/data/AIME/original_data/aime_2024_problems.parquet' 
output_path = '/online1/ycsc_lijt1/lijt1/tyr/Search-o1/data/AIME/aime_2024.json'
# 确保输出目录存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# 2. 读取 Parquet
# 注意: 如果你下载的是 test-00000-of-00001.parquet 这种名字，请相应修改 input_path
try:
    df = pd.read_parquet(input_path)
    print("成功读取 Parquet 文件，列名:", df.columns.tolist())
except Exception as e:
    print(f"读取失败，请检查路径或安装依赖 (pip install pyarrow). 错误: {e}")
    exit(1)
# 3. 格式转换
# Search-o1 需要的字段通常是: id, Question, answer (或者 solution)
# 我们需要查看 dataframe 的列名来做映射。
# 假设 MathArena 数据集的列是: problem, answer, problem_type...
data_list = []
for idx, row in df.iterrows():
    # 尝试兼容不同的列名命名习惯
    question = row.get('problem') or row.get('question') or row.get('Question')
    answer = row.get('answer') or row.get('Answer') or row.get('solution')
    
    # 如果没有 ID，我们自己生成一个
    item = {
        "id": idx,
        "Question": question,
        "answer": str(answer),  # 确保转为字符串
        # 保留可能有的 solution 字段，虽然不是必须
        "solution": row.get('solution', '') 
    }
    data_list.append(item)
# 4. 保存为 JSON
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)
print(f"转换完成！已保存到 {output_path}")
print(f"共处理 {len(data_list)} 条数据。")