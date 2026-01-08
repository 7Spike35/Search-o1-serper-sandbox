import pandas as pd
import json
import os
# 1. 设定路径
input_path = '/online1/ycsc_lijt1/lijt1/wxy/ASearcher-train/evaluation/browse_comp_test_set.csv'
output_path = '/online1/ycsc_lijt1/lijt1/tyr/Search-o1/data/BrowserComp/broswercomp.json'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
# 2. 读取 CSV
try:
    df = pd.read_csv(input_path)
    print("成功读取 CSV 文件，列名:", df.columns.tolist())
except Exception as e:
    print(f"读取失败: {e}")
    exit(1)
# 3. 转换逻辑
data_list = []
for idx, row in df.iterrows():
    # 尽可能兼容多种列名
    # 如果你的 CSV 里问题列叫 'Input' 或 'Query'，请在这里添加
    question = (row.get('question') or row.get('Question') or 
                row.get('problem') or row.get('Problem') or 
                row.get('input') or row.get('query'))
    
    # 答案列
    answer = (row.get('answer') or row.get('Answer') or 
              row.get('solution') or row.get('golden_answer'))
    
    if pd.isna(question):
        continue # 跳过空行
    item = {
        "id": idx,
        "Question": str(question),
        "answer": str(answer) if not pd.isna(answer) else "",
        # 保留可能有的额外信息，比如 URL 列表
        "urls": row.get('urls', []) 
    }
    data_list.append(item)
# 4. 保存
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)
print(f"转换完成！保存为 {output_path}，共 {len(data_list)} 条。")