import pandas as pd
import json
import sys


def convert_gaia_to_aime_format(parquet_path, output_path):
    try:
        print(f"正在读取文件: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        # --- 调试步骤：打印所有列名 ---
        all_columns = df.columns.tolist()
        print(f"文件包含的列名: {all_columns}")

        # 1. 自动寻找包含答案的列名 (增加了 'Final answer' 等变体)
        possible_answer_cols = [
            'Final answer',  # <--- 针对你文件中的实际列名
            'Final Answer',
            'final answer',
            'Final_Answer',
            'answer',
            'Answer',
            'ground_truth'
        ]

        answer_col = None

        # 精确匹配
        for col in possible_answer_cols:
            if col in df.columns:
                answer_col = col
                break

        # 如果精确匹配没找到，尝试忽略大小写的模糊匹配
        if answer_col is None:
            print("正在尝试忽略大小写进行模糊搜索...")
            df_cols_lower = {c.lower(): c for c in df.columns}  # 创建映射：小写->原名
            for target in possible_answer_cols:
                if target.lower() in df_cols_lower:
                    answer_col = df_cols_lower[target.lower()]
                    break

        if answer_col:
            print(f"--> ✅ 成功匹配到答案列: '{answer_col}'")
        else:
            print("❌ 错误: 仍然找不到答案列。请检查列名是否拼写完全不同。")
            return

        # 2. 定义判断为空的函数
        def is_empty(val):
            if pd.isna(val) or val is None:
                return True
            if isinstance(val, str) and val.strip() == '':
                return True
            return False

        # 3. 筛选逻辑
        if 'file_name' in df.columns and 'file_path' in df.columns:
            mask = df['file_name'].apply(is_empty) & df['file_path'].apply(is_empty)
            filtered_df = df[mask]
            print(f"筛选条件 (无文件依赖) 命中: {len(filtered_df)} / {len(df)} 条数据")
        else:
            print("⚠️ 警告: 未找到 'file_name' 或 'file_path' 列，将跳过筛选，转换所有数据。")
            filtered_df = df

        output_data = []

        # 4. 转换数据
        records = filtered_df.to_dict('records')

        for idx, row in enumerate(records, start=0):
            # 获取问题
            if 'Question' in row:
                question = row['Question']
            elif 'question' in row:
                question = row['question']
            else:
                question = ""

            # 获取答案
            raw_answer = row.get(answer_col, "")

            # 清理答案格式
            if raw_answer is None:
                answer_str = ""
            else:
                answer_str = str(raw_answer).strip()

            entry = {
                "id": idx,
                "Question": str(question),
                "answer": answer_str,
                "solution": answer_str
            }
            output_data.append(entry)

        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"✅ 转换完成！共 {len(output_data)} 条数据已保存至 {output_path}")

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 请确保路径正确
    input_parquet = "./2023/validation/metadata.level1.parquet"
    output_json = "./gaia_level1.json"

    convert_gaia_to_aime_format(input_parquet, output_json)