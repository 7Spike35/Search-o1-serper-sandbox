import pandas as pd
import json


def convert_gaia_to_aime_format(parquet_path, output_path):
    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(parquet_path)

        # 定义判断为空的函数 (None, NaN, 或空字符串)
        def is_empty(val):
            if pd.isna(val) or val is None:
                return True
            if isinstance(val, str) and val.strip() == '':
                return True
            return False

        # 筛选条件：file_name 为空 且 file_path 为空
        # 假设 Parquet 文件中包含 'file_name' 和 'file_path' 列
        mask = df['file_name'].apply(is_empty) & df['file_path'].apply(is_empty)
        filtered_df = df[mask]

        # 构建符合 AIME 2025 格式的数据列表
        output_data = []

        # 使用 enumerate 生成从 0 开始的整数 id，保持与样例一致
        for idx, row in enumerate(filtered_df.itertuples(), start=0):
            # 获取问题文本
            question = getattr(row, 'Question', '')

            # 获取答案 (GAIA 数据集通常使用 'Final_Answer' 列名)
            if hasattr(row, 'Final_Answer'):
                answer = getattr(row, 'Final_Answer')
            elif hasattr(row, 'answer'):  # 备用列名
                answer = getattr(row, 'answer')
            else:
                answer = ""

            # 转为字符串
            answer = str(answer)

            # 构建字典条目
            # 注意：参考 aime_2025.json，'solution' 字段在此处被设为与 'answer' 一致
            entry = {
                "id": idx,
                "Question": str(question),
                "answer": answer,
                "solution": answer
            }
            output_data.append(entry)

        # 保存为 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"转换完成。共筛选出 {len(output_data)} 条数据，已保存至 {output_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


if __name__ == "__main__":
    # 输入和输出文件名
    input_parquet = "./2023/validation/metadata.level1.parquet"
    output_json = "./gaia_level1.json"

    convert_gaia_to_aime_format(input_parquet, output_json)