# 定义文件路径（根据实际情况修改路径）
input_file_path = '/home/jxzhou/projects/himol-instruct/data/ChEBI-20_data/train.txt'    # 原始文件路径
output_file_path = '/home/jxzhou/projects/himol-instruct/data/ChEBI-20_data/train_3300.txt'  # 新文件路径

# 读取前3300行数据
with open(input_file_path, 'r', encoding='utf-8') as infile:
    # 使用islice高效读取指定行数
    from itertools import islice
    lines = list(islice(infile, 3301))

# 写入新文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    outfile.writelines(lines)

print(f'已成功导出前{len(lines) - 1}条数据至 {output_file_path}')