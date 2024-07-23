import json

def split_jsonl_file(input_file, output_prefix, n):
    # 计算文件的总行数
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # 计算每份的大小
    lines_per_file = total_lines // n
    remainder = total_lines % n

    # 逐行读取并切分文件
    with open(input_file, 'r', encoding='utf-8') as f:
        for i in range(n):
            output_file = f"{output_prefix}_part_{i+1}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as out_f:
                # 计算当前份的行数
                current_lines = lines_per_file + (1 if i < remainder else 0)
                for _ in range(current_lines):
                    line = f.readline()
                    if not line:
                        break
                    out_f.write(line)

if __name__ == "__main__":
    input_file = "/mnt/wfs/mmshanghaiwfssh/project_searcher-others-a100/user_binghaotang/data/person_train_mllm_swift_pair_pretrain.jsonl"
    output_prefix = "./pretrain/split_file"
    n = 50  # 切分成 5 份
    split_jsonl_file(input_file, output_prefix, n)