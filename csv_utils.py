import os
import pandas as pd
from urllib.request import urlretrieve, build_opener, install_opener
from sklearn.model_selection import train_test_split
from typing import Tuple


# 三个中文情感分析数据集的GitHub原始下载链接
DATASET_URLS = {
    "hotel": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
    "waimai": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
    "weibo": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/weibo_senti_100k/weibo_senti_100k.csv"
}


def ensure_merged_datasets() -> Tuple[str, str, str]:
    """
    下载 hotel, waimai, weibo 三个数据集并合并为一个大数据集
    合并后的数据集按 80%:10%:10% 的比例划分为训练集、验证集和测试集
    如果合并后的文件已存在，则直接跳过处理

    返回:
        (训练集路径, 验证集路径, 测试集路径)
    """
    output_dir = "csv_data"
    os.makedirs(output_dir, exist_ok=True)
    train_csv = os.path.join(output_dir, "merged_train.csv")
    val_csv = os.path.join(output_dir, "merged_val.csv")
    test_csv = os.path.join(output_dir, "merged_test.csv")

    # 如果合并后的文件已存在，直接返回，避免重复下载和处理
    if all(os.path.exists(path) for path in [train_csv, val_csv, test_csv]):
        print("[数据] 合并数据集已存在，跳过处理。")
        return train_csv, val_csv, test_csv

    # 模拟浏览器请求，设置User-Agent防止被GitHub拦截
    opener = build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    install_opener(opener)

    all_dfs = []  # 存放所有成功加载的数据集DataFrame
    print("[数据] 开始从 GitHub 获取原始数据集...")

    # 逐个下载并处理数据集
    for name, url in DATASET_URLS.items():
        try:
            raw_path = os.path.join(output_dir, f"raw_{name}.csv")
            # 如果原始文件尚未下载，则从GitHub下载
            if not os.path.exists(raw_path):
                print(f"  - 正在下载 {name} 数据集...")
                urlretrieve(url, raw_path)

            # 读取CSV文件，尝试UTF-8编码，失败则回退到GBK编码
            try:
                # 微博和外卖数据集使用UTF-8编码
                df = pd.read_csv(raw_path, encoding='utf-8')
            except UnicodeDecodeError:
                # 酒店数据集使用GBK编码
                df = pd.read_csv(raw_path, encoding='gbk')

            # 自动识别标签列和内容列的列名
            # 不同数据集的列名不同，需要统一处理
            target_label = next((c for c in df.columns if c.lower() in ['label', 'cat']), None)
            target_review = next((c for c in df.columns if c.lower() in ['review', 'text']), None)

            if target_label is None or target_review is None:
                # 如果没找到标准列名，假设第一列是标签，第二列是内容
                df = df.iloc[:, [0, 1]]
                df.columns = ['label', 'review']
            else:
                # 提取标签和内容列，统一列名为label和review
                df = df[[target_label, target_review]]
                df.columns = ['label', 'review']

            # 数据清洗，确保数据质量
            df['label'] = pd.to_numeric(df['label'], errors='coerce')  # 将标签转为数值，无法转换的设为NaN
            df = df.dropna(subset=['label', 'review'])  # 删除标签或内容为空的行
            df = df[df['label'].isin([0, 1])]  # 严格只保留二分类样本（0=负面, 1=正面）
            df['label'] = df['label'].astype(int)  # 标签转为整数类型
            df['review'] = df['review'].astype(str).str.strip()  # 内容转为字符串并去除首尾空格
            df = df[df['review'].str.len() > 0]  # 删除空字符串

            print(f"  - {name} 加载成功，有效数据: {len(df)} 条")
            all_dfs.append(df)
        except Exception as e:
            # 单个数据集失败不影响其他数据集的处理
            print(f"  - [警告] 下载/处理 {name} 失败: {e}")

    # 检查是否至少有一个数据集成功加载
    if not all_dfs:
        raise Exception("[错误] 没有任何数据集被成功加载，请检查网络是否能访问 GitHub Raw。")

    # 合并所有数据集
    merged_df = pd.concat(all_dfs, axis=0, ignore_index=True)

    # 全局随机打乱数据顺序，增加泛化能力，使用固定随机种子保证可复现
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"[合并] 完成！总样本数: {len(merged_df)}")
    print(f"  - 正向样本(1): {len(merged_df[merged_df['label'] == 1])}")
    print(f"  - 负向样本(0): {len(merged_df[merged_df['label'] == 0])}")

    # 划分数据集：80%训练，10%验证，10%测试
    # stratify参数确保各子集中正负样本比例与原始数据一致
    train_df, temp_df = train_test_split(
        merged_df, test_size=0.2, random_state=42, stratify=merged_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )

    # 保存为UTF-8-sig编码的CSV文件（带BOM头，Excel可正确打开中文）
    train_df.to_csv(train_csv, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_csv, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_csv, index=False, encoding='utf-8-sig')
    print(f"[保存] 训练集: {len(train_df)}条, 验证集: {len(val_df)}条, 测试集: {len(test_df)}条")

    return train_csv, val_csv, test_csv


if __name__ == "__main__":
    # 执行此脚本将生成合并后的三个数据集文件
    ensure_merged_datasets()
