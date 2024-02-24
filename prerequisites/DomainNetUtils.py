import os
import time
import random
import json
import numpy as np


def statistic(root="DomainDataset/DomainNet"):
    """
    统计每个Domain中各个类别出现的数目
    """
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    categories = sorted(os.listdir(os.path.join(root, domains[0])))
    print(categories)
    result = {}
    for domain in domains:
        domain_image_list_path = os.path.join(root, "image_list", domain + ".txt")
        result[domain] = {}
        for i in range(345):
            result[domain][str(i)] = 0  # 初始化

        with open(domain_image_list_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                filename = line.split(" ")[0]
                class_id = line.split(" ")[1]  # 从零开始标号，0-344
                class_name = filename.split("/")[0]
                result[domain][str(class_id)] += 1

    json.dump(result, open("DomainDataset/statistic_result.json", "w"), indent=4)
    pass

def best_split(M):
    selected_columns = set() # 用于存储已选择的列索引
    total_sum = 0 # 用于累加选中元素的和
    num_rows, num_cols = M.shape
    
    # 对于前num_rows-1行，每行选择57个元素
    selections_per_row = [57] * (num_rows - 1)
    # 最后一行选择60个元素
    selections_per_row.append(60)
    
    row_sums = []  # 记录每一行的和
    selected_indexes = []
    for row, num_selections in zip(M, selections_per_row):
        row_sum = 0
        row_selected_indexes = []
        # 生成元素及其索引的列表，并按值降序排列
        indexed_row = sorted([(value, index) for index, value in enumerate(row)], reverse=True)
        
        selected_count = 0 # 当前行已选择的元素计数
        for value, index in indexed_row:
            # 如果该列还没有被选择，并且当前行的选择数量小于对应行的选择目标
            if index not in selected_columns and selected_count < num_selections:
                total_sum += value # 累加选中元素的值
                row_sum += value

                row_selected_indexes.append(index)
                selected_columns.add(index) # 将列索引添加到已选择集合中
                selected_count += 1 # 更新当前行已选择的元素数量
                
            # 如果当前行已经选择了目标数量的元素，就继续下一行
            if selected_count == num_selections:
                row_sums.append(row_sum)
                selected_indexes.append(row_selected_indexes)
                break
                
    return row_sums, selected_indexes


def design_split(json_file = "DomainDataset/statistic_result.json"):
    """
    根据统计结果，从每个Domain中挑选出57个类别（最后一个Domain有60个类别）
    """
    with open(json_file, "r") as f:
        raw_data = json.load(f)
        f.close()
    domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
    random.shuffle(domains)

    matrix = np.zeros((6, 345), dtype=np.uint16)  # 将样本出现的次数整理为6*345的矩阵
    for i, domain in enumerate(domains):
        row = []
        for key, value in raw_data[domain].items():
            row.append(value)
        matrix[i] = np.array(row)
    selected_sums, selected_indexes = best_split(matrix)
    record_split(domains, selected_indexes, selected_sums)


def record_split(domains_list, selected_indexes, selected_sums):
    """
    记录Domain拆分的方法。
    domains_list: domain出现的顺序
    selected_indexes: 每个domain中出现的类别标号
    selected_nums: 每个domain中的样本总数
    """
    assert len(selected_indexes[0]) == 57 and len(selected_indexes[-1]) == 60, "Domain拆分失败，请检查每个Domain中的类别数目是否符合要求！"
    dir_name = "./DomainDataset/split/split_design_seed_{}_total_{}".format(seed, sum(selected_sums))
    if os.path.exists(dir_name) == False:
        os.makedirs(dir_name)
    filename = "split.json"
    filename = dir_name + "/" + filename

    raw_data = {}
    raw_data["domains_order"] = domains_list
    for i in range(len(domains_list)):
        title = "task_{}".format(i)
        raw_data[title] = {}
        raw_data[title]["classes"] = selected_indexes[i]
        raw_data[title]["sample_nums"] = str(selected_sums[i])
    json.dump(raw_data, open(filename, "w"), indent=4)


def split(design_name):
    """
    根据设计的拆分方案，进行拆分
    """
    json_path = "./DomainDataset/split/{}/split.json".format(design_name)
    raw_data = json.load(open(json_path, "r"))

    train_txt_file_name = "./DomainDataset/split/{}/train.txt".format(design_name)
    test_txt_file_name = "./DomainDataset/split/{}/test.txt".format(design_name)
    train_list, test_list = [], []

    domains_list = raw_data["domains_order"]  # domain出现的顺序
    for i, domain in enumerate(domains_list):
        title = "task_{}".format(i)
        classes_in_this_domain = raw_data[title]["classes"]
        sample_num_in_this_domain = int(raw_data[title]["sample_nums"])
        train_samples_in_this_domain = generate_file_list(domain, classes_in_this_domain, "train", sample_num_in_this_domain)
        test_samples_in_this_domain = generate_file_list(domain, classes_in_this_domain, "test", sample_num_in_this_domain)
        train_list.extend(train_samples_in_this_domain)
        test_list.extend(test_samples_in_this_domain)
        assert len(train_samples_in_this_domain) + len(test_samples_in_this_domain) == sample_num_in_this_domain, "Error: train_sample + test_sample != sample_num_in_this_domain"
    assert len(train_list) + len(test_list) == int(design_name.split("_")[-1])
    
    with open(train_txt_file_name, "w") as f:
        f.writelines(train_list)
        f.close()
    with open(test_txt_file_name, "w") as f:
        f.writelines(test_list)
        f.close()

def generate_file_list(domain_name, classes, mode, sample_num_in_this_domain):
    """
    根据给定的Domain，从中挑选出classes，mode为train/test
    """
    assert mode in ["train", "test"], "Please confirm that mode in either train or test"
    official_split_filename = "./DomainDataset/DomainNet/splits/{}_{}.txt".format(domain_name, mode)
    result = []
    with open(official_split_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            category_id = line.split(" ")[-1]
            if int(category_id) in classes:
                result.append(line + "\n")
    # assert len(result) == sample_num_in_this_domain, "Please check the sample number"
    return result



if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    # statistic()
    # design_split()
    split("split_design_seed_42_total_126540")