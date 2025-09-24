from collections import defaultdict
from prior.glove import GloVe
import pickle
import torch
import re
from itertools import islice
from collections import Counter

def swap_dict_keys_values(base_classid_attributeid_dict):
    swapped_dict = {}
    for key, value in base_classid_attributeid_dict.items():
        for item in value:
            if item not in swapped_dict:
                swapped_dict[item] = [key]
            else:
                swapped_dict[item].append(key)
    sorted_swapped_dict = {k: swapped_dict[k] for k in sorted(swapped_dict)}
    return sorted_swapped_dict


def get_glove(glove_pth,all_class_name):
    glove = GloVe(glove_pth)
    vectors = []
    num = 0
    for lemma_names in all_class_name:
        vectors.append(glove[lemma_names])
        if torch.sum(torch.abs(vectors[-1])) == 0:
            print('No found: {}'.format(lemma_names))
            num+=1
    print(num)
    vectors = torch.stack(vectors)
    return vectors


def get_max_num_attribute(data):
    data = [item for sublist in data for item in sublist]
    ranges = [(1, 9), (10, 24), (25, 54), (55, 58), (59, 73),(74,79),(80,94), (95, 99), (100, 105), (106, 120), (121,135), (136, 149), (150, 152), (153, 167), (168, 182), (183, 197), (198, 212), (213, 217), (218, 222), (223, 236), (237, 240), (241, 244), (245, 248), (249, 263), (264, 278), (279, 293),(294,308),(309,312)]
    most_common_numbers = []
    for r_start, r_end in ranges:
        r_start=r_start-1
        r_end=r_end-1
        count = Counter()
        for num in data:
            if r_start <= num <= r_end:
                count[num] += 1
        if count:
            most_common_number = count.most_common(1)[0][0]
        else:
            most_common_number = None
        most_common_numbers.append(most_common_number)
        most_common_numbers = [num for num in most_common_numbers if num is not None]
    return  most_common_numbers


def get_classid_attributeid_dict(root_1,root_2,root_3,root_4):
    index_classid_dict = {}
    with open(root_1, 'r') as file:
        for line in file:
            # 读取每一行，并根据空格分割
            data = line.split()
            index = int(data[0]) - 1
            class_id = int(data[1]) - 1
            index_classid_dict[index] = class_id

    index_attribute_dict = defaultdict(list)
    with open(root_2, 'r') as file:
        for line in file:
            data = line.split()
            category_id = int(data[0]) - 1
            attribute_id = int(data[1]) - 1
            p = int(data[2])
            if p == 1:
                index_attribute_dict[category_id].append(attribute_id)
        index_attribute_dict = dict(index_attribute_dict)

    attributeid_attribute_dict = {}
    with open(root_3, 'r') as file:
        for line in file:
            data = line.split()
            index = int(data[0]) - 1
            attributeid = data[1]
            attributeid_attribute_dict[index] = attributeid

    classid_class_dict = {}
    class_name_list = []
    with open(root_4, 'r') as file:
        for line in file:
            # 读取每一行，并根据空格分割
            data = line.split()
            classid = int(data[0]) - 1
            classname = data[1]
            classid_class_dict[classid] = classname
            class_name_list.append(classname)
    class_name_list = [re.sub(r'\d+\.?', '', item) for item in class_name_list]

    classid_all_attribute_dict = defaultdict(list)
    for index, class_id in index_classid_dict.items():
        classid_all_attribute_dict[class_id].append(index_attribute_dict[index])
    classid_all_attribute_dict = dict(classid_all_attribute_dict)

    classid_attributeid_dict = {}
    for class_id in range(200):
        data = classid_all_attribute_dict[class_id]
        index = get_max_num_attribute(data)
        classid_attributeid_dict[class_id] = index

    train_attributeid = []
    for i in range(100):
        train_attributeid.extend(classid_attributeid_dict[i])
    train_attributeid = set(train_attributeid)

    test_attributeid = []
    for i in range(100, 200):
        test_attributeid.extend(classid_attributeid_dict[i])
    test_attributeid = set(test_attributeid)

    final_attributeid = train_attributeid
    final_attribute = [attributeid_attribute_dict[attr_id] for attr_id in final_attributeid]
    final_attribute_attributeid_dict = {value: index for index, value in enumerate(final_attribute)}

    final_classid_attributeid_dict = defaultdict(list)
    for syn in range(200):
        attrs = classid_attributeid_dict[syn]
        for idx in attrs:
            if idx in final_attributeid:
                final_classid_attributeid_dict[syn].append(
                    final_attribute_attributeid_dict[attributeid_attribute_dict[idx]])

    final_classid_attributeid_dict = dict(final_classid_attributeid_dict)

    return final_classid_attributeid_dict, final_attribute,class_name_list


if __name__ == '__main__':
    root_image_class_labels='your_image_class_labels.txt'
    root_image_attribute_labels='your_image_attribute_labels.txt'
    root_attributes='your_attributes.txt'
    root_classes='your_classes.txt'
    glove_pth = 'your_glove_pth'

    classid_attributeid_dict,all_attribute_name,all_class_name=get_classid_attributeid_dict(root_image_class_labels, root_image_attribute_labels, root_attributes,root_classes)

    base_classid_attributeid_dict = dict(islice(classid_attributeid_dict.items(), 100))
    novel_classid_attributeid_dict = dict(list(classid_attributeid_dict.items())[-100:])
    base_attributeid_classid_dict = swap_dict_keys_values(base_classid_attributeid_dict)
    attribute_vectors = get_glove(glove_pth, all_attribute_name)
    class_vectors = get_glove(glove_pth, all_class_name)

    obj = {}
    obj['attribute_vectors'] = attribute_vectors
    obj['class_vectors'] = class_vectors
    obj['base_attributeid_classid_dict'] = base_attributeid_classid_dict
    obj['base_classid_attributeid_dict'] = base_classid_attributeid_dict
    obj['novel_classid_attributeid_dict'] = novel_classid_attributeid_dict
    obj['classid_attributeid_dict'] = classid_attributeid_dict

    output = '../cub200_part_prior_train.pickle'
    with open(output, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


