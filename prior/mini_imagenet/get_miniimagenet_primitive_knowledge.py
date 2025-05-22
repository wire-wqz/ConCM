from nltk.corpus import wordnet as wn
from prior.glove import GloVe
import pickle
import torch
import pandas as pd
from itertools import islice


def getnode(x):
    return wn.synset_from_pos_and_offset('n', int(x[1:]))

def getwnid(u):
    s = str(u.offset())
    return 'n' + (8 - len(s)) * '0' + s

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

def make_attribute_node(all_class_name, train_class_name, test_class_name):
    len_class=len(all_class_name)
    all_attrbute = []
    class_attrbute_dict = {}

    for i in range(len_class):
        class_attrbute_dict[all_class_name[i]] = []
        have_attri = False
        for paths in all_class_name[i].hypernym_paths():
            for sys in paths:
                parts = sys.part_meronyms()
                #parts = sys.substance_meronyms()    #
                all_attrbute.extend(parts)
                class_attrbute_dict[all_class_name[i]].extend(parts)
                if len(parts) != 0:
                    have_attri = True
        if have_attri:
            print('number {}: {}'.format(i, 'attribute'))
        else:
            print('number {}: {}'.format(i, 'no attribute'))
        class_attrbute_dict[all_class_name[i]] = list(set(class_attrbute_dict[all_class_name[i]]))
    all_attrbute = list(set(all_attrbute))

    # 获得每个数据集下的属性
    train_attribute = []
    for i in range(len(train_class_name)):
        train_attribute.extend(class_attrbute_dict[train_class_name[i]])
    train_attribute = list(set(train_attribute))

    test_attribute = []
    for i in range(len(test_class_name)):
        test_attribute.extend(class_attrbute_dict[test_class_name[i]])
    test_attribute = list(set(test_attribute))


    train_attribute_rm=[]
    test_attribute_rm=[]
    for attr in train_attribute:
        if attr in test_attribute:
            train_attribute_rm.append(attr)

    all_attribute_name = train_attribute
    print(len(all_attribute_name))

    class_attribute_dict_rm={}
    for syn in all_class_name:
        attrs = class_attrbute_dict[syn]
        class_attribute_dict_rm[syn] = [attr for attr in attrs if attr in all_attribute_name]
    return all_attribute_name, class_attribute_dict_rm


def get_glove(glove_pth,all_class_id):
    glove = GloVe(glove_pth)
    vectors = []
    print("all_class_id")
    print(all_class_id)
    num = 0
    for wnid in all_class_id:
        lemma_names=getnode(wnid).lemma_names()
        vectors.append(glove[lemma_names])
        if torch.sum(torch.abs(vectors[-1])) == 0:
            print('wnid: {}，{}'.format(wnid, getnode(wnid).lemma_names()))
            num+=1
    print(num)
    vectors = torch.stack(vectors)
    return vectors





if __name__ == '__main__':
    dataroot='your_train.csv'
    glove_pth='your_glove_pth'

    df = pd.read_csv(dataroot)
    all_class_id = list(dict.fromkeys(df['label']))
    all_class_name = list(map(getnode, all_class_id))
    train_class_name=all_class_name[0:60]
    test_class_name = all_class_name[60:]

    all_attribute_name, class_attribute_dict = make_attribute_node(all_class_name, train_class_name, test_class_name)
    attribute_attributeid_dict = {index: value for value,index in enumerate(all_attribute_name)}
    classid_attributeid_dict={}
    for index,(key, value_list) in enumerate(class_attribute_dict.items()):
        classid_attributeid_dict[index]=[attribute_attributeid_dict[attr] for attr in value_list]

    base_classid_attributeid_dict = dict(islice(classid_attributeid_dict.items(), 60))
    novel_classid_attributeid_dict = dict(list(classid_attributeid_dict.items())[-40:])
    base_attributeid_classid_dict=swap_dict_keys_values(base_classid_attributeid_dict)
    class_vectors=get_glove(glove_pth,list(map(getwnid, all_class_name)))
    attribute_vectors =get_glove(glove_pth, list(map(getwnid, all_attribute_name)))

    obj = {}
    obj['attribute_vectors'] = attribute_vectors
    obj['class_vectors'] = class_vectors
    obj['base_attributeid_classid_dict'] = base_attributeid_classid_dict
    obj['base_classid_attributeid_dict'] = base_classid_attributeid_dict
    obj['novel_classid_attributeid_dict'] = novel_classid_attributeid_dict
    obj['classid_attributeid_dict'] = classid_attributeid_dict

    output = '../mini_imagenet_part_prior_train.pickle'
    with open(output, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


