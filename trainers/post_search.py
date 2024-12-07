import json
import clip
import torch
import torch.nn.functional as F
import os.path as osp
import scipy.io as sio
from collections import defaultdict
from trainers.zeroshot import select_max_variance_view, attention_weighted_sum, variance_sort, pca_max_view, pca_sort, compute_mahalanobis_distance, compute_m_logit, batch_covaricance, batch_inv_covariance, compute_m_logit_simple, standardize, batch_kmeans, select_dominant_features,batch_pca, attention_optimize_representation, knn_pca,LOF
import os
class_names = {
    'ModelNet40': ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'],
    'ScanObjectNN': ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet'],
    "ModelNet10": ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
}

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def textual_encoder(cfg, clip_model, searched_prompt=None):
    """Encoding prompts.
    """
    prompt = searched_prompt
    prompt_token = torch.cat([clip.tokenize(p) for p in prompt]).cuda()
    text_feat10 = clip_model.encode_text(prompt_token).repeat(1, cfg.MODEL.PROJECT.NUM_VIEWS)
    text_feat1 = clip_model.encode_text(prompt_token)
    return text_feat10, text_feat1


def encode_prompt_lib(clip_model, cfg, dataset=''):
    """Encoding GPT-3 generated text to a feature library.
    """
    save_path = 'prompts/{}_{}_text_feat_lib.mat'.format(dataset, cfg.MODEL.BACKBONE.NAME2)
    if osp.exists(save_path):
        return
    
    print('Encoding prompt...')
    f = open('prompts/{}_1000.json'.format(dataset))
    gpt_sents = json.load(f)
    
    text_feat_lib = {}
    for key in gpt_sents.keys():
        temp_list = []
        for jj in range(len(gpt_sents[key])):
            prompt_token = clip.tokenize(gpt_sents[key][jj]).cuda()
            text_feat = clip_model.encode_text(prompt_token)
            temp_list.append(text_feat)
        text_feat_lib[key] = torch.cat(temp_list).cpu().numpy().squeeze()
    sio.savemat(save_path, text_feat_lib)
    print('End encoding prompt.')
    return


def encode_prompt_lib50(clip_model, cfg, dataset=''):
    """Encoding GPT-3 generated text to a feature library.
    """
    save_path = 'prompts/{}_{}_text_feat_lib50.mat'.format(dataset, cfg.MODEL.BACKBONE.NAME2)
    if osp.exists(save_path):
        return

    print('Encoding prompt...')
    f = open('prompts/{}_best_prompts.json'.format(dataset))
    gpt_sents = json.load(f)

    text_feat_lib = {}
    for key in gpt_sents.keys():
        temp_list = []
        for jj in range(len(gpt_sents[key])):
            prompt_token = clip.tokenize(gpt_sents[key][jj]).cuda()
            text_feat = clip_model.encode_text(prompt_token)
            temp_list.append(text_feat)
        text_feat_lib[key] = torch.cat(temp_list).cpu().numpy().squeeze()
    sio.savemat(save_path, text_feat_lib)
    print('End encoding prompt.')
    return


def save_dict_with_json(dict_data, filename):
    with open(filename, "w") as file:
        json.dump(dict_data, file, indent=4)


def load_dict_with_json(filename):
    with open(filename, "r") as file:
        return json.load(file)


def random_textual_replace(cfg, prompt_feat=None, c_i=None, sent_feat=None):
    """Replace a category prompt from the pre-extracted text feature library.
    """
    temp_prompt_feat = prompt_feat.clone()
    #temp_prompt_feat[c_i, :] = torch.Tensor(sent_feat).float().cuda().repeat(1, cfg.MODEL.PROJECT.NUM_VIEWS)
    temp_prompt_feat[c_i, :] = torch.Tensor(sent_feat).float().cuda()
    return temp_prompt_feat
    
    
def read_prompts(cfg, dataset='modelnet40'):
    f = open('prompts/{}_1000.json'.format(dataset))
    #f = open('prompts/{}_1000_color.json'.format(dataset))
    data = json.load(f)
    txt_feat = sio.loadmat('prompts/{}_{}_text_feat_lib.mat'.format(dataset, cfg.MODEL.BACKBONE.NAME2))
    return data, txt_feat

def read_prompts50(cfg, dataset='modelnet40'):
    f = open('prompts/{}_best_prompts.json'.format(dataset))
    #f = open('prompts/{}_1000_color.json'.format(dataset))
    data = json.load(f)
    txt_feat = sio.loadmat('prompts/{}_{}_text_feat_lib50.mat'.format(dataset, cfg.MODEL.BACKBONE.NAME2))
    return data, txt_feat


@torch.no_grad()
def search_prompt_zs(cfg, vweights, image_feature=None, searched_prompt=None, prompt_lib=None):

    print("\n***** Searching for prompts *****")
    
    labels = torch.load(osp.join(cfg.OUTPUT_DIR, "labels.pt"))

    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME)
    clip_model.eval()
    all_classes = class_names[cfg.DATASET.NAME]
    text_feat10, text_feat1 = textual_encoder(cfg, clip_model, searched_prompt=searched_prompt)
    text_feat10 = text_feat10 / text_feat10.norm(dim=-1, keepdim=True)
    text_feat1 = text_feat1 / text_feat1.norm(dim=-1, keepdim=True)
    # Encoding all GPT generated prompt.
    #file = 'prompts/{}_{}_text_feat_lib50.mat'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)
    file = 'prompts/{}_{}_text_feat_lib.mat'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)

    if not osp.exists(file):
        #--------------用哪个存储的feature
        #encode_prompt_lib50(clip_model, cfg, dataset=cfg.DATASET.NAME.lower())
        encode_prompt_lib(clip_model, cfg, dataset=cfg.DATASET.NAME.lower())
    if image_feature is None:
        image_feat = torch.load(osp.join(cfg.OUTPUT_DIR, "features.pt"))
        # image_feat = torch.load(osp.join(cfg.OUTPUT_DIR, "features20.pt"))
    else:
        image_feat = image_feature
    view_weights = torch.tensor(vweights).cuda()
    #image_feat_w = image_feat.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS, cfg.MODEL.BACKBONE.CHANNEL) * view_weights.reshape(1, -1, 1)
    image_feat_w = image_feat.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS, cfg.MODEL.BACKBONE.CHANNEL)
    #means, select_view_image = select_max_variance_view(image_feat_w)
    #attention_select_view_image = attention_weighted_sum(means, image_feat_w)
    #attention_select_view_image = attention_select_view_image.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL)

    #-----------------------------------------------------替换为kmeans加原型网络的思路，那就再找找原型网络的改进方法，改进一下
    # pca_features = pca_max_view(image_feat_w)
    # attention_pca = attention_weighted_sum(pca_features, image_feat_w)


    # -----------------------------------------------------------kmeans + 原型网络思路
    # centroids, kmean_labels = batch_kmeans(image_feat_w, 2)
    # selected_centroids, selected_features = select_dominant_features(image_feat_w, centroids, kmean_labels)
    # attention_pca = attention_weighted_sum(selected_centroids, image_feat_w)

    #attention_pca = attention_weighted_sum(selected_centroids, selected_features)
    # 一会试试image_feat_w看看结果。


    #------------------------------------------------------------kmeans + pca
    # centroids, kmean_labels = batch_kmeans(image_feat_w, 2)
    # selected_centroids, selected_features = select_dominant_features(image_feat_w, centroids, kmean_labels)
    # pca_features = knn_pca(selected_features)
    # attention_pca = attention_weighted_sum(pca_features, image_feat_w)


    #===============================================================LOF + pca
    selected_features = LOF(image_feat_w)
    pca_features = knn_pca(selected_features)
    attention_pca = attention_weighted_sum(pca_features, image_feat_w)

    # attention_pca = attention_weighted_sum(pca_features, selected_features)
    # 一会尝试一下根据selected features计算attention。

    #attention_means = pca_sort(image_feat_w, pca_features)
    #attention_means = attention_means * view_weights.reshape(1, -1, 1).type(torch.float16)
    #attention_pca = attention_pca.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL).type(clip_model.dtype)
    #indice = variance_sort(image_feat_w)
    #image_feat_w = torch.gather(image_feat_w, dim=1, index=indice.unsqueeze(-1).expand(-1, -1, 512))
    #image_feat_w = image_feat_w * view_weights.reshape(1, -1, 1)
    #image_feat_w = image_feat_w.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL).type(clip_model.dtype)
    
    # Before search
    #logits1 = clip_model.logit_scale.exp() * image_feat @ text_feat10.t() * 1.0
    #logits2 = clip_model.logit_scale.exp() * attention_select_view_image @ text_feat.t() * 1.0
    logits3 = clip_model.logit_scale.exp() * attention_pca @ text_feat1.t() * 1.0
    logits3 = standardize(logits3)
    logits3 = F.softmax(logits3, dim=1)
    #logits = cfg.alpha*logits2 + logits3
    #print(logits3[0,:])

    # inv_path = f"{cfg.OUTPUT_DIR}/cov_inv.pth"
    # if os.path.exists(inv_path):
    #     cov_inv = torch.load(inv_path)
    #     # print("load sucess")
    # else:
    #     cov = batch_covaricance(image_feature)
    #     cov_inv = batch_inv_covariance(cov)
    #     print("compute cov_inv again")
    #     del cov

    #这里添加一个weighted representation，构建weighted 分布看看效果
    # 一会换成kmeans的聚类中心看看，效果会不会变好
    # weighted_representation = attention_optimize_representation(pca_features, image_feat_w)
    weighted_representation = attention_optimize_representation(attention_pca, image_feat_w)
    # weighted_representation = attention_optimize_representation(selected_centroids, image_feat_w)
    # cov = batch_covaricance(image_feature)
    cov = batch_covaricance(weighted_representation)
    cov_inv = batch_inv_covariance(cov)

    # 这个weighted_representation维度(16,10),没法加到里面去啊
    # logits_m = compute_m_logit_simple(text_feature=text_feat1, image_feature=image_feat_w, cov_inv=cov_inv)
    logits_m = compute_m_logit_simple(text_feature=text_feat1, image_feature=weighted_representation, cov_inv=cov_inv)
    logits = (1.0-cfg.alpha)*logits_m + cfg.alpha*logits3
    acc, _ = accuracy(logits, labels, topk=(1, 5))
    acc = (acc / image_feat.shape[0]) * 100
    print(f"=> Before search, zero-shot accuracy: {acc:.2f}")
    
    # Search for prompt for each category
    print("During search:")
    # --------------------用哪一个存储的text feature
    #gpt_sents, text_feat_lib = read_prompts50(cfg, dataset=cfg.DATASET.NAME.lower())
    gpt_sents, text_feat_lib = read_prompts(cfg, dataset=cfg.DATASET.NAME.lower())
    prompts = searched_prompt
    text_feat_ori = text_feat1.clone()
    best_acc = acc

    top_prompts = defaultdict(list)
    for kk in range(0, 4): #
        for ii in range(len(all_classes)):  
            for jj in range(len(gpt_sents[all_classes[ii]])):
                
                text_feat = random_textual_replace(cfg, prompt_feat=text_feat_ori, c_i=ii, sent_feat=text_feat_lib[all_classes[ii]][jj,:])
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                
                #logits1 = clip_model.logit_scale.exp() * image_feat @ text_feat10.t() * 1.0
                #logits2 = clip_model.logit_scale.exp() * attention_select_view_image @ text_feat.t() * 1.0
                logits3 = clip_model.logit_scale.exp() * attention_pca @ text_feat.t() * 1.0
                logits3 = standardize(logits3)
                logits3 = F.softmax(logits3, dim=1)
                #logits = cfg.alpha*logits2 + logits3
                #logits = cfg.alpha *logits2 + logits3

                # 这里使用weighted representation构建多变量高斯分布，用于去噪声
                # logits_m = compute_m_logit_simple(text_feature=text_feat, image_feature=image_feat_w, cov_inv=cov_inv)
                logits_m = compute_m_logit_simple(text_feature=text_feat, image_feature=weighted_representation, cov_inv=cov_inv)

                logits = (1-cfg.alpha)*logits_m + cfg.alpha*logits3
                acc, _ = accuracy(logits, labels, topk=(1, 5))
                acc = (acc / image_feat.shape[0]) * 100
                #print(f"num-{(ii) * (jj)} :  {acc}")
                if acc > best_acc:
                    prompts[ii] = gpt_sents[all_classes[ii]][jj]
                    #text_feat_ori[ii, :] = torch.Tensor(text_feat_lib[all_classes[ii]][jj]).cuda().repeat(1, cfg.MODEL.PROJECT.NUM_VIEWS)
                    text_feat_ori[ii, :] = torch.Tensor(text_feat_lib[all_classes[ii]][jj]).cuda()
                    print('New best accuracy: {:.2f}, i-th class: {}, j-th sentence: {}'.format(acc, ii, jj))
                    best_acc = acc
                '''
                # ---获取最好的prompt 200个不重复 开始
                pp = gpt_sents[all_classes[ii]][jj]
                top_prompts[all_classes[ii]].append((acc, pp))
                top_prompts[all_classes[ii]].sort(reverse=True, key=lambda x: x[0])
                unique_top_prompts = {key: list(set(value)) for key, value in top_prompts.items()}
                if len(unique_top_prompts[all_classes[ii]]) > 200:
                    unique_top_prompts[all_classes[ii]] = unique_top_prompts[all_classes[ii]][:200]
                    top_prompts[all_classes[ii]] = unique_top_prompts[all_classes[ii]]
    best_prompts = defaultdict(list)
    for ii in all_classes:
        for jj in range(200):
            best_prompts[ii].append(top_prompts[ii][jj][1])
    save_dict_with_json(best_prompts, f"./prompts/{cfg.DATASET.NAME.lower()}_best_prompts.json")
    #------获取最好的prompt 结束
    '''
    print('\nThe best prompt is: ')
    print(prompts)
    
    print('\nAfter prompt search, zero-shot accuracy: {}'.format(best_acc))
    return prompts, image_feat


@torch.no_grad()
def search_weights_zs(cfg, prompt, vweights, image_feature=None, ):
    print("\n***** Searching for view weights *****")
    if image_feature is None:
        print("\n***** Loading saved features *****")
        image_feat = torch.load(osp.join(cfg.OUTPUT_DIR, "features.pt"))
    else: 
        image_feat = image_feature
    labels = torch.load(osp.join(cfg.OUTPUT_DIR, "labels.pt"))

    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME)
    clip_model.eval()
    text_feat = textual_encoder(cfg, clip_model, searched_prompt=prompt)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    
    view_weights = torch.tensor(vweights).cuda()

    # # Before search
    logits = clip_model.logit_scale.exp() * image_feat @ text_feat.t() * 1.0
    acc, _ = accuracy(logits, labels, topk=(1, 5))
    acc = (acc / image_feat.shape[0]) * 100
    print(f"=> Before search, zero-shot accuracy: {acc:.2f}")

    # Search
    print("During search:")
    best_acc = acc
    vw = vweights
    # Search_time can be modulated in the config for faster search
    search_time, search_range = cfg.SEARCH.TIME, cfg.SEARCH.RANGE
    search_list = [(i + 1) * search_range / search_time  for i in range(search_time)]
    for a in search_list:
        for b in search_list:
            for c in search_list:
                for d in search_list:
                    for e in search_list:
                        for f in search_list:
                            for g in search_list:
                                # Reweight different views
                                view_weights = torch.tensor([0.75, 0.75, 0.75, a, b, c, d, e, f, g]).cuda()
                                image_feat_w = image_feat.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS, cfg.MODEL.BACKBONE.CHANNEL) * view_weights.reshape(1, -1, 1)
                                image_feat_w = image_feat_w.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL).type(clip_model.dtype)
                                
                                logits = clip_model.logit_scale.exp() * image_feat_w @ text_feat.t() * 1.0                                       
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = (acc / image_feat.shape[0]) * 100

                                if acc > best_acc:
                                    vw = [0.75, 0.75, 0.75, a, b, c, d, e, f, g]
                                    print('New best accuracy: {:.2f}, view weights: {}'.format(acc, vw))
                                    best_acc = acc

    print(f"=> After view weight search, zero-shot accuracy: {best_acc:.2f}")
    return vw



