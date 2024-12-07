import os.path

import torch
from clip import clip
import torch.nn as nn
import torch.nn.functional as F
from trainers.best_param import best_prompt_weight
from trainers.mv_utils_zs import Realistic_Projection
from dassl.engine import TRAINER_REGISTRY, TrainerX
from PIL import Image
from torchvision import transforms
from colorizers import *
import numpy as np
import torchvision.transforms.functional as ft
import os.path as osp
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plot
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

class Textual_Encoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        prompts = best_prompt_weight['{}_{}_test_prompts'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        text_feat_repeat = self.clip_model.encode_text(prompts).repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS)
        # 这里不再是重复10次了，而是直接用一次的。
        # scanobject (15, 512)
        text_feat = self.clip_model.encode_text(prompts)
        return text_feat_repeat, text_feat


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())
    return model

def select_max_variance_view(view_features):

    batch_size=16
    # Step 1: 计算每个样本的视角均值
    # (batch_size, num_view, feature_dim) -> (batch_size, feature_dim)
    means = view_features.mean(dim=1)

    # Step 2: 计算每个样本每个视角特征的方差
    # (batch_size, num_view, feature_dim) - (batch_size, 1, feature_dim)
    # 这里需要利用broadcast机制来计算每个视角与均值之间的差值
    variance = ((view_features - means.unsqueeze(1)) ** 2).mean(dim=2)  # (batch_size, num_view)

    # Step 3: 选择方差最大的视角
    # 对每个样本，找出方差最大的视角的索引
    _, max_variance_indices = variance.max(dim=1)  # (batch_size,)
    _, min_variance_indices = variance.min(dim=1)
    # Step 4: 根据最大方差索引选择特征
    # Gather the features for each sample based on the maximum variance index
    selected_features = view_features[torch.arange(view_features.size(0)), min_variance_indices]  # (batch_size, feature_dim)
    # print(selected_features.shape)
    #return selected_features
    return means, selected_features

def variance_sort(view_features):
    batch_size = 16
    # Step 1: 计算每个样本的视角均值
    # (batch_size, num_view, feature_dim) -> (batch_size, feature_dim)
    means = view_features.mean(dim=1)

    # Step 2: 计算每个样本每个视角特征的方差
    # (batch_size, num_view, feature_dim) - (batch_size, 1, feature_dim)
    # 这里需要利用broadcast机制来计算每个视角与均值之间的差值
    variance = ((view_features - means.unsqueeze(1)) ** 2).mean(dim=2)  # (batch_size, num_view)
    sort_indices = torch.argsort(variance, dim=-1)
    return sort_indices

def attention_weighted_sum(reference_features, view_features):
    """
    对每个样本的特征和其对应的视角特征计算无参数的注意力加权和
    :param reference_features: 输入形状为 (16, 512) 的特征张量
    :param view_features: 输入形状为 (16, 10, 512) 的视角特征张量
    :return: 输出形状为 (16, 512) 的注意力加权特征张量
    """
    # 计算相似度（点积），结果形状为 (16, 10)
    # reference_features: (16, 512) -> 需要扩展维度
    # view_features: (16, 10, 512)
    if reference_features.dim() == 3:
        reference_features = reference_features.squeeze(1)
    reference_features = reference_features / reference_features.norm(dim=-1, keepdim=True)
    view_features = view_features / view_features.norm(dim=-1, keepdim=True)
    similarity = torch.bmm(view_features, reference_features.unsqueeze(2)).squeeze(2)  # (16, 10)
    beta = 10

    similarity_enhance = ((-1) * (beta - beta * similarity)).exp()
    # 计算注意力权重（使用 softmax）

    #这里决定是否使用enhance attention-------------------------------------------------------
    #attention_weights = F.softmax(similarity_enhance, dim=1)  # (16, 10)
    attention_weights = F.softmax(similarity, dim=1)
    #attention_weights = attention_weights / torch.sqrt(torch.tensor(512.0))
    # 这里注意力加个根号下512试试。
    # 对视角特征进行加权和
    # 计算加权后的特征，结果形状为 (16, 512)
    # 这里是通过矩阵乘法得到的，这里我要通过broadcast到(16,10,512)，然后元素乘积得到。
    #weighted_sum = torch.bmm(attention_weights.unsqueeze(1), view_features).squeeze(1)  # (16, 512)
    attention_weights_expand = attention_weights.unsqueeze(-1).expand(-1, -1, 512)
    #attention_weights_expand = attention_weights_expand / torch.sqrt(torch.tensor(512.0))
    weighted_sum = view_features * attention_weights_expand
    weighted_sum = torch.sum(weighted_sum, dim=1)
    # 这里是直接sum，取均值会怎样呢
    return weighted_sum

def attention_optimize_representation(reference_features, view_features):
    if reference_features.dim() == 3:
        reference_features = reference_features.squeeze(1)
    reference_features = reference_features / reference_features.norm(dim=-1, keepdim=True)
    view_features = view_features / view_features.norm(dim=-1, keepdim=True)
    similarity = torch.bmm(view_features, reference_features.unsqueeze(2)).squeeze(2)  # (16, 10)
    attention_weights = F.softmax(similarity, dim=1)  # (16, 10)

    attention_weights_expand = attention_weights.unsqueeze(-1).expand(-1, -1, 512)

    # weighted_representation = view_features * (1 - attention_weights_expand) + reference_features.unsqueeze(1) * attention_weights_expand
    # weighted_representation = view_features + torch.exp((1 - attention_weights_expand))*(reference_features.unsqueeze(1) - view_features)
    # 目前最好的
    weighted_representation = view_features + (1 - attention_weights_expand) * (reference_features.unsqueeze(1) - view_features)
    # weighted_representation = view_features + torch.exp((-1.0 * attention_weights_expand)) * (reference_features.unsqueeze(1) - view_features)
    # 先看看，是增强幅度增加精度，还是减少幅度增加精度
    # return attention_weights
    return weighted_representation



def image_download(image):
    image_reshape = image.reshape(-1, 10, 3, 224, 224)
    # random = torch.randint(0, 16, ())
    to_pil = transforms.ToPILImage()
    for j in range(16):
        for i in range(10):
            #img_pil = to_pil(image_reshape[random, i, :, :, :])
            img_pil = to_pil(image_reshape[j, i, :, :, :])
            #img_pil.save(f"./download/img_contrastive_{random}_view_{i}.jpg")
            #img_pil.save(f"./download_visual/{j}_img_{i}_view.jpg")
            img_pil.save(f"./download_distribution2/{j}_img_{i}_view.jpg")

def color_image(image):
    colorizer_eccv16 = eccv16(pretrained=True).eval()
    colorizer_eccv16.cuda()
    # 先转pil image，然后利用colorization中的函数对
    # 其处理就好了。上色之后再transform转为pytorch中的tensor
    to_pil = transforms.ToPILImage()
    transforms_pil = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_list = []
    for i in range(image.shape[0]):
        img = np.asarray(to_pil(image[i, :, :, :]))
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        tens_l_rs = tens_l_rs.cuda()
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        pil_out_img_eccv16 = Image.fromarray(out_img_eccv16.astype(np.uint8)).convert("RGB")
        tensor_img = transforms_pil(pil_out_img_eccv16)
        tensor_list.append(tensor_img)

    tensor_image = torch.stack(tensor_list)
    return tensor_image

# 直接拉成2d来做pca，然后再拉成3d视角版本。
def pca_max_view(view_features):
    #feature_reshape = view_features.view(view_features.size(0)*10, 512)
    # feature_mean = view_features.mean(dim=1, keepdim=True)
    # feature_center = view_features - feature_mean
    # cov_matrix = torch.matmul(feature_center.transpose(1,2), feature_center) / (feature_center.size(1)-1)
    # _, s, _ = torch.svd(cov_matrix.float())
    # variance = s**2
    # #view_variance = variance.view(view_features.size(0), 10)
    # _, max_variance_indice = variance.max(dim=1)
    # select_features = view_features[torch.arange(view_features.size(0)), max_variance_indice]
    ##data_centered = view_features - view_features.mean(dim=1, keepdim=True)  # (16, 10, 512)

    # 计算协方差矩阵
    ##covariance_matrix = torch.bmm(data_centered.transpose(1, 2), data_centered) / (10 - 1)  # (16, 512, 512)

    # 计算特征值和特征向量
    ##eigvals, eigvecs = torch.linalg.eigh(covariance_matrix.float())  # (16, 512), (16, 512, 512)

    # 选择最大的特征值对应的特征向量
    ##principal_components = eigvecs[torch.arange(view_features.size(0)), :, -1]  # (16, 512)
    #principal_components = principal_components.permute(0, 2, 1)
    ##return principal_components.type(torch.float16)
    #return select_features
    # 上面双#的代码，应该是有问题的，这里只是拿特征值最大的特征向量作为了主成分，按理来说，应该是拿这个特征向量再乘回去的
    # 并且，得到的特征值并不是最大的特征值，也不是最好的特征向量，这里的实现有问题。
    view_features = view_features.transpose(1, 2)  # 16,512,10
    data_centered = view_features - view_features.mean(dim=1, keepdim=True)  # 16, 512, 10
    covariance_matrix = torch.bmm(data_centered.transpose(1, 2), data_centered) / (data_centered.size(0) - 1)  # 16, 10, 10)
    eigvals, eigvecs = torch.linalg.eigh(covariance_matrix.float())  # 16, 10   (16, 10, 10)
    max_values, indices = torch.max(eigvals, dim=1)   # 16,  1
    #max_vectors = torch.gather(eigvecs, 1, indices.unsqueeze(-1).expand(-1, -1, eigvecs.size(2)))   # 16,1,10
    #max_vectors = eigvecs[torch.arange(eigvecs.size(0)).unsqueeze(1), indices.unsqueeze(1)].squeeze(1)
    max_vectors = eigvecs[torch.arange(view_features.size(0)), :, -1].unsqueeze(2)
    principal_components = torch.bmm(data_centered, max_vectors.type(torch.float16)).transpose(1, 2)   # 16, 1, 512
    #principal_components = principal_components.mean(dim=1, keepdim=False)
    principal_components = principal_components.squeeze(1)    # 16, 512
    principal_components = principal_components / principal_components.norm(dim=-1, keepdim=True)
    return principal_components.type(torch.float16)


def knn_pca(view_features):
    batch_size, num_views, feature_dim = view_features.shape
    output_features = []

    for i in range(batch_size):
        # Extract the i-th instance
        instance = view_features[i]

        # Remove zeroed views
        non_zero_views = instance[torch.any(instance != 0, dim=1)]


        # Transpose to shape (num_valid_views, feature_dim)
        transposed_views = non_zero_views.t()

        # Apply PCA
        pca = PCA(n_components=1)
        principal_components = pca.fit_transform(transposed_views.cpu().numpy())

        # Flatten the output to a 1D array
        feature_vector = torch.tensor(principal_components.flatten(), dtype=torch.float16, device="cuda")

        # Append the feature vector to the results
        output_features.append(feature_vector)

    # Concatenate all feature vectors to form the output tensor of shape (batch_size, feature_dim)
    return torch.stack(output_features)

# 这里需要再改进一下
def LOF(data):
    # 初始化 LOF 模型
    lof = LocalOutlierFactor(n_neighbors=4, contamination=0.2)  # contamination参数应根据实情调整
    # lof = IsolationForest(contamination=0.3)
    data_np = data.cpu().numpy()
    # 遍历批次中的每一个数据
    for i in range(data_np.shape[0]):
        # 获取每个数据的 10 个视角 (10, 512)
        sample_views = data_np[i]

        # 使用 LOF 检测离群点
        outlier_predictions = lof.fit_predict(sample_views)  # -1 为离群点，1 为正常点

        # 将离群点的特征置零
        for j in range(sample_views.shape[0]):
            if outlier_predictions[j] == -1:
                data_np[i, j, :] = 0  # 将该视角的特征置零

    # 将 numpy 数组转换回 PyTorch 张量
    data_gpu = torch.from_numpy(data_np).to("cuda")
    return data_gpu




def pca_sort(view_features, reference_features):
    similarity = torch.bmm(view_features, reference_features.unsqueeze(2)).squeeze(2)  # (16, 10)
    indice = torch.argsort(similarity, dim=-1, descending=True)
    # attention_weights = F.softmax(similarity, dim=1)
    # attention_weights = attention_weights / torch.sqrt(torch.tensor(512.0))
    # 这里注意力加个根号下512试试。
    # 对视角特征进行加权和
    # 计算加权后的特征，结果形状为 (16, 512)
    # 这里是通过矩阵乘法得到的，这里我要通过broadcast到(16,10,512)，然后元素乘积得到。
    # weighted_sum = torch.bmm(attention_weights.unsqueeze(1), view_features).squeeze(1)  # (16, 512)
    view_features = torch.gather(view_features, dim=1, index=indice.unsqueeze(-1).expand(-1, -1, 512))
    return view_features

#调整对比度的
def enhance_contrastive(image):
    contrast_factor = 0.7
    enhance_image = ft.adjust_contrast(image, contrast_factor)
    return enhance_image

# 批量计算协方差
def batch_covaricance(data):
    # 数据中心化
    mean = data.mean(dim=1, keepdim=True)
    centered_data = data - mean  # (batch, N, dim)

    # 计算协方差矩阵
    # 使用 torch.matmul 进行批量计算
    cov_matrices = (centered_data.transpose(1, 2) @ centered_data) / (data.size(1) - 1)

    return cov_matrices

def batch_inv_covariance(cov):
    traces = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    traces = traces.unsqueeze(1).unsqueeze(2)
    cov_inv = torch.linalg.pinv(cov+traces*torch.eye(cov.shape[-1]).cuda())
    return cov_inv

def compute_mahalanobis_distance(per_image_view_features, x):
    center_mean = per_image_view_features.mean(dim=0, keepdim=True)
    center_vecs = (per_image_view_features - center_mean)
    cov_inv = center_vecs.shape[1] * torch.linalg.pinv(
        (center_vecs.shape[0] - 1) * center_vecs.T.cov() + center_vecs.T.cov().trace() * torch.eye(
            center_vecs.shape[1]).cuda())

    diff = (x.unsqueeze(0) - center_mean).float()
    dist = torch.sqrt(torch.matmul(diff, cov_inv.float())@diff.t())
    return dist

def compute_m_logit_old(text_feature, image_feature):
    distances = torch.zeros(image_feature.shape[0], text_feature.shape[0]).cuda()
    for i in range(image_feature.shape[0]):
        for j in range(text_feature.shape[0]):
            distances[i,j] = -1.0 * compute_mahalanobis_distance(image_feature[i], text_feature[j])
    return distances

def compute_m_logit(text_feature, image_feature, cfg):
    # 添加存储cov_inv的代码，并且根据数据集名称进行存储和提取，如果已经有了对应的pth文件，就不用计算，
    # 直接提取，如果没有，就计算并存储
    inv_path = f"{cfg.OUTPUT_DIR}/cov_inv.pth"
    if os.path.exists(inv_path):
        cov_inv = torch.load(inv_path)
        #print("load sucess")
    else:
        cov = batch_covaricance(image_feature)
        cov_inv = batch_inv_covariance(cov)
        del cov
    center_mean = image_feature.mean(dim=1, keepdim=True)
    text_feature = text_feature.unsqueeze(0).repeat(center_mean.size(0), 1, 1)
    diff = (text_feature - center_mean).float()
    del text_feature
    dist = diff @ cov_inv.float()
    neg_one = torch.tensor(-1.0, dtype=torch.float32)
    distances = neg_one * torch.sqrt(torch.sum(dist * diff, dim=-1))
    del dist, diff
    return distances
    # distances = []
    # for i in range(image_feature.shape[0]):
    #     diff = (text_feature - center_mean[i]).float()
    #     distance = torch.sqrt(torch.sum(diff @ cov_inv[i].float() * diff, dim=-1))
    #     #distance_diag = distance.diagonal()
    #     distances.append(-1.0 * distance)
    # distances = torch.stack(distances).cuda()
    # 用einsum实现的这个效果是错误的
    # 这里可以直接换成元素乘积，然后在一个维度加和，速度更快应该

def standardize(matrix):
    mean = matrix.mean(dim=1)  # 计算每一列的均值
    std = matrix.std(dim=1)    # 计算每一列的标准差
    standardized_matrix = (matrix - mean.unsqueeze(1)) / std.unsqueeze(1)  # 标准化
    return standardized_matrix

def compute_m_logit_simple(text_feature, image_feature, cov_inv):
    center_mean = image_feature.mean(dim=1, keepdim=True)
    text_feature = text_feature.unsqueeze(0).repeat(center_mean.size(0), 1, 1)
    diff = (text_feature - center_mean).float()
    del text_feature
    dist = diff @ cov_inv.float()
    neg_one = torch.tensor(-1.0, dtype=torch.float32)
    distances = neg_one * torch.sqrt(torch.sum(dist * diff, dim=-1))
    del dist, diff
    distances = standardize(distances)
    distances = F.softmax(distances, dim=1)
    return distances

def cosine_similarity_matrix(p):
    # 计算余弦相似度
    # 首先计算每个样本的 L2 范数
    p_norm = torch.norm(p, dim=2, keepdim=True)  # 形状: (batch_size, num_samples, 1)

    # 计算归一化后的特征向量
    p_normalized = p / p_norm  # 广播: (batch_size, num_samples, num_features)

    # 计算余弦相似度矩阵
    cosine_similarity_matrices = torch.bmm(p_normalized, p_normalized.transpose(1, 2))
    return cosine_similarity_matrices
    # 相似度范围-1到1

'''
# 这个fielder向量的聚类，是图的谱聚类的一种，那么是不是可以直接用聚类算法实现呢
def fielder_group(image_feature, gamma):
    # image_feature 16,10,512
    similarity_matirx = cosine_similarity_matrix(image_feature)
    A = (similarity_matirx > gamma) * similarity_matirx
    # 也可是试试top k的方法
    D = torch.diag_embed(A.sum(dim=2))
    # 计算拉普拉斯矩阵
    laplacian_matrices = D - A

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrices)

    # Fiedler 向量是对应于第二小特征值的特征向量
    # 由于特征值是按升序排序的，因此第二个特征值对应索引 1
    fiedler_vectors = eigenvectors[:, :, 1]
'''

def batch_kmeans(data, num_clusters, num_iters=100, tolerance=1e-4):
    """
    在批次级别上使用 PyTorch 实现 k-means 聚类。

    参数：
    - data: 输入数据，形状为 (batch_size, num_samples, num_features)
    - num_clusters: 聚类的数量
    - num_iters: 最大迭代次数
    - tolerance: 用于判断收敛的容许误差

    返回：
    - centroids: 聚类中心，形状为 (batch_size, num_clusters, num_features)
    - labels: 每个样本的聚类标签，形状为 (batch_size, num_samples)
    """
    batch_size, num_samples, num_features = data.shape
    # 随机初始化聚类中心
    indices = torch.randperm(num_samples)[:num_clusters]
    centroids = data[:, indices, :].clone()

    for i in range(num_iters):
        # 计算每个样本到每个中心的距离
        dists = ((data.unsqueeze(2) - centroids.unsqueeze(1)) ** 2).sum(
            dim=-1)  # (batch_size, num_samples, num_clusters)

        # 找到距离最近的中心
        labels = dists.argmin(dim=-1)  # (batch_size, num_samples)

        # 计算新的聚类中心
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            mask = (labels == k).unsqueeze(-1).float()  # (batch_size, num_samples, 1)
            new_centroids[:, k, :] = (data * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # 检查收敛
        centroid_shift = ((new_centroids - centroids) ** 2).sum()
        if centroid_shift < tolerance:
            break

        centroids = new_centroids

    return centroids, labels


def select_dominant_features(samples, centroids, labels, num_views=10):
    # labels(16,10)
    batch_size, view_count, feature_dim = samples.shape
    counts0 = (labels == 0).sum(dim=1)
    counts1 = (labels == 1).sum(dim=1)
    dominant_class = torch.where(counts0 >= counts1, 0, 1)
    # where对矩阵中的每个元素进行条件比较，true就是0，false就是1
    # 得到（16），每个样本中最大的聚类类别
    masks = labels == dominant_class.unsqueeze(1)
    mask_expanded = masks.unsqueeze(2).expand_as(samples)
    selected_features = samples * mask_expanded.half()
    selected_centroids = centroids[torch.arange(batch_size), dominant_class].unsqueeze(1)
    return selected_centroids, selected_features


# 这里要再想一下，因为pca压缩是压缩10那个维度，这里实现是压缩512，但是如果转置，又会导致一些问题
# 再想想，或者直接实现看看。
def batch_pca(image_feature):
    num_clusters = 2
    centroids, labels = batch_kmeans(image_feature, num_clusters)
    selected_centroids, selected_features = select_dominant_features(image_feature, centroids, labels)
    batch_size, num_views, feature_dim = selected_features.size()
    mask = selected_features.abs().sum(dim=2) > 0
    # (batch_size, num_views), true or false
    valid_counts = mask.sum(dim=1).half()
    # (batch_size,)每个sample可用的view数量
    mask = mask.unsqueeze(2)
    masked_features = selected_features * mask
    sum_features = masked_features.sum(dim=1, keepdim=True)
    mean_features = sum_features / valid_counts.view(-1, 1, 1)
    centered_features = masked_features - mean_features * mask
    # 非零的视角中心化

    # 计算协方差
    centered_features_masked = centered_features * mask
    cov_matrices = torch.bmm(centered_features_masked.transpose(1, 2), centered_features_masked) / (
                valid_counts.view(-1, 1, 1) - 1)

    # 使用linalg.eigh计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices.float())

    # 选择最大的特征值对应的特征向量
    top_eigenvectors = eigenvectors[torch.arange(masked_features.size(0)), :, -1:]  # (batch_size, feature_dim, num_components)

    # 获取每个样本的主成分
    principal_components = top_eigenvectors.transpose(1,2)
    return attention_weighted_sum(principal_components.half(), image_feature)


def visual_attention(image_feat_w, pca_feat):
    if pca_feat.dim() == 3:
        pca_feat = pca_feat.squeeze(1)
    pca_feat = pca_feat / pca_feat.norm(dim=-1, keepdim=True)
    image_feat_w = image_feat_w / image_feat_w.norm(dim=-1, keepdim=True)
    similarity = torch.bmm(image_feat_w, pca_feat.unsqueeze(2)).squeeze(2)
    # (16, 10)
    attention_weights = F.softmax(similarity, dim=1)
    attention_w = attention_weights.detach().cpu().numpy()
    image_feature = image_feat_w.detach().cpu().numpy()
    pca_feature = pca_feat.detach().cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42)
    image_f = tsne.fit_transform(image_feature.reshape(-1, 512))
    image_f = image_f.reshape(-1, 10, 2)
    #pca_f = tsne.fit_transform(pca_feature, perplexity=10)
    for i in range(16):

        #image_f = tsne.fit_transform(image_feature[i, :, :])
        # image_f (10,2)
        # pca_f (1,2)
        print(attention_w)
        sns.scatterplot(x=image_f[i,:, 0].flatten(), y=image_f[i,:, 1].flatten(), sizes=(20, 200), size=attention_w[i,:], legend=False, hue=[0,1,2,3,4,5,6,7,8,9],palette="pastel")
        # # 获取当前图例
        # handles, labels = plt.gca().get_legend_handles_labels()
        #
        # # 提取颜色图例（通常是前5个，具体取决于数据）
        # color_legend_handles = handles[:10]
        # color_legend_labels = labels[:10]
        # plt.legend([], [], frameon=False)
        # # 添加只包括颜色的图例
        # plt.legend(color_legend_handles, color_legend_labels, loc="best")

        for j, (x, y, s) in enumerate(zip(image_f[i, :, 0].flatten(), image_f[i, :, 1].flatten(), attention_w[i,:]), start=0):
            plt.text(x-0.1, y, f'{j}', fontsize=12, verticalalignment='center')


        plt.savefig(f"{i}_attention_visual.pdf", dpi=150, format="pdf")
        plt.show()

# def visual_distribution(text_feature, image_feature):
#     # text_feature (10,512), image_feature(16,10,512)
#     text_feature = text_feature.detach().cpu().numpy()
#     image_feature = image_feature.detach().cpu().numpy()
#     tsne = TSNE(n_components=2, random_state=42, perplexity=9)
#     image_f = tsne.fit_transform(image_feature.reshape(-1, 512))
#     text_f = tsne.fit_transform(text_feature)
#     image_f = image_f.reshape(-1, 10, 2)
#     # text_f (10, 2), image_f (16, 10, 2)
#     means = np.mean(image_f, axis=1)
#     # means (16,2)
#     covariances = np.array([np.cov(sample, rowvar=False) for sample in image_f])
#     # covarances (16,2,2)
#     classname=["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
#     for i in range(16):
#         sns.scatterplot(x=text_f[:, 0].flatten(), y=text_f[:, 1], hue=classname, palette="husl")
#         sns.kdeplot(x=image_f[i, :, 0].flatten(), y=image_f[i, :, 1], thresh=.1, color="black")
#         plt.legend()
#         plt.savefig(f"download_visual_distribution/{i}_distribution.pdf", dpi=150, format="pdf")
#         plt.show()
#         print("means=",means)
#         print("covariance", covariances)


# def visual_probility(logits_m, mean_probility):
#     # logit_m (16,10), mean_probility(16,10)
#     logits_m = logits_m.detach().cpu().numpy()
#     mean_probility = mean_probility.detach().cpu().numpy()
#     classname = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
#     for i in range(16):
#         sns.barplot(x=classname*2, y=logits_m[i,:].tolist()+mean_probility[i,:].tolist(), hue=["Mahalanobis"]*10+["Cosine"]*10)#, palette="muted")
#
#         # ax = plt.gca()
#         # hatches = ['/', '\\']
#         # for j, bar in enumerate(ax.patches):
#         #     hatch = hatches[i // 20]
#         #     bar.set_hatch(hatch)
#
#         plt.legend()
#         plt.savefig(f"download_visual_distribution/{i}_distribution.pdf", dpi=150, format="pdf")
#         plt.show()
#         print("logit_m:", logits_m[i])
#         print("mean_probility:", mean_probility)


def visual_probility(logits_m, mean_probility, label):
    # 将logits_m和mean_probility转为NumPy数组
    logits_m = logits_m.detach().cpu().numpy()
    mean_probility = mean_probility.detach().cpu().numpy()

    max_indice1 = np.argmax(logits_m, axis=1)
    max_indice2 = np.argmax(mean_probility, axis=1)
    if np.array_equal(max_indice1,max_indice2):
        return


    # 类别名称列表
    classname = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
    print("label:", label)
    for i in range(16):
        # 创建数据集
        data = {
            'Category': classname * 2,
            'Value': logits_m[i, :].tolist() + mean_probility[i, :].tolist(),
            'Type': ['Mahalanobis'] * 10 + ['Cosine'] * 10
        }

        data_means={
            'Category': classname,
            'Value': mean_probility[i, :].tolist(),
            'Type': ['Cosine'] * 10
        }
        # 创建DataFrame
        df = pd.DataFrame(data)
        df_means = pd.DataFrame(data_means)

        plt.figure()
        sns.barplot(x='Category', y='Value', hue='Type', data=df, ci=None)
        plt.legend()
        plt.xlabel('Category')
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.savefig(f"download_visual_distribution2/{i}_distribution.pdf", dpi=150, format="pdf")



        plt.figure()
        current_palette = ["#E1812B"]
        sns.barplot(x='Category', y='Value', hue='Type', data=df_means, palette=current_palette)
        plt.ylim(0.08, 0.12)
        plt.legend()
        plt.xlabel('Category')
        plt.xticks(rotation=45)
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.savefig(f"download_visual_distribution2/mean_{i}_distribution.pdf", dpi=150, format="pdf")

    np.savetxt("./download_visual_distribution2/logit_m.txt", logits_m, delimiter=',')
    np.savetxt("./download_visual_distribution2/mean_probability.txt", mean_probility, delimiter=',')





@TRAINER_REGISTRY.register()
class PointCLIPV2_ZS(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.cuda()

        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        
        text_feat10, text_feat1 = textual_encoder()
        self.text_feat1 = text_feat1 / text_feat1.norm(dim=-1, keepdim=True)
        self.text_feat10 = text_feat10 / text_feat10.norm(dim=-1, keepdim=True)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        # Realistic projection
        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        # Store features for post-search
        self.feat_store = []
        self.label_store = []
        self.cov_inv_store = []
        self.view_weights = torch.Tensor(best_prompt_weight['{}_{}_test_weights'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]).cuda()

    def real_proj(self, pc, imsize=224):
        img = self.get_img(pc).cuda()
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
        return img
    
    def model_inference(self, pc, label=None):
        
        with torch.no_grad():
            # Realistic Projection
            images = self.real_proj(pc)
            #images = enhance_contrastive(images)
            #images = color_image(images1)
            images = images.type(self.dtype)
            #images = images.cuda()
            # images shape (160, 3, 224, 224)
            # download image 看看是什么
            # 而且图片好像也没有preprocess的处理啊
            image_download(images)
            # Image features
            if len(images.shape) == 3:
                images = images.unsqueeze(0)
            image_feat = self.visual_encoder(images)
            
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            # 这里就已经进行归一化了，所以计算的就是余弦相似度
            #image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1).type(self.dtype)
            # 去掉提前搜索得到的权重
            image_feat_w = image_feat.reshape(-1, self.num_views, self.channel)
            pca_features = pca_max_view(image_feat_w)
            attention_pca = attention_weighted_sum(pca_features, image_feat_w)

            # 这里实现对应的可视化的代码
            # visual_attention(image_feat_w, pca_features)





            #means, select_view_image = select_max_variance_view(image_feat_w)

            #attention_means = pca_sort(image_feat_w, pca_features)
            #attention_means=attention_means*self.view_weights.reshape(1, -1, 1).type(self.dtype)
            #attention_select_view_image = attention_weighted_sum(means, image_feat_w)
            # dimension (16,10,512)
            #image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype)
            # 得到对应方差的indice
            #indice = variance_sort(image_feat_w)
            #image_feat_w = torch.gather(image_feat_w, dim=1, index=indice.unsqueeze(-1).expand(-1, -1, 512))
            #image_feat_w = image_feat_w * self.view_weights.reshape(1, -1, 1).type(self.dtype)
            #image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype)
            # Store for zero-shot
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)
            self.feat_store.append(image_feat)
            self.label_store.append(label)


            weighted_representation = attention_optimize_representation(attention_pca, image_feat_w)

            # 这里实现可视化分布的代码
            # visual_distribution(self.text_feat1, weighted_representation)

            temp_cov = batch_covaricance(weighted_representation)
            temp_cov_inv = batch_inv_covariance(temp_cov)
            del temp_cov
            self.cov_inv_store.append(temp_cov_inv)
            #attention_select_view_image = attention_select_view_image.reshape(-1, self.num_views * self.channel)
            #attention_pca = attention_pca.reshape(-1, self.num_views * self.channel)
            #logits1 = 100. * image_feat @ self.text_feat10.t()
            #logits2 = 100. * attention_select_view_image @ self.text_feat.t()
            logits3 = 100. * attention_pca @ self.text_feat1.t()
            #logits = logits2 + self.cfg.alpha *logits3
            #logits = self.cfg.alpha * logits2 + logits3
            #logits = 100. * image_feat_w @ self.text_feat.repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS).t()
            logits_m = compute_m_logit_simple(text_feature=self.text_feat1, image_feature=weighted_representation, cov_inv=temp_cov_inv)


            #计算均值向量，然后和text1进行比较，得到 softmax的logit
            image_mean = image_feat_w.mean(dim=1).reshape(-1, 512)
            # image_mean (16, 512)
            mean_probility = F.softmax(torch.mm(image_mean, self.text_feat1.t()).reshape(-1, 10), dim=1)

            visual_probility(logits_m, mean_probility, label)



        return logits3

    def test_zs(self, split=None):
        """A generic testing zero-shot pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == 'val' and self.val_loader is not None:
            data_loader = self.val_loader
            print('Do evaluation on {} set'.format(split))
        else:
            data_loader = self.test_loader
            print('Do evaluation on test set')
        start = time.time()

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)

            output = self.model_inference(input, label)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = '{}/{}'.format(split, k)
            self.write_scalar(tag, v, self.epoch)

        feat_store = torch.cat(self.feat_store)
        label_store = torch.cat(self.label_store)
        print('Save feature: ============================')
        print('Save labels: =============================')
        torch.save(feat_store, osp.join(self.cfg.OUTPUT_DIR, "features20.pt"))
        torch.save(label_store, osp.join(self.cfg.OUTPUT_DIR, "labels20.pt"))

        # torch.save(feat_store, osp.join(self.cfg.OUTPUT_DIR, "features.pt"))
        # torch.save(label_store, osp.join(self.cfg.OUTPUT_DIR, "labels.pt"))

        cov_inv_store = torch.cat(self.cov_inv_store)
        print("save cov_inv:==============================")
        torch.save(cov_inv_store, osp.join(self.cfg.OUTPUT_DIR, "cov_inv20.pth"))
        print('Total time: {}'.format(time.time() - start))

        return list(results.values())[0]