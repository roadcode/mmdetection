# Copyright (c) OpenMMLab. All rights reserved.
"""
Grounding DINO 检测器实现
将 DINO 与 Grounded Pre-training 结合，支持开放集目标检测
论文: https://arxiv.org/abs/2303.05499
官方代码: https://github.com/IDEA-Research/GroundingDINO
"""

import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)


def clean_label_name(name: str) -> str:
    """
    清理标签名称，移除特殊字符和格式

    Args:
        name: 原始标签名称，如 "person_(1)" 或 "cat_dog"

    Returns:
        清理后的标签名称，如 "person" 或 "cat dog"

    Examples:
        >>> clean_label_name("person_(1)")
        'person'
        >>> clean_label_name("cat_dog")
        'cat dog'
    """
    # 移除括号及其内容
    name = re.sub(r'\(.*\)', '', name)
    # 将下划线替换为空格
    name = re.sub(r'_', ' ', name)
    # 将多个空格合并为一个
    name = re.sub(r'  ', ' ', name)
    return name


def chunks(lst: list, n: int) -> list:
    """
    将列表分割成固定大小的块

    Args:
        lst: 输入列表
        n: 每块的大小

    Returns:
        包含子列表的列表，每个子列表最多包含 n 个元素

    Examples:
        >>> chunks([1, 2, 3, 4, 5], 2)
        [[1, 2], [3, 4], [5]]
    """
    all_ = []
    for i in range(0, len(lst), n):
        data_index = lst[i:i + n]
        all_.append(data_index)

    # 验证分割后的总长度与原列表一致
    counter = 0
    for i in all_:
        counter += len(i)
    assert (counter == len(lst))

    return all_


@MODELS.register_module()
class GroundingDINO(DINO):
    """
    Grounding DINO 检测器

    实现论文《Grounding DINO: Marrying DINO with Grounded Pre-training
    for Open-Set Object Detection》

    主要特点：
    1. 支持开放集目标检测（通过文本提示检测任意类别）
    2. 多模态融合：视觉和文本特征在 Transformer 中交互
    3. 灵活的文本输入：支持字符串、列表、自动 NER

    示例:
        >>> # 文本提示模式
        >>> model = GroundingDINO(language_model=bert_cfg, ...)
        >>> # 检测 cat 和 dog
        >>> results = model.predict(img, text="cat . dog .")

        >>> # 列表模式（custom_entities=True）
        >>> results = model.predict(img, text=['cat', 'dog', 'car'])

        >>> # 分块模式（处理大量类别）
        >>> # test_cfg = dict(chunked_size=10)
        >>> results = model.predict(img, text=[类别列表])
    """

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 **kwargs) -> None:
        """
        初始化 Grounding DINO

        Args:
            language_model: 语言模型配置（如 BERT）
            use_autocast: 是否使用混合精度训练
            *args, **kwargs: 传递给父类 DINO 的参数
        """
        self.language_model_cfg = language_model
        # 特殊标记，用于分隔不同的类别文本
        self._special_tokens = '. '
        self.use_autocast = use_autocast
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """
        初始化模型各层组件（除了 backbone、neck 和 bbox_head）
        """
        # 位置编码：使用正弦位置编码
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)

        # Transformer 编码器：处理多模态输入
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)

        # Transformer 解码器：生成检测预测
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)

        # Query 嵌入：用于解码器的可学习查询向量
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        # 验证位置编码参数
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        # 层级嵌入：用于多尺度特征
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        # 记忆变换层：用于特征转换
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # ========== 文本模块 ==========
        # 构建语言模型（如 BERT）
        self.language_model = MODELS.build(self.language_model_cfg)

        # 文本特征映射层：将语言模型输出映射到视觉特征空间
        # 这是多模态融合的关键
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def init_weights(self) -> None:
        """初始化模型权重"""
        super().init_weights()
        # 初始化文本特征映射层的权重
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        """
        将原始类别转换为增强的文本提示

        支持为每个类别添加前缀和后缀，提升检测性能

        Args:
            original_caption: 原始类别列表，如 ['cat', 'dog']
            enhanced_text_prompts: 增强配置，如
                {'cat': {'prefix': 'a ', 'name': 'cat', 'suffix': ' on the road'}}

        Returns:
            caption_string: 组合后的文本字符串
            tokens_positive: 每个实体在文本中的位置 [start, end]

        Examples:
            >>> original_caption = ['cat', 'dog']
            >>> enhanced_prompts = {'cat': {'prefix': 'a '}}
            >>> to_enhance_text_prompts(original_caption, enhanced_prompts)
            ("a cat . dog . ", [[[2, 5]], [[6, 9]]])
        """
        caption_string = ''
        tokens_positive = []

        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]

                # 添加前缀
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']

                # 记录实体起始位置
                start_i = len(caption_string)

                # 添加名称（如果有配置）或原始词
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word

                # 记录实体结束位置
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                # 添加后缀
                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                # 没有增强配置，直接添加原词
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word

            # 添加特殊分隔符
            caption_string += self._special_tokens

        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        """
        将原始类别转换为简单的文本提示（无增强）

        Args:
            original_caption: 原始类别列表

        Returns:
            caption_string: 组合后的文本字符串
            tokens_positive: 每个实体的位置

        Examples:
            >>> to_plain_text_prompts(['cat', 'dog'])
            ("cat . dog . ", [[[0, 3]], [[4, 7]]])
        """
        caption_string = ''
        tokens_positive = []

        for idx, word in enumerate(original_caption):
            # 记录每个词的位置
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            # 添加分隔符
            caption_string += self._special_tokens

        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """
        获取 token 位置和提示文本

        这是文本处理的核心方法，支持多种输入模式

        Args:
            original_caption: 输入文本，可以是：
                - str: "cat . dog ."
                - list: ['cat', 'dog']
                - tuple: ('cat', 'dog')
            custom_entities: 是否使用自定义实体模式
                True 时 original_caption 应为列表
            enhanced_text_prompts: 文本增强配置

        Returns:
            tokenized: Tokenizer 的输出（包含 input_ids 等）
            caption_string: 处理后的完整文本
            tokens_positive: 每个实体的 token 位置列表
            entities: 实体名称列表

        模式说明:
            1. List/Tuple 模式 (custom_entities=True):
               输入: ['cat', 'dog']
               输出: "cat . dog . ", tokens_positive, entities=['cat', 'dog']

            2. String 模式 (custom_entities=False):
               输入: "There is a cat."
               自动运行 NER 提取实体，输出 tokens_positive
        """
        # ========== 模式1: List/Tuple 输入 ==========
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            # 如果是字符串且启用 custom_entities，先分割
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            # 清理每个类别的名称
            original_caption = [clean_label_name(i) for i in original_caption]

            # 根据是否有增强配置选择处理方式
            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # Tokenize 文本
            # 注意：Grounding DINO 的 tokenizer 与 GLIP 不同
            # GLIP 会 padding 到 max_length，Grounding DINO 则可选
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')

            entities = original_caption

        # ========== 模式2: String 输入（自动 NER）==========
        else:
            # 确保以特殊标记结尾
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens

            # Tokenize
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')

            # 运行 NER 提取实体和位置
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        """
        生成 positive map：实体 ID 到 Token ID 的映射

        用于分类时将预测的 token 关联到具体实体

        Args:
            tokenized: Tokenizer 输出
            tokens_positive: 每个实体的 token 位置

        Returns:
            positive_map_label_to_token: {实体ID: [token_id列表]}
            positive_map: Tensor 形式的映射矩阵

        Examples:
            >>> # 文本: "cat . dog . "
            >>> # tokens_positive: [[[0, 3]], [[4, 7]]]
            >>> # 输出: {1: [2], 2: [5]}
            >>> #        (假设 cat 的 token 是 2，dog 的 token 是 5)
        """
        # 创建映射矩阵
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers].max_text_len)

        # 转换为 label_to_token 格式
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)  # plus=1 因为 label 从 1 开始

        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """
        获取 tokens_positive 和提示文本（高级接口）

        支持多种场景，包括直接提供 tokens_positive

        Args:
            original_caption: 输入文本
            custom_entities: 是否使用自定义实体
            enhanced_text_prompt: 文本增强配置
            tokens_positive: 预先计算的 tokens_positive
                - None: 自动计算
                - -1: 不计算（用于识别任务）
                - 其他值: 使用提供的值

        Returns:
            positive_map_label_to_token: 实体到 token 的映射
            caption_string: 处理后的文本
            positive_map: 映射矩阵
            entities: 实体列表
        """
        # 如果预先提供了 tokens_positive
        if tokens_positive is not None:
            if tokens_positive == -1:
                # 特殊值 -1：不计算映射（用于识别任务）
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                # 使用提供的 tokens_positive
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens

                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')

                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                # 从 tokens_positive 提取实体名称
                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))

                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        # ========== 检查是否使用分块模式 ==========
        # 分块模式用于处理大量类别（如 COCO 的 80 类）
        chunked_size = self.test_cfg.get('chunked_size', -1)

        if not self.training and chunked_size > 0:
            # 验证输入格式
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True

            # 使用分块处理
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            # 正常模式
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)

            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        """
        分块处理大量类别

        将长列表分成多个小块，避免超出 tokenizer 的最大长度

        Args:
            original_caption: 类别列表
            enhanced_text_prompts: 文本增强配置

        Returns:
            positive_map_label_to_token_chunked: 分块的映射列表
            caption_string_chunked: 分块的文本列表
            positive_map_chunked: 分块的 positive_map 列表
            entities_chunked: 分块的实体列表

        Examples:
            >>> # 假设有 100 个类别，chunked_size=10
            >>> # 输出将包含 10 个块，每块 10 个类别
        """
        chunked_size = self.test_cfg.get('chunked_size', -1)
        original_caption = [clean_label_name(i) for i in original_caption]

        # 分块处理类别和 ID
        original_caption_chunked = chunks(original_caption, chunked_size)
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size)

        # 存储各块的结果
        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        # 逐块处理
        for i in range(len(ids_chunked)):
            # 生成该块的文本
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])

            # Tokenize
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')

            # 检查长度
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')

            # 生成映射
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            # 存储结果
            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """
        Transformer 前向传播的主流程

        整合了 pre_transformer -> encoder -> pre_decoder -> decoder 的完整流程

        Args:
            img_feats: 图像特征（来自 backbone + neck）
            text_dict: 文本特征字典，包含：
                - 'embedded': 文本嵌入
                - 'text_token_mask': 文本 token mask
                - 'position_ids': 位置 ID
                - 'masks': 注意力 mask
            batch_data_samples: 批量数据样本

        Returns:
            head_inputs_dict: 传递给 bbox_head 的输入字典
        """
        # 1. Pre-transformer：准备 encoder 的输入
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        # 2. Encoder：融合视觉和文本特征
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        # 3. Pre-decoder：生成 top-k 提案
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        # 4. Decoder：生成最终预测
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        """
        Encoder 前向传播

        将视觉特征和文本特征在 encoder 中融合

        Args:
            feat: 视觉特征
            feat_mask: 特征 mask
            feat_pos: 特征位置编码
            spatial_shapes: 各层特征的空间形状
            level_start_index: 各层起始索引
            valid_ratios: 有效比例
            text_dict: 文本特征字典

        Returns:
            encoder_outputs_dict: 包含：
                - memory: 编码后的视觉特征
                - memory_mask: 视觉特征 mask
                - spatial_shapes: 空间形状
                - memory_text: 编码后的文本特征
                - text_token_mask: 文本 token mask
        """
        text_token_mask = text_dict['text_token_mask']

        # Encoder 同时处理视觉和文本
        memory, memory_text = self.encoder(
            query=feat,                      # 视觉特征作为 query
            query_pos=feat_pos,              # 视觉位置编码
            key_padding_mask=feat_mask,      # 视觉特征的 padding mask
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # ========== 文本输入 ==========
            memory_text=text_dict['embedded'],      # 文本嵌入作为 key/value
            text_attention_mask=~text_token_mask,   # 文本注意力 mask（取反）
            position_ids=text_dict['position_ids'], # 文本位置 ID
            text_self_attention_masks=text_dict['masks'])  # 文本自注意力 mask

        # 组装输出
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)

        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """
        Decoder 前�向传播前的准备

        主要功能：从 encoder 输出中选择 top-k 提案作为 decoder 的 query

        Args:
            memory: Encoder 输出的视觉特征
            memory_mask: 视觉特征 mask
            spatial_shapes: 空间形状
            memory_text: Encoder 输出的文本特征
            text_token_mask: 文本 token mask
            batch_data_samples: 数据样本

        Returns:
            decoder_inputs_dict: Decoder 的输入字典
            head_inputs_dict: BBox head 的输入字典
        """
        bs, _, c = memory.shape

        # 1. 生成 encoder 输出和提案
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        # 2. 分类分支：使用文本特征增强的分类器
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len

        # 3. 回归分支：预测边界框坐标
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # 4. 选择 top-k 提案
        # DINO 根据多类分类的最大分数选择 top-k
        # DeformDETR 则根据二分类分数选择
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        # 收集 top-k 的分类分数和坐标
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))

        # 坐标 sigmoid 激活
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # 5. 准备 decoder query
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.training:
            # 训练模式：添加去噪（denoising）query
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            # 推理模式
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        # 6. 组装 decoder 输入
        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )

        # 7. 组装 head 输入（用于计算 encoder loss）
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()

        # 添加文本特征
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask

        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """
        计算损失

        Args:
            batch_inputs: 批量输入图像
            batch_data_samples: 批量数据样本，包含：
                - text: 文本提示
                - gt_instances.labels: 真值标签
                - tokens_positive: 可选的 token 位置

        Returns:
            losses: 损失字典
        """
        # 1. 提取文本和标签
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        # 2. 处理 tokens_positive（如果提供）
        if 'tokens_positive' in batch_data_samples[0]:
            # 使用预定义的 tokens_positive
            tokens_positive = [
                data_samples.tokens_positive
                for data_samples in batch_data_samples
            ]
            positive_maps = []

            for token_positive, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                # Tokenize 文本
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')

                # 根据真值标签筛选 tokens_positive
                new_tokens_positive = [
                    token_positive[label.item()] for label in gt_label
                ]

                # 生成 positive_map
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)

            new_text_prompts = text_prompts
        else:
            # 自动生成 positive_map
            new_text_prompts = []
            positive_maps = []

            if len(set(text_prompts)) == 1:
                # 所有文本提示相同，只计算一次
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)

                # 为每个样本生成 positive_map
                for gt_label in gt_labels:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
            else:
                # 不同样本有不同文本提示
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompt, True)
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        # 3. 提取文本特征
        text_dict = self.language_model(new_text_prompts)

        # 文本特征映射到视觉空间
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        # 4. 附加 positive_map 到数据样本
        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        # 5. 提取视觉特征
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        # 6. Transformer 前向传播
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        # 7. 计算损失
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        """
        推理预测

        Args:
            batch_inputs: 批量输入图像
            batch_data_samples: 批量数据样本
            rescale: 是否将边界框重新缩放到原图尺寸

        Returns:
            batch_data_samples: 包含预测结果的样本
        """
        # 1. 收集所有输入
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []

        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        # 2. 检查是否使用 custom_entities
        if 'custom_entities' in batch_data_samples[0]:
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False

        # 3. 处理文本提示
        if len(text_prompts) == 1:
            # 所有文本提示相同，只计算一次
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities, enhanced_text_prompts[0],
                    tokens_positives[0])
            ] * len(batch_inputs)
        else:
            # 不同图像有不同文本提示
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities,
                                                     enhanced_text_prompt,
                                                     tokens_positive)
                for text_prompt, enhanced_text_prompt, tokens_positive in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]

        # 解包结果
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        # 4. 提取视觉特征
        visual_feats = self.extract_feat(batch_inputs)

        # 5. 分块模式 vs 正常模式
        if isinstance(text_prompts[0], list):
            # ========== 分块模式：处理大量类别 ==========
            # 注意：只支持 batch_size=1
            assert len(batch_inputs) == 1
            count = 0
            results_list = []

            # 展平实体列表
            entities = [[item for lst in entities[0] for item in lst]]

            # 逐块处理
            for b in range(len(text_prompts[0])):
                text_prompts_once = [text_prompts[0][b]]
                token_positive_maps_once = token_positive_maps[0][b]

                # 提取文本特征
                text_dict = self.language_model(text_prompts_once)

                # 文本特征映射
                if self.text_feat_map is not None:
                    text_dict['embedded'] = self.text_feat_map(
                        text_dict['embedded'])

                # 设置 token_positive_map
                batch_data_samples[
                    0].token_positive_map = token_positive_maps_once

                # 前向传播
                head_inputs_dict = self.forward_transformer(
                    copy.deepcopy(visual_feats), text_dict, batch_data_samples)

                # 预测
                pred_instances = self.bbox_head.predict(
                    **head_inputs_dict,
                    rescale=rescale,
                    batch_data_samples=batch_data_samples)[0]

                # 调整标签 ID（因为分块处理）
                if len(pred_instances) > 0:
                    pred_instances.labels += count
                count += len(token_positive_maps_once)
                results_list.append(pred_instances)

            # 合并所有块的结果
            results_list = [results_list[0].cat(results_list)]
            is_rec_tasks = [False] * len(results_list)

        else:
            # ========== 正常模式 ==========
            # 提取文本特征
            text_dict = self.language_model(list(text_prompts))

            # 文本特征映射
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(
                    text_dict['embedded'])

            # 判断是否为识别任务
            is_rec_tasks = []
            for i, data_samples in enumerate(batch_data_samples):
                if token_positive_maps[i] is not None:
                    is_rec_tasks.append(False)
                else:
                    is_rec_tasks.append(True)
                data_samples.token_positive_map = token_positive_maps[i]

            # 前向传播
            head_inputs_dict = self.forward_transformer(
                visual_feats, text_dict, batch_data_samples)

            # 预测
            results_list = self.bbox_head.predict(
                **head_inputs_dict,
                rescale=rescale,
                batch_data_samples=batch_data_samples)

        # 6. 附加标签名称（用于可视化）
        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        # 识别任务
                        label_names.append(entity)
                        continue

                    if labels >= len(entity):
                        # 检测到异常，可能是 NER 问题
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])

                # 附加标签名称
                pred_instances.label_names = label_names

            data_sample.pred_instances = pred_instances

        return batch_data_samples
