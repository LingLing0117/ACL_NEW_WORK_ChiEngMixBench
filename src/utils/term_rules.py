# coding=utf-8
# Copyright 2025 The ChiEngMixBench Authors.
# Licensed under the Apache License, Version 2.0.

"""
Static dictionary for Terminology Compactness Check.
Maps standard Acronyms to their penalized variants (Full names, Chinese translations).
"""

ACRONYM_RULES = {
    "ADALINE": [
        "Adaptive Linear Element",
        "自适应线性单元",
        "自适应线性元素",
        "自适应线性神经元"
    ],
    "ADBench": [
        "Anomaly Detection Benchmark",
        "异常检测基准",
        "异常检测基准"
    ],
    "AED": [
        "AutoEncoder Decoder",
        "自编码器解码器",
        "自动编码器解码器"
    ],
    "AGI": [
        "Artificial General Intelligence",
        "通用人工智能",
        "人工通用智能",
        "强人工智能"
    ],
    "AI": [
        "Artificial Intelligence",
        "人工智能",
        "人工智能"
    ],
    "AIC": [
        "Akaike Information Criterion",
        "赤池信息准则",
        "赤池信息判据"
    ],
    "AIGC": [
        "Artificial Intelligence Generated Content",
        "人工智能生成内容",
        "人工智能生成内容",
        "AI生成内容"
    ],
    "ALBERT": [
        "A Lite BERT for Self-supervised Learning of Language Representations",
        "艾伯特模型",
        "轻量级BERT",
        "用于语言表示的自监督学习的轻量级BERT"
    ],
    "ALS": [
        "Alternating Least Squares",
        "交替最小二乘",
        "交替最小平方"
    ],
    "ANN": [
        "Artificial Neural Network",
        "人工神经网络",
        "人工神经网络",
        "神经网络"
    ],
    "ANNs": [
        "Artificial Neural Networks",
        "人工神经网络",
        "人工神经网络",
        "神经网络"
    ],
    "ARIMA": [
        "AutoRegressive Integrated Moving Average",
        "差分整合移动平均自回归模型",
        "自回归积分滑动平均",
        "自回归积分移动平均"
    ],
    "ASR": [
        "Automatic Speech Recognition",
        "自动语音识别",
        "自动语音识别",
        "语音识别"
    ],
    "AVSR": [
        "Audio-Visual Speech Recognition",
        "视听语音识别",
        "音频-视觉语音识别"
    ],
    "BART": [
        "Denoising Sequence-to-Sequence Pre-training for Text Generation, Translation, and Comprehension",
        "巴特模型",
        "去噪序列到序列预训练",
        "序列到序列去噪预训练模型"
    ],
    "BERT": [
        "Bidirectional Encoder Representations from Transformers",
        "伯特模型",
        "双向编码器表示",
        "来自Transformer的双向编码器表示"
    ],
    "BLEU": [
        "Bilingual Evaluation Understudy",
        "机器翻译评估指标",
        "双语评估替补",
        "BLEU分数"
    ],
    "BN": [
        "Batch Normalization",
        "批量归一化",
        "批归一化",
        "批标准化"
    ],
    "BOS": [
        "Beginning Of Sentence",
        "句首标记",
        "句子开始符",
        "句子的开始"
    ],
    "BOT": [
        "Robot",
        "机器人",
        "聊天机器人",
        "自动化代理"
    ],
    "BPTT": [
        "Backpropagation Through Time",
        "沿时间反向传播",
        "随时间反向传播"
    ],
    "CAM": [
        "Class Activation Map",
        "类别激活图",
        "类别激活映射"
    ],
    "CBOW": [
        "Continuous Bag-Of-Words",
        "连续词袋模型",
        "连续词袋"
    ],
    "CCNet": [
        "Common Crawl Network",
        "Common Crawl数据集",
        "普通爬网网络"
    ],
    "CE": [
        "Cross-Entropy",
        "交叉熵",
        "交叉熵损失"
    ],
    "CEM": [
        "Cross-Entropy Method",
        "交叉熵方法",
        "交叉熵方法"
    ],
    "CFG": [
        "Context-Free Grammar",
        "上下文无关文法",
        "无上下文文法",
        "上下文无关语法"
    ],
    "CIL": [
        "Class-Incremental Learning",
        "类别增量学习",
        "类别增量学习"
    ],
    "CKA": [
        "Centered Kernel Alignment",
        "中心核对齐",
        "居中核对齐"
    ],
    "CL": [
        "Contrastive Learning",
        "Continual Learning",
        "对比学习",
        "持续学习",
        "增量学习"
    ],
    "CLEVR": [
        "Compositional Language and Elementary Visual Reasoning",
        "CLEVR数据集",
        "组合语言和初级视觉推理"
    ],
    "CLIP": [
        "Contrastive Language-Image Pre-training",
        "CLIP模型",
        "对比语言-图像预训练",
        "对比图文预训练"
    ],
    "CLUE": [
        "Chinese Language Understanding Evaluation",
        "中文语言理解测评",
        "线索榜单",
        "中文语言理解评估"
    ],
    "CLUECorpus": [
        "Chinese Language Understanding Evaluation Corpus",
        "CLUE语料库",
        "中文语言理解测评语料库",
        "线索语料库"
    ],
    "CLUES": [
        "Chinese Language Understanding Evaluation Scores",
        "CLUE分数",
        "中文语言理解评估分数"
    ],
    "CM": [
        "Confusion Matrix",
        "混淆矩阵",
        "混淆矩阵"
    ],
    "CNN": [
        "Convolutional Neural Network",
        "卷积神经网络",
        "卷积网络",
        "ConvNet"
    ],
    "CNNS": [
        "Convolutional Neural Networks",
        "卷积神经网络",
        "卷积神经网络"
    ],
    "COCO": [
        "Common Objects in Context",
        "COCO数据集",
        "上下文中的通用目标",
        "上下文中的常见对象"
    ],
    "COLA": [
        "Corpus of Linguistic Acceptability",
        "COLA数据集",
        "语言可接受性语料库"
    ],
    "COS": [
        "Cosine",
        "余弦",
        "余弦",
        "Cos"
    ],
    "COT": [
        "Chain-of-Thought",
        "思维链",
        "思想链",
        "CoT reasoning"
    ],
    "CRF": [
        "Conditional Random Field",
        "条件随机场",
        "条件随机场"
    ],
    "DCC": [
        "Deep Canonical Correlation",
        "深度典型相关",
        "深度典型相关"
    ],
    "DCNN": [
        "Deep Convolutional Neural Network",
        "深度卷积神经网络",
        "深度卷积神经网络",
        "Deep CNN"
    ],
    "DCUNet": [
        "Deep Convolutional U-Net",
        "深度卷积U型网络",
        "深度卷积U形网络"
    ],
    "DDPG": [
        "Deep Deterministic Policy Gradient",
        "深度确定性策略梯度",
        "深度确定性策略梯度"
    ],
    "DDPM": [
        "Denoising Diffusion Probabilistic Models",
        "去噪扩散概率模型",
        "去噪扩散概率模型",
        "Diffusion Model",
        "DDPMs"
    ],
    "DDPMS": [
        "Denoising Diffusion Probabilistic Models",
        "去噪扩散概率模型",
        "去噪扩散概率模型",
        "Diffusion Models",
        "DDPM"
    ],
    "EM": [
        "Expectation-Maximization",
        "期望最大化",
        "期望最大化",
        "EM算法"
    ],
    "EMA": [
        "Exponential Moving Average",
        "指数移动平均",
        "指数移动平均"
    ],
    "ENN": [
        "Evolving Neural Network",
        "演化神经网络",
        "演化神经网络"
    ],
    "ENNs": [
        "Evolving Neural Networks",
        "演化神经网络",
        "演化神经网络"
    ],
    "F1": [
        "F1 Score",
        "F1分数",
        "F1分数",
        "F-measure",
        "F-score"
    ],
    "FFN": [
        "Feed-Forward Network",
        "前馈网络",
        "前馈网络"
    ],
    "FFNN": [
        "Feed-Forward Neural Network",
        "前馈神经网络",
        "前馈神经网络",
        "FFN"
    ],
    "FGSM": [
        "Fast Gradient Sign Method",
        "快速梯度符号法",
        "快速梯度符号法"
    ],
    "FNN": [
        "Feedforward Neural Network",
        "前馈神经网络",
        "前馈神经网络",
        "FFNN",
        "FFN"
    ],
    "FRCNN": [
        "Faster R-CNN",
        "更快的R-CNN",
        "快速区域卷积神经网络",
        "Faster R-CNN"
    ],
    "G2P": [
        "Grapheme-to-Phoneme",
        "字素转音素",
        "字素到音素"
    ],
    "GA": [
        "Genetic Algorithm",
        "遗传算法",
        "遗传算法"
    ],
    "GAN": [
        "Generative Adversarial Network",
        "生成对抗网络",
        "生成对抗网络"
    ],
    "GANS": [
        "Generative Adversarial Networks",
        "生成对抗网络",
        "生成对抗网络",
        "GAN"
    ],
    "GCN": [
        "Graph Convolutional Network",
        "图卷积网络",
        "图卷积网络"
    ],
    "GCNs": [
        "Graph Convolutional Networks",
        "图卷积网络",
        "图卷积网络",
        "GCN"
    ],
    "GGNN": [
        "Gated Graph Neural Network",
        "门控图神经网络",
        "门控图神经网络"
    ],
    "GLUE": [
        "General Language Understanding Evaluation",
        "通用语言理解评估",
        "通用语言理解评估",
        "GLUE benchmark"
    ],
    "GNN": [
        "Graph Neural Network",
        "图神经网络",
        "图神经网络"
    ],
    "GNNS": [
        "Graph Neural Networks",
        "图神经网络",
        "图神经网络",
        "GNN"
    ],
    "GPT": [
        "Generative Pre-trained Transformer",
        "生成式预训练Transformer",
        "生成式预训练转换器"
    ],
    "GPT2": [
        "Generative Pre-trained Transformer 2",
        "生成式预训练Transformer 2",
        "生成式预训练转换器 2",
        "GPT-2"
    ],
    "GPT4": [
        "Generative Pre-trained Transformer 4",
        "生成式预训练Transformer 4",
        "生成式预训练转换器 4",
        "GPT-4"
    ],
    "GRN": [
        "Gated Recurrent Network",
        "门控循环网络",
        "门控循环网络"
    ],
    "GRU": [
        "Gated Recurrent Unit",
        "门控循环单元",
        "门控循环单元"
    ],
    "GRUS": [
        "Gated Recurrent Units",
        "门控循环单元",
        "门控循环单元",
        "GRU"
    ],
    "HAN": [
        "Hierarchical Attention Network",
        "层次注意力网络",
        "分层注意力网络"
    ],
    "HAR": [
        "Human Activity Recognition",
        "人体活动识别",
        "人类活动识别"
    ],
    "HE": [
        "Homomorphic Encryption",
        "同态加密",
        "同态加密"
    ],
    "HMM": [
        "Hidden Markov Model",
        "隐马尔可夫模型",
        "隐马尔可夫模型"
    ],
    "HNN": [
        "Hamiltonian Neural Network",
        "哈密顿神经网络",
        "哈密顿神经网络"
    ],
    "HTN": [
        "Hierarchical Task Network",
        "分层任务网络",
        "层级任务网络"
    ],
    "HUBERT": [
        "Hidden Unit BERT",
        "隐单元BERT",
        "HUBERT模型",
        "隐藏单元BERT"
    ],
    "IDS": [
        "Intrusion Detection System",
        "入侵检测系统",
        "入侵侦测系统"
    ],
    "IID": [
        "Independent and Identically Distributed",
        "独立同分布",
        "独立且同分布"
    ],
    "IL": [
        "Imitation Learning",
        "模仿学习",
        "行为克隆",
        "仿效学习"
    ],
    "ILP": [
        "Integer Linear Programming",
        "Inductive Logic Programming",
        "整数线性规划",
        "归纳逻辑编程",
        "整数规划",
        "归纳逻辑"
    ],
    "KG": [
        "Knowledge Graph",
        "知识图谱",
        "知识图"
    ],
    "KGE": [
        "Knowledge Graph Embedding",
        "知识图谱嵌入",
        "知识图谱表示学习",
        "知识图嵌入"
    ],
    "KL": [
        "Kullback-Leibler Divergence",
        "KL散度",
        "库尔巴克-莱布勒散度",
        "库尔巴克-莱布勒差异"
    ],
    "KNN": [
        "K-Nearest Neighbors",
        "K近邻算法",
        "K近邻",
        "K最近邻居"
    ],
    "KR": [
        "Knowledge Representation",
        "知识表示",
        "知识表达"
    ],
    "KRR": [
        "Kernel Ridge Regression",
        "核岭回归",
        "核脊回归"
    ],
    "LAVIS": [
        "Language-Vision",
        "LAVIS模型",
        "语言-视觉模型",
        "语言视觉"
    ],
    "LBFGS": [
        "Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm",
        "L-BFGS",
        "限量内存BFGS算法",
        "有限内存BFGS算法"
    ],
    "LBP": [
        "Local Binary Pattern",
        "局部二值模式",
        "本地二值模式"
    ],
    "LDA": [
        "Latent Dirichlet Allocation",
        "Linear Discriminant Analysis",
        "潜在狄利克雷分配",
        "线性判别分析",
        "隐狄利克雷分配",
        "线性判别"
    ],
    "LDM": [
        "Latent Diffusion Model",
        "潜在扩散模型",
        "隐扩散模型"
    ],
    "LIF": [
        "Leaky Integrate-and-Fire neuron",
        "漏泄积分放电神经元",
        "渗漏积分放电神经元",
        "漏积分放电神经元"
    ],
    "LIME": [
        "Local Interpretable Model-agnostic Explanations",
        "局部可解释模型无关解释",
        "局部可解释模型无关性解释"
    ],
    "LLAMA": [
        "Large Language Model Meta AI",
        "LLAMA模型",
        "Meta大语言模型",
        "大语言模型Meta AI"
    ],
    "LLM": [
        "Large Language Model",
        "大语言模型",
        "大模型"
    ],
    "LLMs": [
        "Large Language Models",
        "大语言模型 (复数)",
        "多个大语言模型",
        "大语言模型们"
    ],
    "LORAS": [
        "Low-Rank Adaptation",
        "LoRA",
        "低秩适应",
        "低秩适配"
    ],
    "LR": [
        "Logistic Regression",
        "Learning Rate",
        "逻辑回归",
        "学习率",
        "逻辑斯蒂回归",
        "学习速率"
    ],
    "LSA": [
        "Latent Semantic Analysis",
        "潜在语义分析",
        "隐性语义分析",
        "隐语义分析"
    ],
    "LSH": [
        "Locality Sensitive Hashing",
        "局部敏感哈希",
        "位置敏感哈希"
    ],
    "LSTM": [
        "Long Short-Term Memory",
        "长短期记忆网络",
        "长短期记忆"
    ],
    "LSTMs": [
        "Long Short-Term Memories",
        "长短期记忆网络 (复数)",
        "多个长短期记忆网络",
        "长短期记忆们"
    ],
    "LVIS": [
        "Large Vocabulary Instance Segmentation",
        "LVIS数据集",
        "大词汇量实例分割",
        "大词汇实例分割"
    ],
    "MM": [
        "Multimodal",
        "Markov Model",
        "多模态",
        "马尔可夫模型",
        "多模式",
        "马尔科夫模型"
    ],
    "MNIST": [
        "Modified National Institute of Standards and Technology",
        "MNIST数据集",
        "手写数字数据集",
        "修改版美国国家标准与技术研究院"
    ],
    "MOE": [
        "Mixture of Experts",
        "专家混合模型",
        "混合专家模型",
        "专家混合"
    ],
    "MOG": [
        "Mixture of Gaussians",
        "高斯混合模型",
        "高斯混合"
    ],
    "MOS": [
        "Mean Opinion Score",
        "平均意见分数",
        "平均意见得分",
        "平均意见分"
    ],
    "MPC": [
        "Model Predictive Control",
        "Multi-Party Computation",
        "模型预测控制",
        "多方安全计算",
        "多方计算"
    ],
    "MPE": [
        "Maximum a Posteriori Estimation",
        "最大后验估计",
        "最大后验概率估计"
    ],
    "MRC": [
        "Machine Reading Comprehension",
        "机器阅读理解",
        "阅读理解"
    ],
    "MRNN": [
        "Multilayer Recurrent Neural Network",
        "多层循环神经网络",
        "多层递归神经网络"
    ],
    "MSA": [
        "Multi-Scale Attention",
        "Multiple Sequence Alignment",
        "多尺度注意力",
        "多序列比对",
        "多尺度注意机制",
        "多序列对齐"
    ],
    "MSE": [
        "Mean Squared Error",
        "均方误差",
        "平均平方误差"
    ],
    "MT": [
        "Machine Translation",
        "机器翻译",
        "机器翻译"
    ],
    "MTCNN": [
        "Multi-task Cascaded Convolutional Networks",
        "多任务级联卷积网络",
        "多任务级联卷积网络"
    ],
    "MTT": [
        "Multi-Task Training",
        "多任务训练",
        "多任务训练"
    ],
    "MVS": [
        "Multi-View Stereo",
        "多视图立体视觉",
        "多视图立体"
    ],
    "NAS": [
        "Neural Architecture Search",
        "神经网络结构搜索",
        "神经架构搜索"
    ],
    "NCA": [
        "Neighborhood Components Analysis",
        "近邻分量分析",
        "邻域分量分析"
    ],
    "NCC": [
        "Normalized Cross-Correlation",
        "归一化互相关",
        "归一化交叉相关"
    ],
    "NER": [
        "Named Entity Recognition",
        "命名实体识别",
        "命名实体识别"
    ],
    "NLP": [
        "Natural Language Processing",
        "自然语言处理",
        "自然语言处理"
    ],
    "NLU": [
        "Natural Language Understanding",
        "自然语言理解",
        "自然语言理解"
    ],
    "NMT": [
        "Neural Machine Translation",
        "神经机器翻译",
        "神经机器翻译"
    ],
    "NN": [
        "Neural Network",
        "神经网络",
        "神经网络"
    ],
    "NNs": [
        "Neural Networks",
        "神经网络",
        "神经网络"
    ],
    "NP": [
        "Noun Phrase",
        "名词短语",
        "名词短语",
        "Non-Parametric",
        "非参数"
    ],
    "NPU": [
        "Neural Processing Unit",
        "神经网络处理器",
        "神经处理单元"
    ],
    "NSP": [
        "Next Sentence Prediction",
        "下一句预测",
        "下一语句预测"
    ],
    "OCR": [
        "Optical Character Recognition",
        "光学字符识别",
        "光学字符识别"
    ],
    "OD": [
        "Object Detection",
        "目标检测",
        "对象检测"
    ],
    "ONN": [
        "Optical Neural Network",
        "光学神经网络",
        "光学神经网络"
    ],
    "OOV": [
        "Out-Of-Vocabulary",
        "词汇表外词",
        "词汇外",
        "未登录词"
    ],
    "OPT": [
        "Open Pre-trained Transformer",
        "开放预训练Transformer模型",
        "开放预训练变换器"
    ],
    "PALM": [
        "Pathways Language Model",
        "通路语言模型",
        "通路语言模型"
    ],
    "PCA": [
        "Principal Component Analysis",
        "主成分分析",
        "主成分分析"
    ],
    "PET": [
        "Parameter-Efficient Transfer learning",
        "参数高效迁移学习",
        "参数效率迁移学习"
    ],
    "PG": [
        "Policy Gradient",
        "策略梯度",
        "策略梯度"
    ],
    "PGD": [
        "Projected Gradient Descent",
        "投影梯度下降",
        "投影梯度下降"
    ],
    "PIM": [
        "Processing In Memory",
        "存内计算",
        "内存处理",
        "内存计算"
    ],
    "PLA": [
        "Perceptron Learning Algorithm",
        "感知机学习算法",
        "感知器学习算法"
    ],
    "PLM": [
        "Pre-trained Language Model",
        "预训练语言模型",
        "预训练语言模型"
    ],
    "PLMs": [
        "Pre-trained Language Models",
        "预训练语言模型",
        "预训练语言模型"
    ],
    "PNN": [
        "Probabilistic Neural Network",
        "概率神经网络",
        "概率神经网络",
        "Product Neural Network",
        "产品神经网络"
    ],
    "PPO": [
        "Proximal Policy Optimization",
        "近端策略优化",
        "近端策略优化"
    ],
    "PR": [
        "Precision-Recall",
        "精确率-召回率",
        "精确度-召回",
        "Pattern Recognition",
        "模式识别"
    ],
    "PRA": [
        "Path Ranking Algorithm",
        "路径排序算法",
        "路径排序算法"
    ],
    "RBF": [
        "Radial Basis Function",
        "径向基函数",
        "径向基函数"
    ],
    "RCNN": [
        "Region-based Convolutional Neural Network",
        "基于区域的卷积神经网络",
        "基于区域的卷积神经网络"
    ],
    "RELU": [
        "Rectified Linear Unit",
        "整流线性单元",
        "修正线性单元",
        "ReLU"
    ],
    "RF": [
        "Random Forest",
        "随机森林",
        "随机森林"
    ],
    "RL": [
        "Reinforcement Learning",
        "强化学习",
        "强化学习"
    ],
    "RLHF": [
        "Reinforcement Learning from Human Feedback",
        "基于人类反馈的强化学习",
        "基于人类反馈的强化学习"
    ],
    "RNN": [
        "Recurrent Neural Network",
        "循环神经网络",
        "循环神经网络",
        "RNNs"
    ],
    "RS": [
        "Recommender System",
        "推荐系统",
        "推荐系统",
        "推荐算法"
    ],
    "ResNet": [
        "Residual Network",
        "残差网络",
        "残差网络",
        "ResNets",
        "残差连接网络"
    ],
    "ResNet18": [
        "Residual Network with 18 layers",
        "18层残差网络",
        "残差网络18",
        "ResNet-18"
    ],
    "ResNets": [
        "Residual Networks",
        "残差网络",
        "残差网络群",
        "ResNet系列"
    ],
    "SA": [
        "Self-Attention",
        "自注意力",
        "自注意力机制",
        "自我注意",
        "Simulated Annealing",
        "模拟退火",
        "模拟退火"
    ],
    "SAC": [
        "Soft Actor-Critic",
        "软执行器-评论家算法",
        "软行动者-评论家",
        "SAC算法"
    ],
    "SAMs": [
        "Segment Anything Models",
        "分割一切模型",
        "分割任何模型",
        "SAM模型",
        "Sharpness-Aware Minimization",
        "锐度感知最小化",
        "感知锐度最小化"
    ],
    "SAT": [
        "Satisfiability Problem",
        "可满足性问题",
        "可满足性问题",
        "布尔可满足性问题"
    ],
    "SD": [
        "Stable Diffusion",
        "稳定扩散",
        "稳定扩散",
        "Stable Diffusion模型",
        "Standard Deviation",
        "标准差",
        "标准偏差"
    ],
    "SDP": [
        "Semidefinite Programming",
        "半正定规划",
        "半定规划",
        "SDP问题"
    ],
    "SEGAN": [
        "Speech Enhancement Generative Adversarial Network",
        "语音增强生成对抗网络",
        "语音增强生成对抗网络",
        "SEGAN模型"
    ],
    "SGD": [
        "Stochastic Gradient Descent",
        "随机梯度下降",
        "随机梯度下降",
        "SGD算法",
        "随机梯度下降法"
    ],
    "SGDM": [
        "Stochastic Gradient Descent with Momentum",
        "动量随机梯度下降",
        "动量随机梯度下降",
        "SGDM算法",
        "带动量的随机梯度下降"
    ],
    "SGG": [
        "Scene Graph Generation",
        "场景图生成",
        "场景图生成",
        "SGG任务"
    ],
    "SGLD": [
        "Stochastic Gradient Langevin Dynamics",
        "随机梯度朗之万动力学",
        "随机梯度朗之万动力学",
        "SGLD算法"
    ],
    "SIFT": [
        "Scale-Invariant Feature Transform",
        "尺度不变特征变换",
        "尺度不变特征变换",
        "SIFT特征",
        "SIFT算法"
    ],
    "SL": [
        "Supervised Learning",
        "监督学习",
        "监督学习",
        "有监督学习"
    ],
    "SLAM": [
        "Simultaneous Localization and Mapping",
        "即时定位与地图构建",
        "同时定位与映射",
        "SLAM技术",
        "SLAM系统"
    ],
    "SLIC": [
        "Simple Linear Iterative Clustering",
        "简单线性迭代聚类",
        "简单线性迭代聚类",
        "SLIC算法"
    ],
    "SMO": [
        "Sequential Minimal Optimization",
        "序列最小优化",
        "序列最小优化",
        "SMO算法"
    ],
    "SMT": [
        "Statistical Machine Translation",
        "统计机器翻译",
        "统计机器翻译",
        "SMT系统"
    ],
    "SMTNN": [
        "Spiking Multi-Timescale Neural Network",
        "脉冲多时间尺度神经网络",
        "脉冲多时间尺度神经网络"
    ],
    "SRCNN": [
        "Super-Resolution Convolutional Neural Network",
        "超分辨率卷积神经网络",
        "超分辨率卷积神经网络",
        "SRCNN模型"
    ],
    "SRL": [
        "Semantic Role Labeling",
        "语义角色标注",
        "语义角色标注",
        "语义角色识别",
        "Structured Reinforcement Learning",
        "结构化强化学习",
        "结构化强化学习"
    ],
    "SSL": [
        "Self-Supervised Learning",
        "自监督学习",
        "自监督学习",
        "自监督学习方法",
        "Semi-Supervised Learning",
        "半监督学习",
        "半监督学习",
        "半监督学习方法"
    ],
    "SSLM": [
        "Self-Supervised Language Model",
        "自监督语言模型",
        "自监督语言模型",
        "SSLM模型",
        "自监督语言大模型"
    ],
    "SSMs": [
        "State Space Models",
        "状态空间模型",
        "状态空间模型",
        "状态空间模型系列",
        "Self-Supervised Models",
        "自监督模型",
        "自监督模型"
    ],
    "STN": [
        "Spatial Transformer Network",
        "空间变换网络",
        "空间变换网络",
        "STN层"
    ],
    "STNs": [
        "Spatial Transformer Networks",
        "空间变换网络",
        "空间变换网络群",
        "STN系列"
    ],
    "SVM": [
        "Support Vector Machine",
        "支持向量机",
        "支持向量机",
        "SVM模型"
    ],
    "SiamRPN": [
        "Siamese Region Proposal Network",
        "孪生区域候选网络",
        "暹罗区域提案网络",
        "SiamRPN++"
    ],
    "T5": [
        "Text-To-Text Transfer Transformer",
        "文本到文本转换Transformer",
        "文本到文本传递变形器",
        "T5模型"
    ],
    "TAPE": [
        "Tasks Assessing Protein Embeddings",
        "蛋白质嵌入评估任务",
        "蛋白质嵌入评估任务",
        "TAPE基准"
    ],
    "TCNN": [
        "Temporal Convolutional Neural Network",
        "时间卷积神经网络",
        "时间卷积神经网络",
        "TCNN模型"
    ],
    "TD": [
        "Temporal Difference",
        "时序差分",
        "时间差分",
        "TD学习"
    ],
    "TDA": [
        "Topological Data Analysis",
        "拓扑数据分析",
        "拓扑数据分析",
        "TDA方法"
    ],
    "TF": [
        "TensorFlow",
        "TensorFlow",
        "张量流",
        "TF框架",
        "Transformer",
        "Transformer模型",
        "变形器",
        "注意力机制"
    ],
    "TFE": [
        "TensorFlow Extended",
        "TensorFlow扩展",
        "TensorFlow扩展",
        "TFX",
        "TensorFlow生态扩展"
    ],
    "TFX": [
        "TensorFlow Extended",
        "TensorFlow Extended",
        "TensorFlow扩展",
        "TFE",
        "TensorFlow生产级机器学习平台"
    ],
    "TL": [
        "Transfer Learning",
        "迁移学习",
        "转移学习"
    ],
    "TPU": [
        "Tensor Processing Unit",
        "张量处理单元",
        "张量处理器",
        "谷歌TPU"
    ],
    "TTA": [
        "Test-Time Augmentation",
        "测试时增强",
        "测试时间增强",
        "测试时间数据增强"
    ],
    "VLLM": [
        "Very Large Language Model",
        "超大规模语言模型",
        "极大规模语言模型",
        "超大语言模型",
        "极大型语言模型"
    ],
    "ViT": [
        "Vision Transformer",
        "视觉Transformer",
        "视觉变换器",
        "Transformer视觉模型"
    ],
    "WDNN": [
        "Wide & Deep Neural Network",
        "宽深神经网络",
        "宽而深神经网络",
        "宽和深神经网络",
        "Wide and Deep Network"
    ],
    "WGAN": [
        "Wasserstein Generative Adversarial Network",
        "Wasserstein生成对抗网络",
        "沃瑟斯坦生成对抗网络",
        "WGAN"
    ],
    "XAI": [
        "Explainable Artificial Intelligence",
        "可解释人工智能",
        "可解释AI",
        "可解释的AI"
    ],
    "XLNET": [
        "eXtreme-Large NETwork",
        "XLNet模型",
        "广义自回归预训练语言模型",
        "极端大型网络"
    ],
    "YOLO": [
        "You Only Look Once",
        "YOLO算法",
        "你只看一次",
        "你只看一次算法"
    ]
}