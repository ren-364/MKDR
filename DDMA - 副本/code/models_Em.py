import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution,  MultiHeadAttention
import math
from graph_models import OntologyEmbedding
from build_tree import *
from gnn import GNNGraph

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PKANet(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, num_heads=4, device=torch.device('cuda:0')):
        super(PKANet, self).__init__()
        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim*2) for i in range(K)]  # MultiHead
            )
        self.dropout = nn.Dropout(p=0.3)
        self.encoders = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim*2) for _ in range(K - 1)]
            )
        self.query1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 6, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, vocab_size[2])
        )


        # MultiHeadAtten
        self.MultiHead = nn.ModuleList(
            [MultiHeadAttention(emb_dim*2, num_heads=num_heads, dropout=0.0) for _ in range(4)]
        )

        self.pos_encoder = nn.ModuleList(
            [PositionalEncoding(emb_dim * 2, dropout=0.3) for _ in range(2)]
        )
        self.ninp = emb_dim * 2

    def forward(self, input):

        i1_seq = []
        i2_seq = []
        priorMed = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        for adm in input:
            src1 = self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device))
            src1 = self.pos_encoder[0](src1 * self.ninp)
            src2 = self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))
            src2 = self.pos_encoder[1](src2 * self.ninp)

            Med = self.embeddings[2](torch.LongTensor(adm[3] if len(adm[3]) > 0 else [0]).unsqueeze(dim=0).to(
                self.device))

            i1 = mean_embedding(self.dropout(src1))  # (1,1,dim)
            i2 = mean_embedding(self.dropout(src2))
            Med = mean_embedding(Med)

            i1_seq.append(i1)
            i2_seq.append(i2)
            priorMed.append(Med)

        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)
        # priorMed = torch.cat(priorMed, dim=1)  # (1,seq,dim)
        priorMed = Med  # (1,seq,dim)

        # intra Multihead
        o_1, diag_atten = self.MultiHead[0](i1_seq, i1_seq, i1_seq)
        o_2, pro_atten = self.MultiHead[1](i2_seq, i2_seq, i2_seq)

        priorKnowledge, _ = self.MultiHead[2](priorMed, priorMed, priorMed)
        priorKnowledge = priorKnowledge.squeeze(dim=0)

        patient_representations1 = torch.cat([o_1, o_2], dim=-1).squeeze(dim=0)  # (seq, dim*4)

        queries1 = self.query1(patient_representations1)  # (seq, dim)

        query1 = queries1[-1:]  # (1,dim)

        r1 = o_1.squeeze(dim=0)[-1:]
        r2 = o_2.squeeze(dim=0)[-1:]

        output = self.output(torch.cat([r1, r2, priorKnowledge], dim=-1))
        return output
class GetNorm(nn.Module):
    def __init__(self, p=1, dim=1):
        super(GetNorm, self).__init__()
        self.p = p
        self.dim = dim
    def forward(self, xtensor, ytensor):
        x = torch.norm(xtensor, p=self.p, dim=self.dim)
        y = torch.norm(ytensor, p=self.p, dim=self.dim)
        return torch.sqrt(x*y)


class MKDR(nn.Module):
    def  __init__(self, voc, vocab_size, patient_memory, ddi_adj, global_para, med2med, diag2med, ehr_adj, emd_dim=128, num_heads=4, device=torch.device('cuda:0')):
        super(MKDR, self).__init__()

        K = len(vocab_size)
        self.K = K
        self.vocab_size = vocab_size
        self.device = device
        self.embeddings = nn.Embedding(vocab_size[2], emd_dim*2)
        self.dropout = nn.Dropout(p=0.3)
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.query1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emd_dim * 4, emd_dim * 2),  # GRU、Attention
        )

        #self.global_encoder = GNNGraph(**global_para)
        self.ehr_gcn = GCN(voc_size=vocab_size[2], emd_dim=emd_dim*2, adj=ehr_adj)

        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emd_dim * 12, emd_dim * 8),
            nn.ReLU(),
            nn.Linear(emd_dim * 8, vocab_size[2])
        )

        #self.encoders = nn.ModuleList([nn.GRU(emd_dim * 2, emd_dim * 2, batch_first=True) for _ in range(K - 1)])
        #self.attention = nn.ModuleList(
        #    [Attention(emd_dim*2) for _ in range(2)]
        #)

        #共现矩阵
        self.med2med = med2med
        self.diag2med = diag2med
        #self.pro2med = pro2med

        # MultiHeadAttention
        self.MultiHead = nn.ModuleList(
            [MultiHeadAttention(emd_dim*2, num_heads=num_heads, dropout=0.0) for _ in range(3)]
        )

        self.patient_memory = patient_memory
        self.voc = voc
        self.ontologyEmd = nn.ModuleDict({
            'diag': OntologyEmbedding(voc['diag_voc'], build_icd9_tree),
            'pro': OntologyEmbedding(voc['pro_voc'], build_icd9_pro_tree),
            #'med': OntologyEmbedding(voc['med_voc'], build_atc_tree)
        })

        #初步筛选其他患者相似度时diag和pro的比重
        self.a = 0.8
        #结合药物分子信息和编码层次信息
        #self.inter = nn.Parameter(torch.FloatTensor(1))

        #取query的L1正则化
        self.get_norm = GetNorm()
        #self.h = torch.FloatTensor(1)

    def forward(self, input, mol_data, average_projection):

        i1_seq = []
        i2_seq = []


        #获取层次编码
        diagEmd, proEmd = self.ontologyEmd['diag'], self.ontologyEmd['pro']
        diagemdding, proemdding = diagEmd(), proEmd()

        #获取药物分子结构编码
        #global_embeddings = self.global_encoder(**mol_data)
        #global_embeddings = torch.mm(average_projection, global_embeddings)

        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)

        def jaccard(inter, l_cdiag, l_ndiag):
            return inter / min(l_cdiag, l_ndiag)

        def get_patient_memory():
            #病人当前状态信息
            cur_diag = input[-1][0]
            l_cdiag = len(cur_diag)
            cur_pro = input[-1][1]
            l_cpro = len(cur_diag)
            p_id = input[0][3]
            p_age = input[-1][4]
            #初步计算相似度
            b_patient_memory = []
            b_diag_memory = []
            b_pro_memory = []
            b_idx_memory = []
            patient_id = []
            for patient in self.patient_memory:
                patient_id.append(patient[0][3])
                for adm_idx, adm in enumerate(patient):
                    if adm[3] != p_id and adm[4] == p_age:
                        b_patient_memory.append(adm)
                        b_diag_memory.append(adm[0])
                        b_pro_memory.append(adm[1])
                        b_idx_memory.append(adm_idx+1)
            score_diag = []
            score_pro = []
            for diag in b_diag_memory:
                inter = len(set(diag) & set(cur_diag))
                l_ndiag = len(set(diag))
                jaccard_score = 0 if inter == 0 else jaccard(inter, l_cdiag, l_ndiag)
                score_diag.append(jaccard_score)
            for pro in b_pro_memory:
                inter_ = len(set(pro) & set(cur_pro))
                l_npro = len(set(pro))
                jaccard_score = 0 if inter_ == 0 else jaccard(inter_, l_cpro, l_npro)
                score_pro.append(jaccard_score)
            diagScore = torch.tensor(score_diag).to(self.device)
            proScore = torch.tensor(score_pro).to(self.device)
            score = self.a * diagScore + (1 - self.a) * proScore
            _, topk_indices = torch.topk(score, 10)

            #选出健康状态相似的患者，并算出他们所有visit的表示
            c_patient_memory = []
            c_patient_id = [b_patient_memory[i][3] for i in topk_indices]
            c_visit_id = [b_idx_memory[i] for i in topk_indices]
            indices = [patient_id.index(i) for i in c_patient_id]
            for idx, item in enumerate(indices):
                c_patient_memory.append(self.patient_memory[item][:c_visit_id[idx]])

            query_memory = []
            value_memory = torch.zeros(10, self.vocab_size[2]).to(self.device)

            for idx,patient in enumerate(c_patient_memory):
                i1_seq = []
                i2_seq = []
                for adm in patient:
                    src1 = mean_embedding(diagemdding[adm[0]].unsqueeze(dim=0))
                    src2 = mean_embedding(proemdding[adm[1]].unsqueeze(dim=0))
                    i1_seq.append(src1)
                    i2_seq.append(src2)
                i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
                i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

                o_1, diag_atten = self.MultiHead[0](i1_seq, i1_seq, i1_seq)
                o_2, pro_atten = self.MultiHead[1](i2_seq, i2_seq, i2_seq)
                patient_representations1 = torch.cat([o_1, o_2], dim=-1).squeeze(dim=0)  # (seq, dim*4)
                queries1 = self.query1(patient_representations1)  # (seq, dim)

                query_memory.append(queries1[-1:])
                value_memory[idx, patient[-1][2]] = 1
            query_memory_ = torch.stack(query_memory).squeeze()
            return query_memory_, value_memory

        #采用共现矩阵得到的先验药物
        cur_diag = np.zeros((1, self.vocab_size[0]))
        cur_diag[:, input[-1][0]] = 1
        prior_med_id = torch.nonzero(torch.mm(torch.tensor(cur_diag).to(self.device), self.diag2med) > 0)[:,-1]
        priorMed = self.embeddings(prior_med_id if len(prior_med_id) > 0 else torch.LongTensor([0]).to(self.device)).mean(dim=0, keepdim=True).unsqueeze(dim=0)
        priorKnowledge, _ = self.MultiHead[2](priorMed, priorMed, priorMed)
        priorKnowledge = priorKnowledge.squeeze(dim=0)

        #获取患者表示
        for adm in input:
            src1 = mean_embedding(diagemdding[adm[0]].unsqueeze(dim=0))
            src2 = mean_embedding(proemdding[adm[1]].unsqueeze(dim=0))
            i1_seq.append(src1)
            i2_seq.append(src2)
        i1_seq = torch.cat(i1_seq, dim=1)  # (1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1,seq,dim)

        # intra Multihead
        o_1, diag_atten = self.MultiHead[0](i1_seq, i1_seq, i1_seq)
        o_2, pro_atten = self.MultiHead[1](i2_seq, i2_seq, i2_seq)
        patient_representations1 = torch.cat([o_1, o_2], dim=-1).squeeze(dim=0)  # (seq, dim*4)
        queries1 = self.query1(patient_representations1)  # (seq, dim)
        query1 = queries1[-1:]  # (1,dim)

        # 获取全局患者相似度
        query_memory, value_memory = get_patient_memory()
        norm2 = self.get_norm(query1, query_memory)
        key_weights_memory = torch.mm(query1, query_memory.t())/norm2
        #topk_weight, topk_indices = torch.topk(key_weights_memory, 10)
        key_memory = F.softmax(key_weights_memory, dim=-1)
        #value_memory = value_memory[topk_indices, :].squeeze(dim=0)
        weighted_values_memory = key_memory.mm(value_memory)

        # 患者标准信息
        r1 = o_1.squeeze(dim=0)[-1:]
        r2 = o_2.squeeze(dim=0)[-1:]

        # 获取药物HIMI编码
        drug_memory = self.ehr_gcn()

        # 获取历史患者表征
        if len(input) > 1:
            history_keys1 = queries1[:(queries1.size(0)-1)]  # (seq-1, dim)
            history_values = np.zeros((len(input)-1, self.vocab_size[2]))
            for idx, adm in enumerate(input):
                if idx == len(input)-1:
                    break
                history_values[idx, adm[2]] = 1
            history_values = torch.FloatTensor(history_values).to(self.device)  # (seq-1, size)

        # 算当前表征匹配药物
        key_weights1_1 = F.softmax(torch.mm(query1, drug_memory.t()), dim=-1)  # (1, size) 与卷积得到的药物编码算一次权重
        fact1_1 = torch.mm(key_weights1_1, drug_memory)  # (1, dim)
        fact_memory = torch.mm(weighted_values_memory, drug_memory)
        # 患者历史表征相似度
        if len(input) > 1:
            norm1 = self.get_norm(query1, history_keys1)
            visit_weight1 = F.softmax(torch.mm(query1, history_keys1.t())/norm1)   # (1, seq-1)
            weighted_values1 = visit_weight1.mm(history_values)  # (1, size)
            fact2_1 = torch.mm(weighted_values1, drug_memory)  # (1, dim)
        else:
            fact2_1 = fact_memory

        #fact_memory = torch.mm(weighted_values_memory, drug_memory)  # (1, dim)

        # 输出结果
        output = self.output(torch.cat([r1, r2, priorKnowledge, fact1_1, fact2_1, fact_memory], dim=-1))  # (1, dim)
        neg_pred_prob = F.sigmoid(output)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()
        return output, batch_neg

class TAMSGC(nn.Module):
    def __init__(self, size, ehr_adj, ddi_adj, emb_dim=64, device=torch.device('cuda:0'), ddi_in_memory=True):
        super(TAMSGC, self).__init__()
        K = len(size)
        self.K = K
        self.vocab_size = size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.ddi_in_memory = ddi_in_memory
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size[i], emb_dim) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.3)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim*2) for _ in range(K - 1)])
        self.alpha = nn.Linear(emb_dim*2, 1)
        self.beta = nn.Linear(emb_dim*2, emb_dim)
        self.Reain_output = nn.Linear(emb_dim, emb_dim*2)
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim*4, emb_dim),
        )
        self.ehr_gnn = SGC(size=size[2], emb_dim=emb_dim, adj=ehr_adj,  device=device)  # GNN for EHR
        self.ddi_gnn = SGC(size=size[2], emb_dim=emb_dim, adj=ddi_adj,  device=device)  # GNN for DDI
        self.lambda_ = nn.Parameter(torch.FloatTensor(1))
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, size[2])
        )

        self.init_weights()

    def forward(self, patient):

#        the embeddings of diagnosis and procedure
        diagnosis_seq = []
        procedure_seq = []
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)  # (1,1,dim)
        for admission_time in patient:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(admission_time[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(admission_time[1]).unsqueeze(dim=0).to(self.device))))
            diagnosis_seq.append(i1)
            procedure_seq.append(i2)
        diagnosis_seq = torch.cat(diagnosis_seq, dim=1)  #(1,seq,dim)
        procedure_seq = torch.cat(procedure_seq, dim=1)  #(1,seq,dim)

        # Temporal Attention Mechanism
        diagnosis_output, diagnosis_hidden = self.encoders[0](
            diagnosis_seq
        )  # for diagnosis
        attn_o1_alpha = F.tanhshrink(self.alpha(diagnosis_output))  # (visit, 1)
        attn_o1_beta = F.tanh(self.beta(diagnosis_output))
        diagnosis_output = attn_o1_alpha * attn_o1_beta * diagnosis_seq  # (visit, emb)
        diagnosis_output = torch.sum(diagnosis_output, dim=0).unsqueeze(dim=0)  # (1, emb)
        diagnosis_output = self.Reain_output(diagnosis_output)

        procedure_output, procedure_hidden = self.encoders[1](
            procedure_seq
        )  # for procedure
        attn_o2_alpha = F.tanhshrink(self.alpha(procedure_output))
        attn_o2_beta = F.tanh(self.beta(procedure_output))
        procedure_output = attn_o2_alpha * attn_o2_beta * procedure_seq  # (visit, emb)
        procedure_output = torch.sum(procedure_output, dim=0).unsqueeze(dim=0)  # (1, emb)
        procedure_output = self.Reain_output(procedure_output)

        # RNN
        # diagnosis_output, diagnosis_hidden = self.encoders[0](
        #     diagnosis_seq
        # )  # for dia
        # procedure_output, procedure_hidden = self.encoders[1](
        #     procedure_seq
        # )  # for procedure

        patient_representations = torch.cat([diagnosis_output, procedure_output], dim=-1).squeeze(dim=0)  # (seq, dim*4)
        queries = self.query(patient_representations)  # (seq, dim)
        P = queries[-1:]  # (1,dim)

        #  medication representation
        if self.ddi_in_memory:
            medication_representation_K = self.ehr_gnn() - self.ddi_gnn() * self.lambda_  # (size, dim)
        else:
            medication_representation_K = self.ehr_gnn()

        if len(patient) > 1:
            history_P = queries[:(queries.size(0)-1)]  # (seq-1, dim)
            history_medication = np.zeros((len(patient)-1, self.vocab_size[2]))
            for idx, adm in enumerate(patient):
                if idx == len(patient)-1:
                    break
                history_medication[idx, adm[2]] = 1
            history_medication = torch.FloatTensor(history_medication).to(self.device)  # (seq-1, size)

        weights1 = F.softmax(torch.mm(P, medication_representation_K.t()), dim=-1)  # (1, size)
        R1 = torch.mm(weights1, medication_representation_K)  # (1, dim)

        if len(patient) > 1:
            weight2 = F.softmax(torch.mm(P, history_P.t()))  # (1, seq-1)
            weighted_values = weight2.mm(history_medication)  # (1, size)
            R2 = torch.mm(weighted_values, medication_representation_K)  # (1, dim)
        else:
            R2 = R1

        output = self.output(torch.cat([P, R1, R2], dim=-1))  # (1, dim)

        if self.training:
            neg_pred_prob = F.sigmoid(output)
            neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
            batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

            return output, batch_neg
        else:
            return output

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        self.lambda_.data.uniform_(-initrange, initrange)

class SGC(nn.Module):
    def __init__(self, size, emb_dim, adj, device=torch.device('cuda')):
        super(SGC, self).__init__()
        self.voc_size = size
        self.emb_dim = emb_dim
        self.device = device
        self.x = torch.eye(size).to(device)
        # adj = self.normalize(adj + np.eye(adj.shape[0]))
        # self.x = torch.FloatTensor(adj).to(device)
        self.W = nn.Linear(size, emb_dim)

    def forward(self):
        #print(self.x)
        return self.W(self.x)

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GCN(nn.Module):
    def __init__(self, voc_size, emd_dim, adj,  device=torch.device('cuda:0')):
        super(GCN, self).__init__()
        self.voc_size = voc_size
        self.emd_dim = emd_dim
        self.device = device

        adj = self.normalize(adj        + np.eye(adj.shape[0]))

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)
        # self.gcns = nn.ModuleList()
        # self.gcn = GraphConvolution(voc_size, emb_dim)
        self.gcn1 = GraphConvolution(voc_size, emd_dim)
        self.dropout = nn.Dropout(p=0.3)
        # for i in range(layers-1):
        #     self.gcns.append(GraphConvolution(emb_dim, emb_dim))
        self.gcn2 = GraphConvolution(emd_dim, emd_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)

        # node_embedding = self.gcn(self.x, self.adj)
        # for indx in range(len(self.gcns)):
        #     node_embedding = F.relu(node_embedding)
        #     node_embedding = self.dropout(node_embedding)
        #     node_embedding = self.gcns[indx](node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

class GCN_1(nn.Module):
    def __init__(self, voc_size, emb_dim, adj, device=torch.device('cuda:0')):
        super(GCN_1, self).__init__()
        self.voc_size = voc_size
        self.emb_dim = emb_dim
        self.device = device

        # adj = self.normalize(np.ones(adj.shape[0]) - adj - np.eye(adj.shape[0]))
        adj = self.normalize(adj + np.eye(adj.shape[0]))
        # adj = self.normalize(adj)

        self.adj = torch.FloatTensor(adj).to(device)
        self.x = torch.eye(voc_size).to(device)

        self.gcn1 = GraphConvolution(voc_size, emb_dim)
        self.dropout = nn.Dropout(p=0.3)
        self.gcn2 = GraphConvolution(emb_dim, emb_dim)

    def forward(self):
        node_embedding = self.gcn1(self.x, self.adj)
        node_embedding = F.relu(node_embedding)
        node_embedding = self.dropout(node_embedding)
        node_embedding = self.gcn2(node_embedding, self.adj)
        return node_embedding

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diagflat(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx






