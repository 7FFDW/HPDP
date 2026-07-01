import math

import torch
import torch.nn as nn

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, output_dim=512, model_name=r'E:\code\PMIL-main\models\Meimei\biobert'):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        bert_output_dim = self.bert.config.hidden_size
        self.projection = nn.Linear(bert_output_dim, output_dim)

    def forward(self, text_list, device):


        inputs = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            device)

        with torch.no_grad():
            outputs = self.bert(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :]


        projected_embedding = self.projection(cls_embedding)

        return projected_embedding





class FilmFusion(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        self.param_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim * 2)
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, patch_features, text_feature):

        params = self.param_generator(text_feature)
        gamma, beta = torch.chunk(params, 2, dim=-1)


        fused_features = gamma * patch_features + beta


        final_features = self.ln(patch_features + fused_features)

        return final_features



class MoEP(nn.Module):
    def __init__(self, num_prior_prototypes, num_adaptive_prototypes, embed_dim=512, temperature=0.1):

        super().__init__()


        self.prior_prototypes = nn.Parameter(torch.randn(num_prior_prototypes, embed_dim))


        self.adaptive_prototypes = nn.Parameter(torch.randn(num_adaptive_prototypes, embed_dim))

        self.temperature = temperature



    def forward(self, slide_features):

        slide_features_norm = F.normalize(slide_features, dim=1)
        prior_prototypes_norm = F.normalize(self.prior_prototypes, dim=1)
        cosine_sim_prior = torch.matmul(slide_features_norm, prior_prototypes_norm.T)
        routing_weights = F.softmax(cosine_sim_prior / self.temperature, dim=1)


        interaction_scores = torch.matmul(slide_features, self.adaptive_prototypes.T)
        transformation_weights = F.softmax(interaction_scores / self.temperature, dim=1)


        all_attn_weights = torch.cat([routing_weights, transformation_weights], dim=1)


        aggregated_prototypes = torch.matmul(all_attn_weights.T, slide_features)

        return aggregated_prototypes


class HierarchicalFusion(nn.Module):
    def __init__(self, embed_dim=512, num_supervised_prototypes=4, num_free_prototypes=12):
        super().__init__()
        self.num_supervised = num_supervised_prototypes
        self.num_free = num_free_prototypes
        self.num_prototypes = num_supervised_prototypes + num_free_prototypes





        self.fusion_module = FilmFusion(embed_dim)
        self.propagation_attn = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)



    def forward(self, aggregated_prototypes, text_feature):
        fused_prototypes = self.fusion_module(aggregated_prototypes, text_feature)
        slide_features_b = aggregated_prototypes.unsqueeze(0)
        fused_prototypes_b = fused_prototypes.unsqueeze(0)
        propagated_info, _ = self.propagation_attn(
            query=slide_features_b, key=fused_prototypes_b, value=fused_prototypes_b
        )
        final_features = self.ln(slide_features_b + propagated_info)
        return final_features.squeeze(0)





class FixedSinusoidalEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        if output_dim % 4 != 0:
            raise ValueError(f"output_dim must be divisible by 4, but got {output_dim}")

        self.output_dim = output_dim


        half_dim = output_dim // 2


        div_term = torch.exp(torch.arange(0, half_dim, 2).float() * (-math.log(10000.0) / half_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, coords):

        div_term_on_device = self.div_term.to(coords.device)

        pe_x = coords[:, 0:1] * div_term_on_device
        pe_y = coords[:, 1:2] * div_term_on_device

        sin_x = torch.sin(pe_x)
        cos_x = torch.cos(pe_x)
        sin_y = torch.sin(pe_y)
        cos_y = torch.cos(pe_y)

        return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)

class HPDP(nn.Module):
    def __init__(self, n_classes=2, dropout=0.25,num_supervised_prototypes=4,
                 num_free_prototypes=12):
        super(HPDP, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1



        self.visual_feature_extractor = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
            nn.Linear(1024, self.L)
        )

        self.pos_encoder = FixedSinusoidalEncoder(output_dim=self.L)
        self.text_encoder = TextEncoder(output_dim=self.L)



        self.moep = MoEP(num_supervised_prototypes, num_free_prototypes)


        self.fusion_module = HierarchicalFusion(
            embed_dim=self.L,
            num_supervised_prototypes=num_supervised_prototypes,
            num_free_prototypes=num_free_prototypes
        )



        self.attention_a = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh(), nn.Dropout(dropout))
        self.attention_b = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid(), nn.Dropout(dropout))
        self.attention_c = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, n_classes))






    def forward(self, x, coords, text):
        device = x.device


        initial_instance_features = self.visual_feature_extractor(x.squeeze(0))
        positional_embedding = self.pos_encoder(coords.squeeze(0))
        slide_features = initial_instance_features + positional_embedding

        global_text_feature = self.text_encoder(text, device)

        aggregated_prototypes = self.moep(slide_features)



        multimodal_features = self.fusion_module(aggregated_prototypes, global_text_feature)  


        a = self.attention_a(multimodal_features)
        b = self.attention_b(multimodal_features)
        A = a.mul(b)
        A = self.attention_c(A)
        A = torch.transpose(A, 0, 1)
        A = F.softmax(A, dim=1)
        bag_feature = torch.mm(A, multimodal_features)

        logits = self.classifier(bag_feature)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        return logits, Y_prob, Y_hat, A,bag_feature.mean(dim=0, keepdim=True)



