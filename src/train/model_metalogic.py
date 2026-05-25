import torch
import torch.nn as nn
import torch.nn.functional as F

from metalogic.utils import set_seed, RunningMean
from metalogic.metalgoicnet import \
    LogicLerpInputLayer, LogicMetaLerpLayer, LerpChaining, \
    And_or_Not
from model_base import LearnerBase


class LerpModel(nn.Module):
    def __init__(self, rank, length, depth, width, n_relations,
                 init_var, temperature, dense, z_dim=100): # 添加时间相关参数
        super().__init__()
        self.n_variables = 1
        self.rank = rank
        self.length = length
        self.depth = depth
        self.width = width
        self.n_relations = n_relations
        self.init_var = init_var
        self.temperature = temperature
        self.dense = dense
        self.z_dim = z_dim  # 保存 z_dim
        self.build_model(n_relations)

    def build_model(self, relations):
        # Building Lerp
        self.lerp_input_layer = LogicLerpInputLayer(
            self.n_relations, self.width, self.dense)
        self.lerp_hidden_layers = nn.ModuleList([
            LogicMetaLerpLayer(self.width, self.n_relations,
                               ("And", "Or", "Chaining", "Copy", "Not"),
                               True, self.temperature, self.dense)
            for _ in range(max(0, self.depth-1))
        ])

        # Building Chains
        self.constraints = nn.ModuleList([
            And_or_Not(self.rank, self.width)
            for _ in range(2 * self.n_relations * (self.length + 1))
        ])
        self.chaining_layers = nn.ModuleList([
            LerpChaining(self.rank, self.n_relations, True, self.dense)
            for _ in range(2 * self.n_relations * self.length)
        ])

        hyper_input_dim = self.z_dim
        
        # liyongfang创建规则权重生成器 (HyperGAN部分)
        # 如果是时序模式，输入是 [c_q; e_t]，否则是噪声z
        self.rule_generator = nn.Sequential(
            nn.Linear(hyper_input_dim, 512),
            nn.LayerNorm(512), #LayerNorm 对 batch size 不敏感，适合小 batch 或变长 batch
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),  #将BatchNorm1d修改为LayerNorm
            nn.ELU(),
            nn.Linear(512, 2 * self.n_relations * self.rank)
        )

        # liyongfang初始化最后一层权重为0
        nn.init.zeros_(self.rule_generator[-1].weight)
        if self.rule_generator[-1].bias is not None:
            nn.init.zeros_(self.rule_generator[-1].bias)

        # 初始化其他参数
        for params in self.parameters():
            nn.init.normal_(params, 0, self.init_var)

    def interpret_lerp(self, layer, index, relations):
        '''这段代码定义了一个名为 interpret_lerp 的递归方法，用于解释一个特定层的 LERP（线性插值）模型的行为。
        该方法通过解析给定层的操作、权重及其与其他层的关系来生成人类可读的解释。'''
        if layer == -1:
            layer = self.depth-1
        assert self.n_relations == len(relations)
        if layer == 0:
            weight, subindex = self.lerp_input_layer.weights[
                :, index].softmax(0).max(0)
            if subindex >= self.n_relations:
                output = weight, relations[subindex - self.n_relations]+"^T"
            else:
                output = weight, relations[subindex]
        else:
            lerp_layer = self.lerp_hidden_layers[layer-1]
            op_weight, op_index = lerp_layer.op_weights[index].softmax(
                0).max(0)
            arg1_weight, arg1_index = lerp_layer.arg1_weights[
                :, index].softmax(0).max(0)
            arg2_weight, arg2_index = lerp_layer.arg2_weights[
                :, index].softmax(0).max(0)
            op = lerp_layer.ops[op_index]
            chain_weight, chain_index = lerp_layer.chain.weights[
                index].softmax(0).max(0)
            if chain_index >= self.n_relations:
                chain_rel = relations[chain_index-self.n_relations]+"^T"
            else:
                chain_rel = relations[chain_index]

            arg1_result = self.interpret_lerp(
                layer-1, arg1_index, relations)
            arg1_weight = arg1_weight * arg1_result[0]
            arg2_result = self.interpret_lerp(
                layer-1, arg2_index, relations)
            arg2_weight = arg2_weight * arg2_result[0]

            if op == "Copy":
                output = op_weight * arg2_weight, arg2_result[1]
            if op == "Chaining":
                output = op_weight * arg2_weight * chain_weight, \
                    f"{chain_rel}--{arg2_result[1]}"
            if op == "Not":
                output = op_weight * arg1_weight, f"Not({arg1_result[1]})"
            elif op == "And":
                output = op_weight * arg1_weight * arg2_weight,\
                    f"And({arg1_result[1]}, {arg2_result[1]})"
            elif op == "Or":
                output = op_weight * arg1_weight * arg2_weight,\
                    f"Or({arg1_result[1]}, {arg2_result[1]})"
        return output[0].item(), output[1]

    def interpret_rule(self, query, rule_index, relations):
        # liyongfang 使用固定z（全0）生成规则权重
        z = torch.zeros(1, self.z_dim, device=next(self.rule_generator.parameters()).device)
        flat_rule_weights = self.rule_generator(z)
        rule_weights = flat_rule_weights.view(2 * self.n_relations, self.rank)

        weight = rule_weights[query, rule_index].sigmoid().item
        text = "X"
        for i in range(self.length+1):
            if i > 0:
                _chain = self.chaining_layers[query*self.length+i-1]
                rel_weight, rel = _chain.weights[rule_index].softmax(0).max(0)
                equity_weight, equity = _chain.equity_weight[
                    rule_index].softmax(0).max(0)
                if equity == 1:
                    weight = weight * equity_weight
                else:
                    weight = weight * equity_weight * rel_weight
                    if rel >= self.n_relations:
                        rel_text = relations[rel-self.n_relations]
                        text = text + "--" + rel_text + "^T"
                    else:
                        rel_text = relations[rel]
                        text = text + "--" + rel_text

            if self.depth > 0:
                _constraint = self.constraints[query*(self.length+1)+i]
                gate_weight = _constraint.gate_weights[rule_index].sigmoid()
                if gate_weight > 0.5:
                    weight = weight * gate_weight
                else:
                    select_weight, select_index = _constraint.select_weights[
                        rule_index].softmax(0).max(0)
                    lerp_weight, lerp_text = self.interpret_lerp(
                        -1, select_index, relations)
                    weight = weight * (1-gate_weight) * lerp_weight * \
                        select_weight
                    text = text + f"(AND {lerp_text})"
        return weight.item(), text

    def forward(self, query, tails, heads, num_entity, database, timestamps=None):
        # 生成规则权重
        if self.training:
            # 训练时使用随机噪声
            z = torch.randn(1, self.z_dim, device=next(self.rule_generator.parameters()).device)
        else:
            # 测试时使用固定z（全0）
            z = torch.zeros(1, self.z_dim, device=next(self.rule_generator.parameters()).device)

        flat_rule_weights = self.rule_generator(z)
        rule_weights = flat_rule_weights.view(2 * self.n_relations, self.rank)

        tails_onehot = F.one_hot(tails, num_classes=num_entity)
        # Getting Lerp
        database = [database[i] for i in range(self.n_relations)]
        if self.dense:
            database = torch.stack(database).to_dense()
            if query < self.n_relations:
                database[query % self.n_relations, heads, tails] = 0
            else:
                database[query % self.n_relations, tails, heads] = 0
        lerp, quantified = self.lerp_input_layer(database)
        for _layer in self.lerp_hidden_layers:
            lerp = _layer(lerp, database)

        if self.depth > 0:
            x = self.constraints[query*(self.length+1)](
                tails_onehot[:, None], lerp)
        else:
            x = tails_onehot[:, None].expand(
                tails_onehot.shape[0], self.rank, num_entity).float()
        # Chaining
        chaining_layers = self.chaining_layers[
            query*self.length:(query+1)*self.length
        ]
        constraints = self.constraints[
            query*(self.length+1)+1:(query+1)*(self.length+1)
        ]
        for _chaining, _constraint in zip(
                chaining_layers, constraints):
            x = _chaining(x, database)
            if self.depth > 0:
                x = _constraint(x, lerp)

        # liyonfgang 使用生成的规则权重
        x = x * (rule_weights[query] - 2).sigmoid()[None, :, None]
        x = x.sum(1)

        return x


class Learner(LearnerBase):
    """
    This class builds a computation graph that represents the
    neural ILP model and handles related graph running acitivies,
    including update, predict, and get_attentions for given queries.

    Args:
        option: hyper-parameters
    """

    def __init__(self, option):
        super().__init__()
        self.seed = option.seed
        self.num_step = option.num_step
        self.rank = option.rank
        self.num_layer = option.num_layer
        self.rnn_state_size = option.rnn_state_size

        self.norm = not option.no_norm
        self.thr = option.thr
        self.dropout = option.dropout
        self.learning_rate = option.learning_rate
        self.accuracy = option.accuracy
        self.top_k = option.top_k

        self.num_entity = option.num_entity
        self.num_operator = option.num_operator
        self.query_is_language = option.query_is_language

        if not option.query_is_language:
            self.num_query = option.num_query
            self.query_embed_size = option.query_embed_size
        else:
            self.vocab_embed_size = option.vocab_embed_size
            self.query_embed_size = self.vocab_embed_size
            self.num_vocab = option.num_vocab
            self.num_word = option.num_word

        set_seed(self.seed)
        self.no_rev_in_model = option.no_rev_in_model
        self.num_relation = option.num_relation
        self.width = option.width
        self.length = option.length
        self.depth = option.depth
        self.init_var = option.init_var
        self.model_name = option.model
        self.soft_logic = option.soft_logic
        self.sparse = option.sparse
        self.dense = option.dense
        self.z_dim = option.z_dim if hasattr(option, 'z_dim') else 100  # 添加 z_dim
        
        # 获取时间戳信息（如果数据支持）
        self.num_timestamps = getattr(option, 'num_timestamps', 0)
        self.time_emb_dim = getattr(option, 'time_emb_dim', 50)
        self.use_temporal = getattr(option, 'use_temporal', False)

        self.lerp = LerpModel(
            self.rank, self.length, self.depth, self.width,
            self.num_relation, self.init_var, 1, self.dense, self.z_dim)
        self.running_mean = {
            split: {
                cat: RunningMean(0.97)
                for cat in ("loss", "in_top")
            } for split in ("train", "test")
        }

    def _inner_run_graph(self, queries, heads, tails, database):
        targets = F.one_hot(heads, num_classes=self.num_entity)
        _query = queries[0, 0]
        predictions = self.lerp(
            _query, tails, heads, self.num_entity, database)

        if self.norm:
            predictions /= predictions.sum(1).unsqueeze(1)

        final_loss = - torch.sum(
            targets * predictions.clamp_min(self.thr).log(), 1)

        if not self.accuracy:
            topk = predictions.topk(self.top_k)[1]
            in_top = (heads.unsqueeze(1) == topk).any(dim=1)
        else:
            _, indices = predictions.topk(self.top_k, sorted=False)[1]
            in_top = torch.squeeze(indices) == heads

        results = {
            "final_loss": final_loss,
            "in_top": in_top,
            "predictions": predictions,
        }
        return results

    def update(self, qq, hh, tt, mdb):
        results = self._run_graph(qq, hh, tt, mdb)
        self.running_mean["train"]["loss"].update(results["final_loss"].mean())
        self.running_mean["train"]["in_top"].update(
            results["in_top"].float().mean())
        return results["final_loss"], results["in_top"]

    def predict(self, qq, hh, tt, mdb):
        with torch.no_grad():
            results = self._run_graph(qq, hh, tt, mdb)
        self.running_mean["test"]["loss"].update(results["final_loss"].mean())
        self.running_mean["test"]["in_top"].update(
            results["in_top"].float().mean())
        return results["final_loss"], results["in_top"]

    def get_predictions_given_queries(self, qq, hh, tt, mdb):
        with torch.no_grad():
            results = self._run_graph(qq, hh, tt, mdb)
            return results["in_top"], results["predictions"]

    def get_attentions_given_queries(self, queries):
        with torch.no_grad():
            qq = queries
            hh = [0] * len(queries)
            tt = [0] * len(queries)
            mdb = {r: ([(0, 0)], [0.], (self.num_entity, self.num_entity))
                   for r in range(self.num_operator // 2)}
            results = self._run_graph(qq, hh, tt, mdb)
            return results["attention_operators"]

    def get_vocab_embedding(self):
        return self.vocab_embedding.weight
