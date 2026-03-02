import torch 
from torch import nn, optim
import numpy as np
from abc import ABC, abstractmethod
import pytorch_lightning as pl
from utils import AllGather, CrossEn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from models.cluster import CTM, TCBlock
import torch.nn.functional as F
import os
import glob
from utils import fingerprint_metrics, mol_translation_metrics, retrieva_metrics, text_translation_metrics
allgather = AllGather.apply

class BaseModel(pl.LightningModule, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        self.gen_valid_start_epoch = getattr(args, "gen_valid_start_epoch", 45)
        self.gen_valid_interval = getattr(args, "gen_valid_interval", 5)
        
    @abstractmethod
    def forward(
        self, text, smiles
    ):
        return NotImplemented
    
    def training_step(self, batch, batch_idx):
        
        loss = self(
                batch["description"], batch["smiles"], self.args.alpha
            )        

        self.log("train_loss", loss, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True, on_step=True)
        
        return loss

    def _should_run_gen_valid(self):
        if self.args.mode == "eval":
            return True
        current_epoch = self.trainer.current_epoch
        if current_epoch >= self.args.start_valid_epoch:
            return True
        if current_epoch < self.gen_valid_start_epoch:
            return False
        return (current_epoch - self.gen_valid_start_epoch) % self.gen_valid_interval == 0
    
    def validation_step(self, batch, batch_idx):
        # extract text and smiles feature for retrieval
        result = {}
        text_feat, smiles_feat = self.get_text_smiles_feat(batch["description"], batch["smiles"])
        text_feat_all = allgather(text_feat)
        smiles_feat_all = allgather(smiles_feat)
        
        result = {
            "text_feat":text_feat_all,
            "smiles_feat":smiles_feat_all,
        }
        
        # for generation
        if self._should_run_gen_valid():
            if self.trainer.local_rank==0:
                if not os.path.exists(self.args.temp_dir):
                    os.mkdir(self.args.temp_dir) 
            if self.args.task == "genmol":
                texts = []
                gts = []
                outputs = []
                
                text = batch["description"]
                smiles = batch["smiles"]
                input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.molt5.device)
                model_outputs = self.molt5_m.generate(input_ids, num_beams=5, max_length=512)
                generate_smiles = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
                texts.append(text[0])
                gts.append(smiles[0])
                outputs.append(generate_smiles)

                file_name = f'valid_tmp_{self.trainer.global_rank}_epoch={self.trainer.current_epoch}.txt'
                file_pt = os.path.join(self.args.temp_dir, file_name)
                with open (file_pt,"a") as f:
                    for row in zip(texts, gts, outputs):
                        f.write('\t'.join(row) + '\n')
            
            else:
                smiles = []
                gts = []
                outputs = []
                
                text = batch["description"]
                smiles = batch["smiles"]
                input_ids = self.tokenizer(smiles, return_tensors="pt").input_ids.to(self.molt5.device)
                model_outputs = self.molt5_m.generate(input_ids, num_beams=5, max_length=512)
                generate_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
                smiles.append(smiles[0])
                gts.append(text[0])
                outputs.append(generate_text)

                file_name = f'valid_tmp_{self.trainer.global_rank}_epoch={self.trainer.current_epoch}.txt'
                file_pt = os.path.join(self.args.temp_dir, file_name)
                with open (file_pt,"a") as f:
                    for row in zip(smiles, gts, outputs):
                        f.write('\t'.join(row) + '\n')
                        
            torch.distributed.barrier()
        
        return result

    def validation_epoch_end(self, validation_step_outputs):
        sim_matrix = []
        text_feats, smiles_feats = [], []   # [len(validset), dim]
        for batch in validation_step_outputs:
            text_feats.extend(batch["text_feat"])
            smiles_feats.extend(batch["smiles_feat"])
        all_text_feats = torch.stack(text_feats, dim=0)
        all_smiles_feats = torch.stack(smiles_feats, dim=0)
        batch_text_feat = torch.split(all_text_feats, 32)
        batch_smiles_feat = torch.split(all_smiles_feats, 32)
        
        with torch.no_grad():
            for idx1, text_feat in enumerate(batch_text_feat):
                each_row = []
                for idx2, smiles_feat in enumerate(batch_smiles_feat):
                    logits = text_feat @ smiles_feat.t()
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        ts_metrics = retrieva_metrics(sim_matrix)
        st_metrics = retrieva_metrics(sim_matrix.T)
        
        self.log("valid_TtS_R@1", ts_metrics['R1'], on_epoch=True)
        self.log("valid_TtS_R@5", ts_metrics['R5'], on_epoch=True)
        self.log("valid_TtS_R@10", ts_metrics['R10'], on_epoch=True)
        self.log("valid_TtS_R@50", ts_metrics['R50'], on_epoch=True)
        self.log("valid_TtS_MedianR", ts_metrics['MedianR'], on_epoch=True)
        self.log("valid_TtS_MeanR", ts_metrics['MeanR'], on_epoch=True)
        
        self.log("valid_StT_R@1", st_metrics['R1'], on_epoch=True)
        self.log("valid_StT_R@5", st_metrics['R5'], on_epoch=True)
        self.log("valid_StT_R@10", st_metrics['R10'], on_epoch=True)
        self.log("valid_StT_R@50", st_metrics['R50'], on_epoch=True)
        self.log("valid_StT_MedianR", st_metrics['MedianR'], on_epoch=True)
        self.log("valid_StT_MeanR", st_metrics['MeanR'], on_epoch=True)

        
        if self._should_run_gen_valid() & (self.args.mode == "train"):

            # merge prediction files
            search_name = f'valid_tmp_*_epoch={self.trainer.current_epoch}.txt'
            gather_name = f'valid_predict_all_epoch={self.trainer.current_epoch}.txt'
            eval_result_name = f'valid_eval_all_epoch={self.trainer.current_epoch}.txt'
            
            file_paths = glob.glob(os.path.join(self.args.temp_dir, search_name))
            gather_file_path = os.path.join(self.args.temp_dir, gather_name)
            if self.args.task =="genmol":
                with open(gather_file_path, "w") as f:
                    f.write('description\tground truth\toutput\n')
                    for file_path in file_paths:
                        with open(file_path, "r") as file:
                            f.write(file.read())
            else:
                with open(gather_file_path, "w") as f:
                    f.write('SMILES\tground truth\toutput\n')
                    for file_path in file_paths:
                        with open(file_path, "r") as file:
                            f.write(file.read())
            torch.distributed.barrier()
                 
            # eval metric
            eval_file_path = os.path.join(self.args.temp_dir, eval_result_name)
            log_io = open(eval_file_path, 'w+')
            if self.args.task =="genmol":
                bleu_score, exact_match_score, levenshtein_score,_ = mol_translation_metrics.evaluate(gather_file_path)
                validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fingerprint_metrics.evaluate(gather_file_path, 2)
                # fcd_metric_score = fcd_metric.evaluate(file_path)
                log_io.write(f'For {self.args.version}_epoch={self.trainer.current_epoch}\n')
                log_io.write(f'BLEU: {round(bleu_score, 3)}\n')
                log_io.write(f'Exact: {round(exact_match_score, 3)}\n')
                log_io.write(f'Levenshtein: {round(levenshtein_score, 3)}\n')
                log_io.write(f'MACCS FTS: {round(maccs_sims_score, 3)}\n')
                log_io.write(f'RDK FTS: {round(rdk_sims_score, 3)}\n')
                log_io.write(f'Morgan FTS: {round(morgan_sims_score, 3)}\n')
                # log_io.write(f'FCD Metric: {round(fcd_metric_score, 3)}\n')
                log_io.write(f'Validity: {round(validity_score, 3)}\n')
                log_io.write('\n')
                log_io.flush()
                log_io.close()      
                self.log("valid_bleu_score", round(bleu_score, 3), on_epoch=True)
                self.log("valid_Exact", round(exact_match_score, 3), on_epoch=True)
                self.log("valid_levenshtein_score", round(levenshtein_score, 3), on_epoch=True)
                self.log("valid_maccs_sims_score", round(maccs_sims_score, 3), on_epoch=True)
                self.log("valid_rdk_sims_score", round(rdk_sims_score, 3), on_epoch=True)
                self.log("valid_morgan_sims_score", round(morgan_sims_score, 3), on_epoch=True)     
                self.log("valid_validity_score", round(validity_score, 3), on_epoch=True)
            else:
                bleu2, bleu4, rouge_1, rouge_2, rouge_l = text_translation_metrics.evaluate(
                                                                        str(self.args.data_dir) + "/pretrained/SciBERT", gather_file_path, 512)    
                log_io.write(f'For {self.args.version}_epoch={self.trainer.current_epoch}\n')
                log_io.write(f'BLEU2: {round(bleu2, 3)}\n')
                log_io.write(f'BLEU4: {round(bleu4, 3)}\n')
                log_io.write(f'ROUGE1: {round(rouge_1, 3)}\n')
                log_io.write(f'ROUGE2: {round(rouge_2, 3)}\n')
                log_io.write(f'ROUGEL: {round(rouge_l, 3)}\n')
                log_io.write('\n')
                log_io.flush()
                log_io.close()      
                self.log("valid_BLEU2", round(bleu2, 3), on_epoch=True)
                self.log("valid_BLEU4", round(bleu4, 3), on_epoch=True)
                self.log("valid_ROUGE1", round(rouge_1, 3), on_epoch=True)
                self.log("valid_ROUGE2", round(rouge_2, 3), on_epoch=True)
                self.log("valid_ROUGEL", round(rouge_l, 3), on_epoch=True)          
        else:
            if self.args.task =="genmol":
                self.log("valid_bleu_score", 0, on_epoch=True)
                self.log("valid_Exact", 0, on_epoch=True)
                self.log("valid_levenshtein_score", 0, on_epoch=True)
                self.log("valid_maccs_sims_score", 0, on_epoch=True)
                self.log("valid_rdk_sims_score", 0, on_epoch=True)
                self.log("valid_morgan_sims_score", 0, on_epoch=True)     
                self.log("valid_validity_score", 0, on_epoch=True)   
            else:
                self.log("valid_BLEU2", 0, on_epoch=True)
                self.log("valid_BLEU4", 0, on_epoch=True)
                self.log("valid_ROUGE1", 0, on_epoch=True)
                self.log("valid_ROUGE2", 0, on_epoch=True)
                self.log("valid_ROUGEL", 0, on_epoch=True)
                 
    def test_step(self, batch, batch_idx):
        # extract text and smiles feature for retrieval on single gpu
        if not os.path.exists(self.args.temp_dir):
            os.mkdir(self.args.temp_dir)
        result = {}
        text_feat, smiles_feat = self.get_text_smiles_feat(batch["description"], batch["smiles"])
        
        result = {
            "text_feat":text_feat,
            "smiles_feat":smiles_feat,
        }
        
        # for generation
        if self.args.task == "genmol":
            texts = []
            gts = []
            outputs = []
            
            text = batch["description"]
            smiles = batch["smiles"]
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.molt5.device)
            model_outputs = self.molt5_m.generate(input_ids, num_beams=5, max_length=512)
            generate_smiles = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            texts.append(text[0])
            gts.append(smiles[0])
            outputs.append(generate_smiles)

            file_name = f'test_tmp_epoch={self.trainer.current_epoch}.txt'
            file_pt = os.path.join(self.args.temp_dir, file_name)
            with open (file_pt,"a") as f:
                if batch_idx==0:
                    f.write('description\tground truth\toutput\n')
                for row in zip(texts, gts, outputs):
                    f.write('\t'.join(row) + '\n')
        
        else:
            smiles = []
            gts = []
            outputs = []
            
            text = batch["description"]
            smiles = batch["smiles"]
            input_ids = self.tokenizer(smiles, return_tensors="pt").input_ids.to(self.molt5.device)
            model_outputs = self.molt5_m.generate(input_ids, num_beams=5, max_length=512)
            generate_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            smiles.append(smiles[0])
            gts.append(text[0])
            outputs.append(generate_text)

            file_name = f'test_tmp_epoch={self.trainer.current_epoch}.txt'
            file_pt = os.path.join(self.args.temp_dir, file_name)
            with open (file_pt,"a") as f:
                if batch_idx==0:
                    f.write('SMILES\tground truth\toutput\n')
                for row in zip(smiles, gts, outputs):
                    f.write('\t'.join(row) + '\n')
        
        return result

    def test_epoch_end(self, test_step_outputs):

        sim_matrix = []
        text_feats, smiles_feats = [], []   # [len(validset), dim]
        for batch in test_step_outputs:
            text_feats.extend(batch["text_feat"])
            smiles_feats.extend(batch["smiles_feat"])
        all_text_feats = torch.stack(text_feats, dim=0)
        all_smiles_feats = torch.stack(smiles_feats, dim=0)
        batch_text_feat = torch.split(all_text_feats, 32)
        batch_smiles_feat = torch.split(all_smiles_feats, 32)
        
        with torch.no_grad():
            for idx1, text_feat in enumerate(batch_text_feat):
                each_row = []
                for idx2, smiles_feat in enumerate(batch_smiles_feat):
                    logits = text_feat @ smiles_feat.t()
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        ts_metrics = retrieva_metrics(sim_matrix)
        st_metrics = retrieva_metrics(sim_matrix.T)

        eval_retrieval_name = f'test_result_retrieval_eval_all_epoch={self.trainer.current_epoch}.txt'
        eval_retrieval_file_path = os.path.join(self.args.temp_dir, eval_retrieval_name)
        log_io = open(eval_retrieval_file_path, 'w+')
        
        log_io.write(f'For {self.args.version}_epoch={self.trainer.current_epoch}\n')
        log_io.write(f'=======Text to Smiles======= \n')
        log_io.write(f'test_TtS_R@1: {round(ts_metrics["R1"], 3)}\n')
        log_io.write(f'test_TtS_R@5: {round(ts_metrics["R5"], 3)}\n')
        log_io.write(f'test_TtS_R@10: {round(ts_metrics["R10"], 3)}\n')
        log_io.write(f'test_TtS_R@50: {round(ts_metrics["R50"], 3)}\n')
        log_io.write(f'test_TtS_MedianR: {round(ts_metrics["MedianR"], 3)}\n')
        log_io.write(f'test_TtS_MeanR: {round(ts_metrics["MeanR"], 3)}\n')
        log_io.write(f'=======Smiles to Text======= \n')
        log_io.write(f'test_StT_R@1: {round(st_metrics["R1"], 3)}\n')
        log_io.write(f'test_StT_R@5: {round(st_metrics["R5"], 3)}\n')
        log_io.write(f'test_StT_R@10: {round(st_metrics["R10"], 3)}\n')
        log_io.write(f'test_StT_R@50: {round(st_metrics["R50"], 3)}\n')
        log_io.write(f'test_StT_MedianR: {round(st_metrics["MedianR"], 3)}\n')
        log_io.write(f'test_StT_MeanR: {round(st_metrics["MeanR"], 3)}\n')
        log_io.write('\n')
        log_io.flush()
        log_io.close()  
    
        # search prediction files
        search_name = f'test_tmp_epoch={self.trainer.current_epoch}.txt'
        eval_result_name = f'test_result_eval_all_epoch={self.trainer.current_epoch}.txt'
        file_path = os.path.join(self.args.temp_dir, search_name)
        eval_file_path = os.path.join(self.args.temp_dir, eval_result_name)
        log_io = open(eval_file_path, 'w+')
        
        if self.args.task == "genmol":
            # eval metric
            bleu_score, exact_match_score, levenshtein_score,_ = mol_translation_metrics.evaluate(file_path)
            validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score = fingerprint_metrics.evaluate(file_path, 2)
            # fcd_metric_score = fcd_metric.evaluate(file_path)
            log_io.write(f'For {self.args.version}_epoch={self.trainer.current_epoch}\n')
            log_io.write(f'BLEU: {round(bleu_score, 3)}\n')
            log_io.write(f'Exact: {round(exact_match_score, 3)}\n')
            log_io.write(f'Levenshtein: {round(levenshtein_score, 3)}\n')
            log_io.write(f'MACCS FTS: {round(maccs_sims_score, 3)}\n')
            log_io.write(f'RDK FTS: {round(rdk_sims_score, 3)}\n')
            log_io.write(f'Morgan FTS: {round(morgan_sims_score, 3)}\n')
            # log_io.write(f'FCD Metric: {round(fcd_metric_score, 3)}\n')
            log_io.write(f'Validity: {round(validity_score, 3)}\n')
            log_io.write('\n')
            log_io.flush()
            log_io.close()
        else:
            bleu2, bleu4, rouge_1, rouge_2, rouge_l = text_translation_metrics.evaluate(
                                                                    str(self.args.data_dir) + "/pretrained/SciBERT", file_path, 512)    
            log_io.write(f'For {self.args.version}_epoch={self.trainer.current_epoch}\n')
            log_io.write(f'BLEU2: {round(bleu2, 3)}\n')
            log_io.write(f'BLEU4: {round(bleu4, 3)}\n')
            log_io.write(f'ROUGE1: {round(rouge_1, 3)}\n')
            log_io.write(f'ROUGE2: {round(rouge_2, 3)}\n')
            log_io.write(f'ROUGEL: {round(rouge_l, 3)}\n')
            log_io.write('\n')
            log_io.flush()
            log_io.close()            
    
    def predict_step(self, batch, batch_idx):    
        if self.args.task == "genmol":
            texts = []
            gts = []
            outputs = []
            
            text = batch["description"]
            smiles = batch["smiles"]
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.molt5.device)
            model_outputs = self.molt5_m.generate(input_ids, num_beams=5, max_length=512)
            generate_smiles = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            texts.append(text[0])
            gts.append(smiles[0])
            outputs.append(generate_smiles)

            file_name = f'tmp_{self.trainer.global_rank}.txt'
            file_pt = os.path.join(self.args.temp_dir, file_name)
            with open (file_pt,"a") as f:
                for row in zip(texts, gts, outputs):
                    f.write('\t'.join(row) + '\n')
        else:
            smiles = []
            gts = []
            outputs = []
            
            text = batch["description"]
            smiles = batch["smiles"]
            input_ids = self.tokenizer(smiles, return_tensors="pt").input_ids.to(self.molt5.device)
            model_outputs = self.molt5_m.generate(input_ids, num_beams=5, max_length=512)
            generate_text = self.tokenizer.decode(model_outputs[0], skip_special_tokens=True)
            smiles.append(smiles[0])
            gts.append(text[0])
            outputs.append(generate_text)

            file_name = f'tmp_{self.trainer.global_rank}.txt'
            file_pt = os.path.join(self.args.temp_dir, file_name)
            with open (file_pt,"a") as f:
                for row in zip(smiles, gts, outputs):
                    f.write('\t'.join(row) + '\n')
        # torch.distributed.barrier()
        
    def configure_optimizers(self):

        params = [
            {
                "params": self.molt5.parameters(),
                "lr": self.args.molt5_lr,
            },
        ]
            
        optimizer = optim.Adam(
            params,
            weight_decay=self.args.decay
        )


        return optimizer
    
class Atomas(BaseModel):
    def __init__(self,
                 args,
                ):
        super().__init__(
            args=args,
            )

        self.molt5, self.tokenizer, model_dim = self.from_molt5()
        self.text_linear = nn.Linear(args.max_lenth, 1)
        self.smiles_linear = nn.Linear(args.max_lenth, 1)
        ## create momentum encoders  
        self.molt5_m, _, _ = self.from_molt5()
        self.text_linear_m = nn.Linear(args.max_lenth, 1)
        self.smiles_linear_m = nn.Linear(args.max_lenth, 1)
        self.model_pairs = [[self.molt5,self.molt5_m],
                            [self.text_linear,self.text_linear_m],
                            [self.smiles_linear,self.smiles_linear_m],
                           ]  
        self.copy_params()
        ## create the queue
        self.register_buffer("text_queue", torch.randn(model_dim, args.queue_size))
        self.register_buffer("smiles_queue", torch.randn(model_dim, args.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.smiles_queue = nn.functional.normalize(self.smiles_queue, dim=0)
        
        self.queue_size = args.queue_size
        self.momentum = args.momentum
        self.temp = nn.Parameter(0.07*torch.ones([])) 
        # Hierarchical Alignment
        self.text_weight_fc_0 = nn.Sequential(
                nn.Linear(model_dim, 2 * model_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * model_dim, 1))
        self.smiles_weight_fc_0 = nn.Sequential(
                nn.Linear(model_dim, 2 * model_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * model_dim, 1))
        
        self.text_weight_fc_1 = nn.Sequential(
                nn.Linear(model_dim, 2 * model_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * model_dim, 1))
        self.smiles_weight_fc_1 = nn.Sequential(
                nn.Linear(model_dim, 2 * model_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * model_dim, 1))   
        
        self.text_weight_fc_2 = nn.Sequential(
                nn.Linear(model_dim, 2 * model_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * model_dim, 1))
        self.smiles_weight_fc_2 = nn.Sequential(
                nn.Linear(model_dim, 2 * model_dim), nn.ReLU(inplace=True),
                nn.Linear(2 * model_dim, 1))  
        self.wti_loss = CrossEn()      
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.t_ctm1 = CTM(sample_ratio=0.25, embed_dim=model_dim, dim_out=model_dim, k=3) 
        self.t_block1 = TCBlock(dim=model_dim, num_heads=8)
        self.t_ctm2 = CTM(sample_ratio=0.5, embed_dim=model_dim, dim_out=model_dim, k=3)    
        self.t_block2 = TCBlock(dim=model_dim, num_heads=8)

        self.s_ctm1 = CTM(sample_ratio=0.25, embed_dim=model_dim, dim_out=model_dim, k=3)
        self.s_block1 = TCBlock(dim=model_dim, num_heads=8)
        self.s_ctm2 = CTM(sample_ratio=0.5, embed_dim=model_dim, dim_out=model_dim, k=3)
        self.s_block2 = TCBlock(dim=model_dim, num_heads=8)

        # 任务相关配置：缺失片段/片段到分子/关键词填空
        self.missing_frag_min = getattr(args, "missing_frag_min", 1)
        self.missing_frag_max = getattr(args, "missing_frag_max", 2)
        
    def _format_smiles(self, smiles):
        # 分子用 <bos>/<eos> 包裹
        return f"<bos>{smiles}<eos>"

    def _format_fragment(self, fragment):
        # 片段用 <bof>/<eof> 包裹
        return f"<bof>{fragment}<eof>"

    def _sentinel_token(self, idx):
        # T5 的 sentinel token
        return f"<extra_id_{idx}>"

    def _build_missing_fragment_task(self, description, fragments, rng):
        # 缺失片段补全任务：
        # 1) 在 fragments 中随机选出若干片段进行遮挡
        # 2) 输入侧用 <extra_id_k> 替换被遮挡片段，未遮挡片段保留
        # 3) 输出侧按 sentinel 顺序拼接被遮挡的真实片段
        if not fragments:
            input_text = f"missing fragment: fragments ; description {description}"
            target_text = ""
            return input_text, target_text

        # 至少遮挡 1 个片段，最多不超过 len-1，避免全部遮挡导致输入为空
        max_missing = min(self.missing_frag_max, max(len(fragments) - 1, 1))
        min_missing = min(self.missing_frag_min, max_missing)
        if min_missing >= max_missing:
            missing_count = max_missing
        else:
            missing_count = rng.randint(min_missing, max_missing + 1)
        missing_indices = list(rng.choice(len(fragments), size=missing_count, replace=False))
        missing_set = set(missing_indices)

        # 为每个被遮挡位置分配唯一 sentinel token
        sentinel_map = {}
        for idx, frag_idx in enumerate(sorted(missing_indices)):
            sentinel_map[frag_idx] = self._sentinel_token(idx)

        # 输入侧：遮挡位置用 sentinel 替换，其余片段加 <bof>/<eof>
        input_fragments = [
            (sentinel_map[i] if i in missing_set else self._format_fragment(frag))
            for i, frag in enumerate(fragments)
        ]

        # 输出侧：按 sentinel 顺序拼接被遮挡的真实片段
        target_segments = [
            f"{sentinel_map[i]} {self._format_fragment(fragments[i])}"
            for i in sorted(missing_indices)
        ]

        input_text = f"missing fragment: fragments {' '.join(input_fragments)}; description {description}"
        target_text = " ".join(target_segments)
        return input_text, target_text

    def _build_frag2mol_task(self, description, fragments, smiles):
        # 片段到分子生成任务：
        # 输入为片段序列 + 描述，输出为完整分子 SMILES
        input_fragments = [self._format_fragment(frag) for frag in fragments]
        input_text = f"fragments to molecule: {' '.join(input_fragments)}; description {description}"
        target_text = self._format_smiles(smiles)
        return input_text, target_text

    def _build_keyword_fill_task(self, description, keywords, rng):
        # 关键词填空任务（基于 description）：
        # 1) 用 keywords 在 description 中做大小写敏感的精确匹配
        # 2) 命中位置用 <extra_id_k> 替换，生成输入文本
        # 3) 输出侧按出现顺序拼接被替换的原始片段
        if not keywords:
            input_text = f"keyword fill: description {description}"
            target_text = ""
            return input_text, target_text

        # 预处理关键词：去空并按长度降序，便于重叠时优先最长匹配
        cleaned_keywords = [kw for kw in keywords if isinstance(kw, str) and kw]
        if not cleaned_keywords:
            input_text = f"keyword fill: description {description}"
            target_text = ""
            return input_text, target_text

        # 按比例抽取需要掩码的关键词数量（40%，下取整，至少 1）
        mask_count = max(1, int(len(cleaned_keywords) * 0.4))
        if mask_count > len(cleaned_keywords):
            mask_count = len(cleaned_keywords)
        masked_keyword_indices = rng.choice(len(cleaned_keywords), size=mask_count, replace=False)
        masked_keywords = [cleaned_keywords[i] for i in masked_keyword_indices]

        # 仅在被选中的关键词上进行匹配和掩码
        cleaned_keywords = sorted(set(masked_keywords), key=len, reverse=True)

        # 收集所有匹配区间 (start, end, keyword)
        spans = []
        for kw in cleaned_keywords:
            start = 0
            while True:
                idx = description.find(kw, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(kw), kw))
                start = idx + len(kw)

        if not spans:
            input_text = f"keyword fill: description {description}"
            target_text = ""
            return input_text, target_text

        # 重叠处理：优先最长匹配，其次靠前匹配
        spans.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
        selected = []
        occupied = []
        for span in spans:
            s_start, s_end, _ = span
            overlap = any(not (s_end <= o_start or s_start >= o_end) for o_start, o_end in occupied)
            if overlap:
                continue
            selected.append(span)
            occupied.append((s_start, s_end))

        # 按文本顺序编号 sentinel，构造输入/输出
        selected.sort(key=lambda x: x[0])
        masked_parts = []
        target_segments = []
        cursor = 0
        for idx, (s_start, s_end, kw) in enumerate(selected):
            sentinel = self._sentinel_token(idx)
            masked_parts.append(description[cursor:s_start])
            masked_parts.append(sentinel)
            target_segments.append(f"{sentinel} {description[s_start:s_end]}")
            cursor = s_end
        masked_parts.append(description[cursor:])

        masked_description = "".join(masked_parts)
        input_text = f"keyword fill: description {masked_description}"
        target_text = " ".join(target_segments)
        return input_text, target_text

    def _build_task_pairs(self, batch):
        # 基于 batch 构造三种任务的输入/输出对（分别返回）
        descriptions = batch.get("description", [])
        smiles_list = batch.get("smiles", [])
        fragments_list = batch.get("fragments", [])
        keywords_list = batch.get("keywords", [])

        missing_inputs = []
        missing_targets = []
        frag2mol_inputs = []
        frag2mol_targets = []
        keyword_inputs = []
        keyword_targets = []

        # 使随机过程在每个 step 稳定可复现
        step = int(getattr(self.trainer, "global_step", 0)) if hasattr(self, "trainer") else 0
        rng = np.random.RandomState(step)

        for i in range(len(smiles_list)):
            description = str(descriptions[i]) if i < len(descriptions) else ""
            smiles = str(smiles_list[i]) if i < len(smiles_list) else ""

            fragments = fragments_list[i] if i < len(fragments_list) else []
            if isinstance(fragments, str):
                fragments = [f for f in fragments.split() if f]
            keywords = keywords_list[i] if i < len(keywords_list) else []
            if isinstance(keywords, str):
                keywords = [k for k in keywords.split() if k]

            # 缺失片段补全
            mf_inp, mf_tgt = self._build_missing_fragment_task(description, fragments, rng)
            missing_inputs.append(mf_inp)
            missing_targets.append(mf_tgt)

            # 片段到分子生成
            f2m_inp, f2m_tgt = self._build_frag2mol_task(description, fragments, smiles)
            frag2mol_inputs.append(f2m_inp)
            frag2mol_targets.append(f2m_tgt)

            # 关键词填空
            kf_inp, kf_tgt = self._build_keyword_fill_task(description, keywords, rng)
            keyword_inputs.append(kf_inp)
            keyword_targets.append(kf_tgt)

        return (
            missing_inputs,
            missing_targets,
            frag2mol_inputs,
            frag2mol_targets,
            keyword_inputs,
            keyword_targets,
        )

    def _compute_seq2seq_loss(self, inputs, targets):
        # 计算三任务的 seq2seq 损失
        max_len = int(getattr(self.args, "max_lenth", 512))
        model_inputs = self.tokenizer(
            inputs,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.molt5.device)
        model_targets = self.tokenizer(
            targets,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.molt5.device)
        labels = model_targets.input_ids.masked_fill(
            model_targets.input_ids == self.tokenizer.pad_token_id, -100
        )
        outputs = self.molt5(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss

    def forward(self, batch, alpha):
        # 原始 Atomas 训练：文本-分子对比 + 生成 + 分层对齐
        text = batch["description"]
        smiles = batch["smiles"]
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
            
        logit_scale = self.logit_scale.exp()
    
        
        text_ids = self.tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors='pt').to(self.molt5.device) 
        smiles_ids = self.tokenizer(smiles, padding="max_length", max_length=512, truncation=True, return_tensors='pt').to(self.molt5.device) 
        
        # extract feats
        text_output = self.molt5.encoder(text_ids.input_ids, attention_mask = text_ids.attention_mask)     
        text_feats = text_output.last_hidden_state # shape = [bs, max_len, text_dim]
        smiles_output = self.molt5.encoder(smiles_ids.input_ids, attention_mask = smiles_ids.attention_mask) 
        smile_feats = smiles_output.last_hidden_state # shape = [bs, max_len, smiles_dim]
        
        ###============== text-smile CL moco ===================###
        # moco
        text_feats_moco = self.text_linear(text_feats.permute(0, 2, 1).contiguous()).squeeze(-1)    # shape = [bs, text_dim]
        smiles_feats_moco = self.smiles_linear(smile_feats.permute(0, 2, 1).contiguous()).squeeze(-1)   # shape = [bs, smiles_dim]
        ## normalized features
        text_feats_moco = text_feats_moco / text_feats_moco.norm(dim=-1, keepdim=True)
        smiles_feats_moco = smiles_feats_moco / smiles_feats_moco.norm(dim=-1, keepdim=True)  
        
        # get momentum features
        with torch.no_grad(): 
            if self.training:
                self._momentum_update()
            text_output_m = self.molt5_m.encoder(text_ids.input_ids, attention_mask = text_ids.attention_mask)
            text_feats_m = text_output_m.last_hidden_state
            text_feats_moco_m = self.text_linear_m(text_feats_m.permute(0, 2, 1).contiguous()).squeeze()
            text_feats_moco_m = text_feats_moco_m / text_feats_moco_m.norm(dim=-1, keepdim=True)
            text_feat_all = torch.cat([text_feats_moco_m.t(),self.text_queue.clone().detach()],dim=1)
            
            smiles_output_m = self.molt5_m.encoder(smiles_ids.input_ids, attention_mask = smiles_ids.attention_mask)
            smile_feats_m = smiles_output_m.last_hidden_state
            smiles_feats_moco_m = self.smiles_linear_m(smile_feats_m.permute(0, 2, 1).contiguous()).squeeze()
            smiles_feats_moco_m = smiles_feats_moco_m / smiles_feats_moco_m.norm(dim=-1, keepdim=True)
            smiles_feat_all = torch.cat([smiles_feats_moco_m.t(),self.smiles_queue.clone().detach()],dim=1)
            sim_t2s_m = text_feats_moco_m @ smiles_feat_all / self.temp  
            sim_s2t_m = smiles_feats_moco_m @ text_feat_all / self.temp 
            sim_targets = torch.zeros(sim_t2s_m.size()).to(self.molt5.device)
            sim_targets.fill_diagonal_(1)          

            sim_t2s_targets = alpha * F.softmax(sim_t2s_m, dim=1) + (1 - alpha) * sim_targets
            sim_s2t_targets = alpha * F.softmax(sim_s2t_m, dim=1) + (1 - alpha) * sim_targets  
        sim_t2s = text_feats_moco @ smiles_feat_all / self.temp
        sim_s2t = smiles_feats_moco @ text_feat_all / self.temp
        
        loss_t2s = -torch.sum(F.log_softmax(sim_t2s, dim=1)*sim_t2s_targets,dim=1).mean()
        loss_s2t = -torch.sum(F.log_softmax(sim_s2t, dim=1)*sim_s2t_targets,dim=1).mean()
        
        loss_tsc = (loss_t2s+loss_s2t)/2      
        
        if self.training:
            self._dequeue_and_enqueue(text_feats_moco_m, smiles_feats_moco_m)   
        

        if self.args.task =="genmol":
            decoder_input_ids = smiles_ids.input_ids.clone()
            decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 
        
            decoder_output = self.molt5(
                encoder_outputs=text_output,
                attention_mask=text_ids.attention_mask,
                decoder_attention_mask=smiles_ids.attention_mask,
                labels = decoder_targets,
                return_dict = True,
            )
            
        elif self.args.task =="gentext":
            decoder_input_ids = text_ids.input_ids.clone()
            decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 
        
            decoder_output = self.molt5(
                encoder_outputs=smiles_output,
                attention_mask=smiles_ids.attention_mask,
                decoder_attention_mask=text_ids.attention_mask,
                labels = decoder_targets,
                return_dict = True,
            )
        loss_lm = decoder_output.loss        
            
        
        text_feats = allgather(text_feats)
        text_masks = allgather(text_ids.attention_mask)
        
        
        smile_feats = allgather(smile_feats)
        smile_masks = allgather(smiles_ids.attention_mask)
        batch_size = text_feats.shape[0]
        torch.distributed.barrier()  # force sync  
        t_idx_token = torch.arange(text_feats.size(1))[None, :].repeat(text_feats.size(0), 1)
        t_agg_weight = text_feats.new_ones(text_feats.size(0), text_feats.size(1), 1)
        t_token_dict = {'x': text_feats,
                        'token_num': text_feats.size(1),
                        'idx_token': t_idx_token,
                        'agg_weight': t_agg_weight,
                        'mask': text_masks}
        
        s_idx_token = torch.arange(smile_feats.size(1))[None, :].repeat(smile_feats.size(0), 1)
        s_agg_weight = smile_feats.new_ones(smile_feats.size(0), smile_feats.size(1), 1)
        s_token_dict = {'x': smile_feats,
                        'token_num': smile_feats.size(1),
                        'idx_token': s_idx_token,
                        'agg_weight': s_agg_weight,
                        'mask': smile_masks}
        # level 0
        t2s_logits_0, s2t_logits_0, logits_0 = self.align_level_0(t_token_dict, s_token_dict)
        loss_wti_t2s_0 = self.wti_loss(t2s_logits_0 * logit_scale)
        loss_wti_s2t_0 = self.wti_loss(s2t_logits_0 * logit_scale)
        loss_wti_0 = (loss_wti_t2s_0 + loss_wti_s2t_0) / 2
    
        # level 1
        t_token_dict = self.t_block1(self.t_ctm1(t_token_dict))
        s_token_dict = self.s_block1(self.s_ctm1(s_token_dict))


        t2s_logits_1, s2t_logits_1, logits_1 = self.align_level_1(t_token_dict, s_token_dict)
        loss_wti_t2s_1 = self.wti_loss(t2s_logits_1 * logit_scale)
        loss_wti_s2t_1 = self.wti_loss(s2t_logits_1 * logit_scale)
        loss_wti_1 = (loss_wti_t2s_1 + loss_wti_s2t_1) / 2

        
        # level 2
        t_token_dict = self.t_block2(self.t_ctm2(t_token_dict))
        s_token_dict = self.s_block2(self.s_ctm2(s_token_dict))
        
        t2s_logits_2, s2t_logits_2, logits_2 = self.align_level_2(t_token_dict, s_token_dict)
        loss_wti_t2s_2 = self.wti_loss(t2s_logits_2 * logit_scale)
        loss_wti_s2t_2 = self.wti_loss(s2t_logits_2 * logit_scale)
        loss_wti_2 = (loss_wti_t2s_2 + loss_wti_s2t_2) / 2
        
        loss_wti = loss_wti_0 + loss_wti_1 + loss_wti_2

        # 新增三任务：缺失片段补全 / 片段到分子 / 关键词填空
        (
            missing_inputs,
            missing_targets,
            frag2mol_inputs,
            frag2mol_targets,
            keyword_inputs,
            keyword_targets,
        ) = self._build_task_pairs(batch)

        loss_missing = self._compute_seq2seq_loss(missing_inputs, missing_targets)
        loss_frag2mol = self._compute_seq2seq_loss(frag2mol_inputs, frag2mol_targets)
        loss_keyword = self._compute_seq2seq_loss(keyword_inputs, keyword_targets)
        
        return loss_tsc, loss_lm, loss_wti, loss_missing, loss_frag2mol, loss_keyword
    
    def align_level_0(self, t_token_dict, s_token_dict):
        
        return self.wti(t_token_dict["x"], t_token_dict["mask"], s_token_dict["x"], s_token_dict["mask"], level_num=0)
    
    def align_level_1(self, t_token_dict, s_token_dict):
        
        return self.wti(t_token_dict["x"], t_token_dict["mask"], s_token_dict["x"], s_token_dict["mask"], level_num=1)
    
    def align_level_2(self, t_token_dict, s_token_dict):
        
        return self.wti(t_token_dict["x"], t_token_dict["mask"], s_token_dict["x"], s_token_dict["mask"], level_num=2)
    
    def wti(self, text_features, text_mask, smiles_features, smiles_mask, level_num):

        text_weight_fc = getattr(self, f"text_weight_fc_{level_num}")
        smiles_weight_fc = getattr(self, f"smiles_weight_fc_{level_num}")
        
        # generate fusion weights
        text_weight = text_weight_fc(text_features).squeeze(2) # [bs, N_t, dim] -> [bs, N_t]
        text_weight = torch.softmax(text_weight, dim=-1)    # [bs, N_t]
        
        smiles_weight = smiles_weight_fc(smiles_features).squeeze(2)  # [bs, N_s, dim] -> [bs, N_s]
        smiles_weight = torch.softmax(smiles_weight, dim=-1)    # [bs, N_s]
        
        # normalized features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        smiles_features = smiles_features / smiles_features.norm(dim=-1, keepdim=True)
        
        # token interaction
        retrieve_logits = torch.einsum('atd,bsd->abts', [text_features, smiles_features])
        if level_num==0:
            retrieve_logits = torch.einsum('abts,at->abts', [retrieve_logits, text_mask])
            retrieve_logits = torch.einsum('abts,bs->abts', [retrieve_logits, smiles_mask])
        
        t2s_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abts -> abt
        t2s_logits = torch.einsum('abt,at->ab', [t2s_logits, text_weight])

        s2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abts -> abs
        s2t_logits = torch.einsum('abs,bs->ab', [s2t_logits, smiles_weight])

        _retrieve_logits = (t2s_logits + s2t_logits) / 2.0
        
        return _retrieve_logits, _retrieve_logits.T, retrieve_logits
    
    def get_text_smiles_feat(self, text, smiles):
        text_ids = self.tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors='pt').to(self.molt5.device) 
        smiles_ids = self.tokenizer(smiles, padding="max_length", max_length=512, truncation=True, return_tensors='pt').to(self.molt5.device) 
        
        # extract feats
        text_output = self.molt5.encoder(text_ids.input_ids, attention_mask = text_ids.attention_mask)     
        text_feat = text_output.last_hidden_state # shape = [bs, max_len, text_dim]
        smiles_output = self.molt5.encoder(smiles_ids.input_ids, attention_mask = smiles_ids.attention_mask) 
        smile_feat = smiles_output.last_hidden_state # shape = [bs, max_len, smiles_dim]
        # moco
        text_feats_moco = self.text_linear(text_feat.permute(0, 2, 1).contiguous()).squeeze(-1)    # shape = [bs, text_dim]
        smiles_feats_moco = self.smiles_linear(smile_feat.permute(0, 2, 1).contiguous()).squeeze(-1)   # shape = [bs, smiles_dim]
        ## normalized features
        text_feats_moco = text_feats_moco / text_feats_moco.norm(dim=-1, keepdim=True)
        smiles_feats_moco = smiles_feats_moco / smiles_feats_moco.norm(dim=-1, keepdim=True)  
        
        return text_feats_moco, smiles_feats_moco
    
    def get_smiles_feat(self, smiles):
        
        smiles_ids = self.tokenizer(smiles, padding="max_length", max_length=512, truncation=True, return_tensors='pt').to(self.molt5.device) 
        # extract feats
        smiles_output = self.molt5.encoder(smiles_ids.input_ids, attention_mask = smiles_ids.attention_mask) 
        smile_feat = smiles_output.last_hidden_state # shape = [bs, max_len, smiles_dim]
        # moco
        smiles_feats_moco = self.smiles_linear(smile_feat.permute(0, 2, 1).contiguous()).squeeze(-1)   # shape = [bs, smiles_dim]
        ## normalized features
        smiles_feats_moco = smiles_feats_moco / smiles_feats_moco.norm(dim=-1, keepdim=True)  
        
        return smile_feat

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient 
                
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, text_feat, smile_feat):
        # gather keys before updating queue
        text_feats = concat_all_gather(text_feat)
        smile_feats = concat_all_gather(smile_feat)

        batch_size = text_feats.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.smiles_queue[:, ptr:ptr + batch_size] = smile_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr     
    
    def from_molt5(self):
        # pretrain_dir = str(self.args.data_dir) + "/pretrained/molt5decoder-" + str(self.args.model_size)
        pretrain_dir = "/home/jxzhou/model/molt5/molt5-" + str(self.args.model_size)
        model = T5ForConditionalGeneration.from_pretrained(pretrain_dir)
        tokenizer = T5Tokenizer.from_pretrained(pretrain_dir, model_max_length=512)
        model_dim = model.shared.weight.shape[-1]
        return model, tokenizer, model_dim
    
    
    def training_step(self, batch, batch_idx):
        # 训练时同时计算原有损失与新增三任务的 seq2seq 损失
        loss_tsc, loss_lm, loss_wti, loss_missing, loss_frag2mol, loss_keyword = self(
                batch, self.args.alpha
            )
        missing_loss_wt = float(getattr(self.args, "missing_fragment_loss_wt", 1.0))
        frag2mol_loss_wt = float(getattr(self.args, "frag2mol_loss_wt", 1.0))
        keyword_loss_wt = float(getattr(self.args, "keyword_loss_wt", 1.0))
        loss_task = (
            missing_loss_wt * loss_missing
            + frag2mol_loss_wt * loss_frag2mol
            + keyword_loss_wt * loss_keyword
        )
        loss = (
            self.args.tsclosswt * loss_tsc
            + self.args.lmlosswt * loss_lm
            + self.args.wtilosswt * loss_wti
            + loss_task
        )
        
        self.log("train_loss_tol", loss, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("train_loss_tsc", loss_tsc, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("train_loss_lm", loss_lm, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("train_loss_wti", loss_wti, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("train_loss_task", loss_task, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("train_loss_missing", loss_missing, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("train_loss_frag2mol", loss_frag2mol, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("train_loss_keyword", loss_keyword, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        
        return loss
                         
    def configure_optimizers(self):
            
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.args.molt5_lr,
            weight_decay=self.args.decay
        )

        return optimizer
    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output 