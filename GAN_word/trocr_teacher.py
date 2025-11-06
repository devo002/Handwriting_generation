
# trocr_teacher.py
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

class TrocrTeacher(torch.nn.Module):
    def __init__(self, name="/home/woody/iwi5/iwi5333h/model/trocr-base-handwritten", device="cuda"):
        super().__init__()
        self.processor = TrOCRProcessor.from_pretrained(name)
        self.model = VisionEncoderDecoderModel.from_pretrained(name).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        


    @torch.no_grad()
    def predict(self, imgs_tensor):
        """
        imgs_tensor: [B,1,H,W] or [B,3,H,W] in [0,1]
        Returns: texts (list[str]), confidences (tensor[B]) in [0,1]
        """
        # ensure 3 channels for TrOCR
        if imgs_tensor.shape[1] == 1:
            imgs_tensor = imgs_tensor.repeat(1, 3, 1, 1)

        imgs = (imgs_tensor * 255).clamp(0, 255).to(torch.uint8).cpu()
        pil_list = [torch.permute(img, (1, 2, 0)).numpy() for img in imgs]  # HWC numpy arrays

        inputs = self.processor(images=pil_list, return_tensors="pt").to(self.device)

        # --- Minimal banlist: space, period, comma ---
        tok = self.processor.tokenizer
        bad_words_ids = []
        for t in [" ", ".", ","]:
            ids = tok.encode(t, add_special_tokens=False)
            if ids:
                bad_words_ids.append(ids)

        gen = self.model.generate(
            **inputs,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=64,
            num_beams=1,                 # keep greedy unless you also want beams
            bad_words_ids=bad_words_ids  # <--- ban " ", ".", ","
        )

        texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)

        # Confidence = average max prob across tokens
        probs = []
        for step_logits in gen.scores:
            step_prob = step_logits.softmax(dim=-1).max(dim=-1).values
            probs.append(step_prob)
        conf = torch.stack(probs, dim=0).mean(dim=0) if probs else torch.zeros(len(texts))
        return texts, conf.to(self.device)
