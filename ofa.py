import torch
import numpy as np
import sys
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
from torchvision import transforms
import cv2
import json
import os
import pickle
import requests
from tqdm import tqdm
import os
import urllib.request


class OFA_PhraseGrounding:
    def __init__(self, file_path = "checkpoints/refcocog.pt", url = "https://ofa-silicon.oss-us-west-1.aliyuncs.com/checkpoints/refcocog_large_best.pt"):
        os.makedirs("checkpoints/", exist_ok=True)     
        if not os.path.exists(file_path): 
            urllib.request.urlretrieve(url, file_path)
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]
        tasks.register_task('refcoco', RefcocoTask)
        self.use_cuda = torch.cuda.is_available()
        self.use_fp16 = True
        overrides={"bpe_dir":"ext/OFA/utils/BPE"}
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cp = os.path.join(file_path)
        self.models, cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
                utils.split_paths(cp),
                arg_overrides=overrides
            )
        cfg.common.seed = 7
        cfg.generation.beam = 5
        cfg.generation.min_len = 4
        cfg.generation.max_len_a = 0
        cfg.generation.max_len_b = 4
        cfg.generation.no_repeat_ngram_size = 3

        # Fix seed for stochastic decoding
        if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
            np.random.seed(cfg.common.seed)
            utils.set_torch_seed(cfg.common.seed)

        # Move models to GPU
        for model in self.models:
            model.eval()
            if self.use_fp16:
                model.half()
            if self.use_cuda and not cfg.distributed_training.pipeline_model_parallel:
                model.cuda()
            model.prepare_for_inference_(cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, cfg.generation)

        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        # Text preprocess
        self.bos_item = torch.LongTensor([self.task.src_dict.bos()])
        self.eos_item = torch.LongTensor([self.task.src_dict.eos()])
        self.pad_idx = self.task.src_dict.pad()
        self.patch_image_size = cfg.task.patch_image_size
    
    def encode_text(self, text, length=None, append_bos=False, append_eos=False):
        s = self.task.tgt_dict.encode_line(
            line=self.task.bpe.encode(text.lower()),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([self.bos_item, s])
        if append_eos:
            s = torch.cat([s, self.eos_item])
        return s

    def construct_sample(self, image: Image, text: str):
        w, h = image.size
        w_resize_ratio = torch.tensor(self.patch_image_size / w).unsqueeze(0)
        h_resize_ratio = torch.tensor(self.patch_image_size / h).unsqueeze(0)
        patch_image = self.patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = self.encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(self.pad_idx).long().sum() for s in src_text])
        sample = {
            "id":np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "w_resize_ratios": w_resize_ratio,
            "h_resize_ratios": h_resize_ratio,
            "region_coords": torch.randn(1, 4)
        }
        return sample

    # Function to turn FP32 to FP16
    def apply_half(self, t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    @torch.no_grad()
    def get_output(self, image, text):
        sample = self.construct_sample(image, text)
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample
        sample = utils.apply_to_sample(self.apply_half, sample) if self.use_fp16 else sample
        result, scores = eval_step(self.task, self.generator, self.models, sample)
        return result


if __name__ == "__main__":
    def _get_annotation(ref_path, task):
        ref_file = os.path.join(ref_path, task, 'refs(unc).p')
        data = {}
        data['dataset'] = task
        with open(ref_file, 'rb') as f:
            refs = pickle.load(f)
        data['refs'] = refs
        path = os.path.join(ref_path, task, 'instances.json')
        with open(path, 'r') as f:
            instances = json.load(f)
        data['images'] = instances['images']
        data['annotations'] = instances['annotations']
        data['categories'] = instances['categories']
        return data

    def _download_image(url, save_path):
        if not os.path.exists(save_path):
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
            else:
                return None
        image = cv2.imread(save_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _open_cache():
        os.makedirs('.cache/', exist_ok=True)

    def _get_bbox(ann_id, annotations):
        for annotation in annotations:
            if annotation['id'] == ann_id:
                return [annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0] + annotation['bbox'][2], annotation['bbox'][1] + annotation['bbox'][3]], annotation['category_id']  
    def _calculate_iou(bbox1, bbox2):
        x1_tl, y1_tl, x1_br, y1_br = bbox1
        x2_tl, y2_tl, x2_br, y2_br = bbox2
        x_intersection_tl = max(x1_tl, x2_tl)
        y_intersection_tl = max(y1_tl, y2_tl)
        x_intersection_br = min(x1_br, x2_br)
        y_intersection_br = min(y1_br, y2_br)
        intersection_width = max(0, x_intersection_br - x_intersection_tl)
        intersection_height = max(0, y_intersection_br - y_intersection_tl)
        intersection_area = intersection_width * intersection_height
        bbox1_area = (x1_br - x1_tl) * (y1_br - y1_tl)
        bbox2_area = (x2_br - x2_tl) * (y2_br - y2_tl)
        union_area = bbox1_area + bbox2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    model = OFA_PhraseGrounding()
    dataset = "refcoco"
    ref_path = "/home/menteebot/perception/ref/"
    data = _get_annotation(ref_path, dataset)
    hit = []
    pbar = tqdm(data['refs'])
    for ix, item in enumerate(pbar):
        if ix > 500:
            break
        with torch.no_grad():
            os.makedirs('.cache/' + item['file_name'].split('_')[1], exist_ok=True)
            name = 'COCO_train2014_' + item['file_name'].split('_')[2] + '.jpg'
            curr_url = os.path.join('http://images.cocodataset.org', item['file_name'].split('_')[1], name)
            save_path = os.path.join('.cache', item['file_name'].split('_')[1], name)
            _download_image(curr_url, save_path)
            image = Image.open(save_path)
            gt = _get_bbox(item['ann_id'], data['annotations'])
            # for sentence in item['sentences']:
            text = item['sentences'][0]['sent']
            result = model.get_output(image, text)
            pred = [int(result[0]["box"][0]), int(result[0]["box"][1]), int(result[0]["box"][2]), int(result[0]["box"][3])]
            if _calculate_iou(pred, gt[0]) > 0.5:
                hit.append(1)
            else:
                hit.append(0)
            pbar.set_description('acc: %.3f' % np.mean(hit))