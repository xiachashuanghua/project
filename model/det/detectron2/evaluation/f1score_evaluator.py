import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict, defaultdict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
import pandas as pd

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from pycocotools.cocoeval import COCOeval
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from.coco_evaluation import instances_to_coco_json
from.evaluator import DatasetEvaluator


class F1ScoreEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None):
        self._tasks = ('segm',)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.info(
                f"'{dataset_name}' is not registered by `register_coco_instances`. Therefore trying to convert it to COCO format..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        self._results = OrderedDict()
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        return copy.deepcopy(self._results)

    def _eval_predictions(self, tasks, predictions):
        self._logger.info("Preparing results for COCO format...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v - 1: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id - 1 in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id - 1]

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=None
                )
                if len(coco_results) > 0
                else None
            )
            res = self._derive_coco_results(coco_eval, task, class_names=self._metadata.get("thing_classes"))
            self._results['f1-' + task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = ["F1-all", "F1-0.5", "F1-0.8", "F1-small", "F1-medium", "F1-large"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # 计算总体 F1 分数
        def calculate_overall_f1():
            overall_prc_dict = defaultdict(list)
            overall_rec_dict = defaultdict(list)
            for evalImg in coco_eval.evalImgs:
                if evalImg is None:
                    continue
                if evalImg['maxDet']!= coco_eval.params.maxDets[-1]:
                    continue
                cate_class, area, dt, dtIg, gtIds, gtIg = [evalImg[x] for x in ['category_id', 'aRng', 'dtMatches', 'dtIgnore', 'gtIds', 'gtIgnore']]
                area = tuple(area)
                thrs = coco_eval.params.iouThrs
                num_gt = len(gtIds) - np.sum(gtIg)
                T, D = dt.shape
                if num_gt == 0:
                    continue
                tp = np.logical_and(dt, np.logical_not(dtIg)).sum(1) * 1.0
                fp = np.logical_and(np.logical_not(dt), np.logical_not(dtIg)).sum(1) * 1.0
                fn = np.logical_and(np.logical_not(evalImg['gtMatches']), np.logical_not(gtIg)[np.newaxis, :]).sum(1) * 1.0
                for i in range(T):
                    overall_prc_dict[(cate_class, i, area)].append(tp[i] / (tp[i] + fp[i] + np.spacing(1)))
                    overall_rec_dict[(cate_class, i, area)].append(tp[i] / (num_gt + np.spacing(1)))

            overall_f1_scores = defaultdict(list)
            area_to_key = [(tuple(area), st) for area, st in zip(coco_eval.params.areaRng, ['all', 'small', 'medium', 'large'])]
            area_to_key = dict(area_to_key)
            for key in overall_prc_dict.keys():
                cate_cls, iou_thr, area = key
                areakey = area_to_key[area]
                pr = np.array(overall_prc_dict[key])
                rc = np.array(overall_rec_dict[key])
                assert len(pr) == len(rc)
                f1 = 2 * pr * rc / (pr + rc + np.spacing(1))
                if len(f1) == 0:
                    continue
                overall_f1_scores[(iou_thr, areakey)].append(f1.mean())

            overall_results = {
                'all': [],
                '0.5': [],
                '0.8': [],
                'small': [],
                'medium': [],
                'large': []
            }
            for key, val in overall_f1_scores.items():
                overall_results[key[1]].append(val)
                if key[1] == 'all':
                    iou_thr = coco_eval.params.iouThrs[key[0]]
                    if str(iou_thr) in ['0.5', '0.8']:
                        overall_results[str(iou_thr)].append(val)

            for k, v in overall_results.items():
                overall_results[k] = np.around(np.mean(v), 4)

            return overall_results

        # 计算每个类别的 F1 分数
        def calculate_per_category_f1(area_to_key):
            per_category_results = {}
            for category_id in set([evalImg['category_id'] for evalImg in coco_eval.evalImgs if evalImg is not None]):
                category_name = None
                if category_id - 1 < len(class_names):
                    category_name = class_names[category_id - 1]
                else:
                    self._logger.warning(f"Invalid category_id: {category_id}")
                    continue
                category_prc_dict = defaultdict(list)
                category_rec_dict = defaultdict(list)
                for evalImg in coco_eval.evalImgs:
                    if evalImg is None or evalImg['category_id']!= category_id:
                        continue
                    cate_class, area, dt, dtIg, gtIds, gtIg = [evalImg[x] for x in ['category_id', 'aRng', 'dtMatches', 'dtIgnore', 'gtIds', 'gtIgnore']]
                    area = tuple(area)
                    thrs = coco_eval.params.iouThrs
                    num_gt = len(gtIds) - np.sum(gtIg)
                    T, D = dt.shape
                    if num_gt == 0:
                        continue
                    tp = np.logical_and(dt, np.logical_not(dtIg)).sum(1) * 1.0
                    fp = np.logical_and(np.logical_not(dt), np.logical_not(dtIg)).sum(1) * 1.0
                    fn = np.logical_and(np.logical_not(evalImg['gtMatches']), np.logical_not(gtIg)[np.newaxis, :]).sum(1) * 1.0
                    for i in range(T):
                        category_prc_dict[(cate_class, i, area)].append(tp[i] / (tp[i] + fp[i] + np.spacing(1)))
                        category_rec_dict[(cate_class, i, area)].append(tp[i] / (num_gt + np.spacing(1)))
                category_f1_scores = defaultdict(list)
                for key in category_prc_dict.keys():
                    cate_cls, iou_thr, area = key
                    areakey = area_to_key[area]
                    pr = np.array(category_prc_dict[key])
                    rc = np.array(category_rec_dict[key])
                    assert len(pr) == len(rc)
                    f1 = 2 * pr * rc / (pr + rc + np.spacing(1))
                    if len(f1) == 0:
                        continue
                    category_f1_scores[(iou_thr, areakey)].append(f1.mean())
                category_results = {
                    'all': [],
                    '0.5': [],
                    '0.8': [],
                    'small': [],
                    'medium': [],
                    'large': []
                }
                for key, f1_mean in category_f1_scores.items():
                    iou_thr, areakey = key
                    category_results[areakey].append(f1_mean)
                    if areakey == 'all':
                        iou_thr = coco_eval.params.iouThrs[iou_thr]
                        if str(iou_thr) in ['0.5', '0.8']:
                            category_results[str(iou_thr)].append(f1_mean)
                for k, v in category_results.items():
                    category_results[k] = np.around(np.mean(v), 4)
                per_category_results[category_name] = category_results

            return per_category_results

        overall_f1_results = calculate_overall_f1()
        area_to_key = [(tuple(area), st) for area, st in zip(coco_eval.params.areaRng, ['all', 'small', 'medium', 'large'])]
        area_to_key = dict(area_to_key)
        per_category_f1_results = calculate_per_category_f1(area_to_key)

        # Final dict of results
        results_with_per_category = {f'F1_{k}': v for k, v in overall_f1_results.items()}
        for category_name, category_result in per_category_f1_results.items():
            for metric_key, metric_value in category_result.items():
                results_with_per_category[f'F1_{category_name}_{metric_key}'] = metric_value

        # Check if all values are numerical before summing
        all_numeric = all(isinstance(value, (int, float)) for value in results_with_per_category.values())
        if not all_numeric:
            self._logger.error("Some values in results_with_per_category are not numerical.")
            return {metric: float("nan") for metric in metrics}

        # 使用 pandas 创建表格
        df = pd.DataFrame.from_dict(results_with_per_category, orient='index', columns=['Value'])
        formatted_table = df.to_string(index=True, justify='left', formatters={col: lambda x: f'{x:.4f}' if isinstance(x, float) else x for col in df.columns})

        self._logger.info(f"Evaluation results for F1-score:\n{formatted_table}")
        if not np.isfinite(sum(results_with_per_category.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")
        return results_with_per_category



def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        for c in coco_results:
            c.pop("bbox", None)
    else:
        raise ValueError(f"iou_type {iou_type} not supported")

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()

    return coco_eval