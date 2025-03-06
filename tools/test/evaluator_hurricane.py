from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.evaluation.video_evaluation import VideoEvaluator

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for the given dataset.
    """
    if output_folder is None:
        output_folder = cfg.OUTPUT_DIR
        
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "vis":
        return VideoEvaluator(
            dataset_name,
            cfg,
            True,  # distributed
            output_dir=output_folder,
        )
    raise NotImplementedError(
        "No evaluator implementation for evaluator_type: {}".format(evaluator_type)
    )
