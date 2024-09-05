import time
import os
import yaml
import fastapi
import uvicorn
import torch
from pathlib import Path

from nnunet_serve_utils import (
    get_info,
    get_gpu_memory,
    InferenceRequest,
    nnUNetPredictor,
    wraper,
)

if __name__ == "__main__":
    app = fastapi.FastAPI()

    with open("model-serve-spec.yaml") as o:
        models_specs = yaml.safe_load(o)
    alias_dict = {}
    for k in models_specs["models"]:
        model_name = models_specs["models"][k]["name"]
        alias_dict[model_name] = model_name
        if "aliases" in models_specs["models"][k]:
            for alias in models_specs["models"][k]["aliases"]:
                alias_dict[alias] = model_name
            del models_specs["models"][k]["aliases"]
    if "model_folder" not in models_specs:
        raise ValueError(
            "model_folder must be specified in model-serve-spec.yaml"
        )
    model_folder = models_specs["model_folder"]
    model_paths = [
        os.path.dirname(x) for x in Path(model_folder).rglob("fold_0")
    ]
    model_dictionary = {
        m.split(os.sep)[-2]: {
            "path": m,
            "model_information": get_info(f"{m}/dataset.json"),
        }
        for m in model_paths
    }
    model_dictionary = {
        m: {**model_dictionary[m], **models_specs["models"].get(m, None)}
        for m in model_dictionary
        if m in models_specs["models"]
    }

    @app.get("/model_info")
    def model_info():
        return model_dictionary

    @app.post("/infer")
    def infer(inference_request: InferenceRequest):
        nnunet_id = inference_request.nnunet_id
        if isinstance(nnunet_id, str):
            if nnunet_id not in alias_dict:
                return {
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata_path": None,
                    "nnunet_id": nnunet_id,
                    "status": "error",
                    "error": f"{nnunet_id} is not a valid nnunet_id",
                }
            nnunet_info = model_dictionary[alias_dict[nnunet_id]]
            nnunet_path = nnunet_info["path"]
            metadata_path = nnunet_info.get("metadata", None)
            min_mem = nnunet_info.get("min_mem", 4000)
        else:
            nnunet_path = []
            min_mem = 0
            for nn in nnunet_id:
                if nn not in alias_dict:
                    return {
                        "time_elapsed": None,
                        "gpu": None,
                        "nnunet_path": None,
                        "metadata_path": None,
                        "nnunet_id": nnunet_id,
                        "status": "error",
                        "error": f"{nnunet_id} is not a valid nnunet_id",
                    }
                nnunet_info = model_dictionary[alias_dict[nn]]
                nnunet_path.append(nnunet_info["path"])
                curr_min_mem = nnunet_info.get("min_mem", 4000)
                if curr_min_mem > min_mem:
                    min_mem = curr_min_mem
            metadata_path = nnunet_info.get("metadata", None)

        free = False
        while free is False:
            gpu_memory = get_gpu_memory()
            max_gpu_memory = max(gpu_memory)
            device_id = [
                i
                for i in range(len(gpu_memory))
                if gpu_memory[i] == max_gpu_memory
            ][0]
            if max_gpu_memory > min_mem:
                free = True

        a = time.time()

        params = inference_request.__dict__
        if "tta" in inference_request:
            mirroring = inference_request.tta
        else:
            mirroring = True

        try:
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=mirroring,
                device=torch.device("cuda", device_id),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )

            for k in ["nnunet_id", "tta", "min_mem", "aliases"]:
                if k in params:
                    del params[k]

            output_paths = wraper(
                **params,
                predictor=predictor,
                nnunet_path=nnunet_path,
                metadata_path=metadata_path,
            )
            del predictor
            error = None
            status = "success"
        except Exception as e:
            output_paths = {}
            status = "fail"
            error = str(e)
        torch.cuda.empty_cache()
        b = time.time()

        return {
            "time_elapsed": b - a,
            "gpu": device_id,
            "nnunet_path": nnunet_path,
            "metadata_path": metadata_path,
            "nnunet_id": nnunet_id,
            "status": status,
            "error": error,
            **output_paths,
        }

    uvicorn.run(app, host="0.0.0.0", port=12345)
