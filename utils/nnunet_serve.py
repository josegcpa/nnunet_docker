import time
import os
import yaml
import fastapi
import uvicorn
import torch
from pathlib import Path

from nnunet_serve_utils import (
    FAILURE_STATUS,
    SUCCESS_STATUS,
    get_info,
    get_default_params,
    get_series_paths,
    wait_for_gpu,
    InferenceRequest,
    predict,
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
        m: {
            **model_dictionary[m],
            **models_specs["models"].get(m, None),
            "default_args": models_specs["models"][m].get("default_args", {}),
        }
        for m in model_dictionary
        if m in models_specs["models"]
    }

    @app.get("/model_info")
    def model_info():
        return model_dictionary

    @app.post("/infer")
    def infer(inference_request: InferenceRequest):
        params = inference_request.__dict__
        nnunet_id = params["nnunet_id"]
        # check if nnunet_id is a list
        if isinstance(nnunet_id, str):
            if nnunet_id not in alias_dict:
                return {
                    "time_elapsed": None,
                    "gpu": None,
                    "nnunet_path": None,
                    "metadata_path": None,
                    "request": inference_request.__dict__,
                    "status": FAILURE_STATUS,
                    "error": f"{nnunet_id} is not a valid nnunet_id",
                }
            nnunet_info = model_dictionary[alias_dict[nnunet_id]]
            nnunet_path = nnunet_info["path"]
            min_mem = nnunet_info.get("min_mem", 4000)
            default_args = nnunet_info.get("default_args", {})
        else:
            nnunet_path = []
            default_args = []
            min_mem = 0
            for nn in nnunet_id:
                if nn not in alias_dict:
                    return {
                        "time_elapsed": None,
                        "gpu": None,
                        "nnunet_path": None,
                        "metadata_path": None,
                        "request": inference_request.__dict__,
                        "status": FAILURE_STATUS,
                        "error": f"{nnunet_id} is not a valid nnunet_id",
                    }
                nnunet_info = model_dictionary[alias_dict[nn]]
                nnunet_path.append(nnunet_info["path"])
                curr_min_mem = nnunet_info.get("min_mem", 4000)
                if curr_min_mem > min_mem:
                    min_mem = curr_min_mem
                default_args.append(nnunet_info.get("default_args", {}))
        metadata_path = nnunet_info.get("metadata", None)
        default_params = get_default_params(default_args)

        print(default_params)

        # assign
        for k in default_params:
            if k not in params:
                params[k] = default_params[k]
            else:
                if params[k] is None:
                    params[k] = default_params[k]

        series_paths, code, error_msg = get_series_paths(
            params["study_path"],
            series_folders=params["series_folders"],
            n=len(nnunet_id) if isinstance(nnunet_id, list) else None,
        )

        if code == FAILURE_STATUS:
            return {
                "time_elapsed": None,
                "gpu": None,
                "nnunet_path": None,
                "metadata_path": None,
                "status": FAILURE_STATUS,
                "request": inference_request.__dict__,
                "error": error_msg,
            }

        device_id = wait_for_gpu(min_mem)

        if "tta" in params:
            mirroring = params["tta"]
        else:
            mirroring = True

        a = time.time()
        if os.environ.get("DEBUG", 0) == "1":
            output_paths = predict(
                series_paths=series_paths,
                metadata_path=metadata_path,
                mirroring=mirroring,
                device_id=device_id,
                params=params,
                nnunet_path=nnunet_path,
            )
            error = None
            status = SUCCESS_STATUS
        else:
            try:
                output_paths = predict(
                    series_paths=series_paths,
                    metadata_path=metadata_path,
                    mirroring=mirroring,
                    device_id=device_id,
                    params=params,
                    nnunet_path=nnunet_path,
                )
                error = None
                status = SUCCESS_STATUS
                torch.cuda.empty_cache()
                b = time.time()

            except Exception as e:
                output_paths = {}
                status = FAILURE_STATUS
                error = str(e)
        torch.cuda.empty_cache()
        b = time.time()

        return {
            "time_elapsed": b - a,
            "gpu": device_id,
            "nnunet_path": nnunet_path,
            "metadata_path": metadata_path,
            "request": inference_request.__dict__,
            "status": status,
            "error": error,
            **output_paths,
        }

    uvicorn.run(app, host="0.0.0.0", port=12345)
