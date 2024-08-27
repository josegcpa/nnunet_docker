import time
import os
import yaml
import fastapi
import uvicorn
import torch
from fastapi import HTTPException
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
        a = time.time()
        free = False
        while free is False:
            gpu_memory = get_gpu_memory()
            max_gpu_memory = max(gpu_memory)
            device_id = [
                i
                for i in range(len(gpu_memory))
                if gpu_memory[i] == max_gpu_memory
            ][0]
            if max_gpu_memory > 4000:
                free = True

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device("cuda", device_id),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        prior_use_mirroring = predictor.use_mirroring
        nnunet_id = inference_request.nnunet_id
        if isinstance(nnunet_id, str):
            if nnunet_id not in model_dictionary:
                raise HTTPException(
                    404, f"{nnunet_id} is not a valid nnUNet model"
                )
            nnunet_info = model_dictionary[nnunet_id]
            nnunet_path, metadata_path = nnunet_info["path"], nnunet_info.get(
                "metadata", None
            )
        else:
            nnunet_path = []
            for nn in nnunet_id:
                nnunet_info = model_dictionary[nn]
                nnunet_path.append(nnunet_info["path"])
            metadata_path = nnunet_info.get("metadata", None)

        params = inference_request.__dict__
        if "tta" in inference_request:
            predictor.use_mirroring = inference_request.tta
        del params["nnunet_id"]
        del params["tta"]
        output_paths = wraper(
            **params,
            predictor=predictor,
            nnunet_path=nnunet_path,
            metadata_path=metadata_path,
        )
        predictor.use_mirroring = prior_use_mirroring
        b = time.time()

        del predictor
        torch.cuda.empty_cache()

        return {
            "time_elapsed": b - a,
            "gpu": device_id,
            "nnunet_path": nnunet_path,
            "metadata_path": metadata_path,
            **output_paths,
        }

    uvicorn.run(app, host="0.0.0.0", port=12345)
