"""
End-to-end pipeline: Isaac Sim synthetic data generation → TAO training → MLflow registration.

Compile:  python pipelines/isaac_training_pipeline.py
Submit:   upload pipelines/isaac_training_pipeline.yaml via RHOAI dashboard or API
"""
from kfp import dsl, compiler
from kfp import kubernetes


# ---------------------------------------------------------------------------
# Step 1: Clone the SDG repo to a shared PVC
# ---------------------------------------------------------------------------
@dsl.container_component
def clone_sdg_repo():
    return dsl.ContainerSpec(
        image="alpine/git:latest",
        command=["/bin/sh", "-c"],
        args=[
            "if [ -d /workspace/sdg/palletjack_sdg ]; then "
            "echo 'SDG repo already cloned, skipping'; "
            "else "
            "rm -rf /workspace/sdg/lost+found && "
            "git clone --depth 1 "
            "https://github.com/NVIDIA-AI-IOT/synthetic_data_generation_training_workflow.git "
            "/workspace/sdg; "
            "fi"
        ],
    )


# ---------------------------------------------------------------------------
# Step 2: Generate synthetic data with Isaac Sim Replicator
#
# Parameters are passed as positional args and referenced via $1..$4
# in the shell script. This is the standard KFP v2 pattern for
# container_component parameter passing on DSPA — set_env_variable
# does not resolve pipeline parameter channels (KFP issue #10111).
# ---------------------------------------------------------------------------
@dsl.container_component
def generate_synthetic_data(
    num_frames: int,
    height: int,
    width: int,
    distractors: str,
):
    return dsl.ContainerSpec(
        image="nvcr.io/nvidia/isaac-sim:4.5.0",
        command=["/bin/bash"],
        args=[
            "-c",
            'echo "=== Isaac Sim Headless Datagen ===" && nvidia-smi && '
            "cd /workspace/sdg/palletjack_sdg && "
            "/isaac-sim/python.sh standalone_palletjack_sdg.py "
            '--headless True --height "$1" --width "$2" '
            '--num_frames "$3" --distractors "$4" '
            '--data_dir "/output/palletjack_data/$4" && '
            'echo "=== Datagen complete ===" && '
            'echo "RGB: $(ls /output/palletjack_data/$4/Camera/rgb/ | wc -l)" && '
            'echo "Labels: $(ls /output/palletjack_data/$4/Camera/object_detection/ | wc -l)"',
            "--",  # end of bash options
            height, width, num_frames, distractors,
        ],
    )


# ---------------------------------------------------------------------------
# Step 3: Convert KITTI annotations to TFRecords
# ---------------------------------------------------------------------------
@dsl.container_component
def convert_to_tfrecords():
    return dsl.ContainerSpec(
        image="nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5",
        command=["/bin/bash", "-c"],
        args=[
            "echo '=== Converting KITTI to TFRecords ===' && "
            "mkdir -p /workspace/tfrecords/warehouse && "
            "detectnet_v2 dataset_convert "
            "-d /workspace/specs/tfrecords_warehouse.txt "
            "-o /workspace/tfrecords/warehouse/tfrecord && "
            "echo '=== Conversion complete ===' && "
            "ls -la /workspace/tfrecords/warehouse/"
        ],
    )


# ---------------------------------------------------------------------------
# Step 4: Train DetectNet_v2
# ---------------------------------------------------------------------------
@dsl.container_component
def train_detectnet(encryption_key: str):
    return dsl.ContainerSpec(
        image="nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5",
        command=["/bin/bash"],
        args=[
            "-c",
            "echo '=== TAO DetectNet_v2 Training ===' && nvidia-smi && "
            "mkdir -p /workspace/pretrained /workspace/output/resnet18_palletjack && "
            "wget -q --content-disposition "
            "'https://api.ngc.nvidia.com/v2/models/nvidia/tao/pretrained_detectnet_v2"
            "/versions/resnet18/files/resnet18.hdf5' "
            "-O /workspace/pretrained/resnet18.hdf5 && "
            "detectnet_v2 train "
            "-e /workspace/specs/training_resnet18.txt "
            "-r /workspace/output/resnet18_palletjack "
            '-k "$1" '
            "--gpus 1 && "
            "echo '=== Training complete ===' && "
            "find /workspace/output/ -type f | head -20 && "
            "ls -la /workspace/output/resnet18_palletjack/weights/",
            "--",
            encryption_key,
        ],
    )


# ---------------------------------------------------------------------------
# Step 5: Evaluate results and register model in MLflow
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    packages_to_install=["mlflow>=3.5.0", "boto3>=1.34.0"],
)
def evaluate_and_register(
    mlflow_tracking_uri: str,
    mlflow_s3_endpoint: str,
    experiment_name: str,
    num_frames: int,
    height: int,
    width: int,
    distractors: str,
    epochs: int,
    batch_size: int,
    map_threshold: float,
) -> str:
    """Parse training status.json, log metrics to MLflow, register model if mAP > threshold."""
    import json
    import os

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = mlflow_s3_endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("MINIO_ACCESS_KEY", "minio")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("MINIO_SECRET_KEY", "minio123")
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    # RHOAI MLflow uses TLS + K8s workspace auth:
    # - Bearer token from ServiceAccount for authentication
    # - X-MLFLOW-WORKSPACE header (= pod namespace) for workspace context
    sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    sa_ns_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.exists(sa_token_path):
        with open(sa_token_path) as f:
            os.environ["MLFLOW_TRACKING_TOKEN"] = f.read().strip()
    workspace = "default"
    if os.path.exists(sa_ns_path):
        with open(sa_ns_path) as f:
            workspace = f.read().strip()

    import mlflow
    from mlflow.tracking.request_header.registry import _request_header_provider_registry

    class _WorkspaceHeaderProvider:
        """Injects X-MLFLOW-WORKSPACE header for RHOAI MLflow K8s auth."""
        def in_context(self):
            return True
        def request_headers(self):
            return {"X-MLFLOW-WORKSPACE": workspace}

    _request_header_provider_registry.register(_WorkspaceHeaderProvider)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Parse training results (TAO writes NDJSON — one JSON object per line)
    status_file = "/workspace/output/resnet18_palletjack/status.json"
    with open(status_file) as f:
        lines = [line.strip() for line in f if line.strip()]
    status = json.loads(lines[-1]) if lines else {}

    final_map = 0.0
    final_loss = 0.0
    if "results" in status:
        results = status["results"]
        if isinstance(results, list) and results:
            last = results[-1]
            final_map = last.get("mean_average_precision", 0.0)
            final_loss = last.get("loss", 0.0)
        elif isinstance(results, dict):
            final_map = results.get("mean_average_precision", 0.0)
            final_loss = results.get("loss", 0.0)

    model_path = "/workspace/output/resnet18_palletjack/weights/model.hdf5"
    model_exists = os.path.exists(model_path)

    print(f"Final mAP: {final_map}")
    print(f"Final loss: {final_loss}")
    print(f"Model exists: {model_exists}")
    print(f"mAP threshold: {map_threshold}")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "num_frames": num_frames,
            "image_height": height,
            "image_width": width,
            "distractors": distractors,
            "epochs": epochs,
            "batch_size": batch_size,
            "architecture": "resnet18",
            "detector": "detectnet_v2",
        })

        mlflow.log_metrics({
            "final_map": final_map,
            "final_loss": final_loss,
        })

        mlflow.log_artifact(status_file, artifact_path="training_output")

        spec_file = "/workspace/specs/training_resnet18.txt"
        if os.path.exists(spec_file):
            mlflow.log_artifact(spec_file, artifact_path="specs")

        if model_exists and final_map >= map_threshold:
            mlflow.log_artifact(model_path, artifact_path="model")
            client = mlflow.MlflowClient()
            try:
                client.create_registered_model("palletjack-detectnet-v2")
            except mlflow.exceptions.MlflowException:
                pass  # already exists
            client.create_model_version(
                name="palletjack-detectnet-v2",
                source=f"runs:/{run.info.run_id}/model",
                run_id=run.info.run_id,
            )
            result = f"REGISTERED (mAP={final_map:.4f} >= {map_threshold})"
        elif model_exists:
            mlflow.log_artifact(model_path, artifact_path="model")
            result = f"LOGGED_ONLY (mAP={final_map:.4f} < {map_threshold})"
        else:
            result = f"NO_MODEL (model file not found)"

        mlflow.set_tag("result", result)
        print(f"Result: {result}")
        print(f"MLflow run ID: {run.info.run_id}")

    return result


# ---------------------------------------------------------------------------
# Helpers — must be defined before pipeline function (evaluated at import)
# ---------------------------------------------------------------------------
def _configure_gpu(task):
    """Standard GPU scheduling for NVIDIA containers on OpenShift.

    runAsUser=0 is handled by the anyuid SCC granted to
    pipeline-runner-isaac-pipelines SA — DSPA 2.5.0 does not support
    kubernetes.set_security_context() (KFP issue #3866).
    """
    task.set_accelerator_type("nvidia.com/gpu")
    task.set_accelerator_limit("1")
    task.set_memory_limit("32Gi")
    task.set_cpu_limit("8")
    task.set_memory_request("16Gi")
    task.set_cpu_request("4")
    kubernetes.add_node_selector(task, "nvidia.com/gpu.product", "NVIDIA-L40S")
    kubernetes.add_toleration(
        task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule",
    )


def _configure_pvc(
    task,
    workspace=False,
    output=False,
    synthetic_data=False,
    model_artifacts=False,
    specs=False,
    tfrecords=False,
    pretrained=False,
):
    """Mount standard PVCs and volumes for pipeline tasks."""
    if workspace:
        kubernetes.mount_pvc(
            task, pvc_name="workspace-data", mount_path="/workspace/sdg",
        )
    if output:
        kubernetes.mount_pvc(
            task, pvc_name="synthetic-data", mount_path="/output",
        )
    if synthetic_data:
        kubernetes.mount_pvc(
            task, pvc_name="synthetic-data", mount_path="/workspace/data",
        )
    if model_artifacts:
        kubernetes.mount_pvc(
            task, pvc_name="model-artifacts", mount_path="/workspace/output",
        )
    if specs:
        kubernetes.use_config_map_as_volume(
            task, config_map_name="tao-specs", mount_path="/workspace/specs",
        )
    if tfrecords:
        kubernetes.mount_pvc(
            task, pvc_name="tfrecords", mount_path="/workspace/tfrecords",
        )
    if pretrained:
        kubernetes.empty_dir_mount(
            task, volume_name="pretrained", mount_path="/workspace/pretrained",
        )


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
@dsl.pipeline(
    name="isaac-synth-train-validate",
    description=(
        "Synthetic data generation with Isaac Sim Replicator, "
        "model training with TAO DetectNet_v2, "
        "and experiment tracking + model registration with MLflow."
    ),
)
def isaac_training_pipeline(
    num_frames: int = 100,
    height: int = 544,
    width: int = 960,
    distractors: str = "warehouse",
    epochs: int = 2,
    batch_size: int = 4,
    encryption_key: str = "tao_key",
    mlflow_tracking_uri: str = "https://mlflow.redhat-ods-applications.svc:8443",
    mlflow_s3_endpoint: str = "http://minio.isaac-mlops-poc.svc:9000",
    experiment_name: str = "palletjack-detectnet-v2",
    map_threshold: float = 0.0,
):
    # -- Step 1: Clone SDG repo to workspace PVC --
    clone_task = clone_sdg_repo()
    _configure_pvc(clone_task, workspace=True)

    # -- Step 2: Isaac Sim datagen (GPU) --
    datagen_task = generate_synthetic_data(
        num_frames=num_frames,
        height=height,
        width=width,
        distractors=distractors,
    ).after(clone_task)
    # Static env vars are fine with set_env_variable (literal strings only)
    datagen_task.set_env_variable("ACCEPT_EULA", "Y")
    datagen_task.set_env_variable("PRIVACY_CONSENT", "Y")
    _configure_pvc(datagen_task, workspace=True, output=True)
    _configure_gpu(datagen_task)
    kubernetes.set_image_pull_secrets(datagen_task, ["ngc-secret"])
    for name, path in [
        ("cache", "/isaac-sim/kit/cache"),
        ("glcache", "/root/.cache/nvidia/GLCache"),
        ("computecache", "/root/.nv/ComputeCache"),
        ("omniverse", "/.nvidia-omniverse"),
    ]:
        kubernetes.empty_dir_mount(datagen_task, volume_name=name, mount_path=path)

    # -- Step 3: Convert KITTI → TFRecords --
    convert_task = convert_to_tfrecords().after(datagen_task)
    _configure_pvc(convert_task, synthetic_data=True, specs=True, tfrecords=True)
    _configure_gpu(convert_task)
    kubernetes.set_image_pull_secrets(convert_task, ["ngc-secret"])

    # -- Step 4: Train DetectNet_v2 (GPU) --
    train_task = train_detectnet(
        encryption_key=encryption_key,
    ).after(convert_task)
    _configure_pvc(
        train_task,
        synthetic_data=True, model_artifacts=True, specs=True,
        tfrecords=True, pretrained=True,
    )
    _configure_gpu(train_task)
    kubernetes.set_image_pull_secrets(train_task, ["ngc-secret"])
    kubernetes.use_secret_as_env(
        train_task, secret_name="ngc-api-key",
        secret_key_to_env={"NGC_API_KEY": "NGC_API_KEY"},
    )

    # -- Step 5: Evaluate + register in MLflow (no GPU) --
    eval_task = evaluate_and_register(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_s3_endpoint=mlflow_s3_endpoint,
        experiment_name=experiment_name,
        num_frames=num_frames,
        height=height,
        width=width,
        distractors=distractors,
        epochs=epochs,
        batch_size=batch_size,
        map_threshold=map_threshold,
    ).after(train_task)
    _configure_pvc(eval_task, model_artifacts=True, specs=True)
    kubernetes.use_secret_as_env(
        eval_task, secret_name="minio-credentials",
        secret_key_to_env={
            "accesskey": "MINIO_ACCESS_KEY",
            "secretkey": "MINIO_SECRET_KEY",
        },
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        isaac_training_pipeline, "pipelines/isaac_training_pipeline.yaml",
    )
    print("Compiled to pipelines/isaac_training_pipeline.yaml")
