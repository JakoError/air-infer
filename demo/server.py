# server.py
import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import Tensor, ModelConfig
from pytriton.triton import Triton, TritonConfig


@batch
def image_to_text(IMAGE):
    # IMAGE is (B, 224, 224, 3) when @batch is used
    bsz = IMAGE.shape[0]

    # BYTES tensor: prefer np.object_ holding python bytes
    msg = b"hello-from-pytriton"
    out = np.full((bsz, 1), msg, dtype=np.object_)
    return {"TEXT_OUT": out, "IMAGE_OUT": np.ascontiguousarray(IMAGE).copy()}


def main():
    cfg = TritonConfig(
        grpc_address="127.0.0.1",
        http_address="127.0.0.1",
        metrics_address="127.0.0.1",
        grpc_port=9100,
        http_port=8100,
        metrics_port=8101,
        log_verbose=1,
    )

    # Official blocking-mode pattern: context manager + serve()
    with Triton(config=cfg) as triton:
        triton.bind(
            model_name="EchoText",
            infer_func=image_to_text,
            inputs=[Tensor(name="IMAGE", dtype=np.uint8, shape=(224, 224, 3))],
            # Official docs allow bytes dtype for BYTES tensors; shape=(1,) means per-sample one string
            outputs=[
                Tensor(name="TEXT_OUT", dtype=bytes, shape=(1,)),
                Tensor(name="IMAGE_OUT", dtype=np.uint8, shape=(224, 224, 3)),
            ],
            config=ModelConfig(max_batch_size=16),
            strict=True,
        )
        print("Serving on gRPC 127.0.0.1:9100, HTTP 127.0.0.1:8100 ...")
        triton.serve()


if __name__ == "__main__":
    main()
