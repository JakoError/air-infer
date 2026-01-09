# client.py
import numpy as np
from pytriton.client import ModelClient


def main():
    client = ModelClient("grpc://127.0.0.1:9100", "EchoText", lazy_init=False)
    client.wait_for_model(timeout_s=10)

    img = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    resp = client.infer_sample(IMAGE=img)

    # 1) decode TEXT_OUT
    text_out = resp["TEXT_OUT"]
    # usually shape (1,) or (1,1), element is bytes/np.object_
    text_bytes = np.squeeze(text_out).item()
    text_str = text_bytes.decode("utf-8", errors="replace")
    print("TEXT_OUT:", text_str)

    # 2) compare IMAGE_OUT
    img_out = resp["IMAGE_OUT"]
    equal = np.array_equal(img, img_out)
    print("IMAGE_OUT:", img_out.shape, img_out.dtype, "equal:", equal)

    # 3) diagnostics if mismatch
    if not equal:
        # cast to int to avoid uint8 wrap-around in subtraction
        diff = img.astype(np.int16) - img_out.astype(np.int16)
        print("diff: min =", diff.min(), "max =", diff.max(), "nonzero =", np.count_nonzero(diff))

    client.close()


if __name__ == "__main__":
    main()
