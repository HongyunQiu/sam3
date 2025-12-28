from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

import matplotlib

# 使用非交互式后端，避免服务器环境没有显示设备时报错
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402


app = FastAPI(title="SAM3 Web Demo")

# 全局缓存模型和处理器，避免每次请求都重新加载
_model = None
_processor: Sam3Processor | None = None


def get_processor() -> Sam3Processor:
    global _model, _processor
    if _processor is None:
        _model = build_sam3_image_model()
        _processor = Sam3Processor(_model)
    return _processor


def render_results_to_png(image: Image.Image, outputs: dict) -> io.BytesIO:
    """
    复用 sam3 自带的可视化逻辑，将
    - masks
    - boxes
    - scores
    叠加到图片上，并以 PNG 格式输出到内存中。
    """
    # 清理旧图像，避免内存不断增长
    plt.close("all")

    # 这里会画出所有 masks + boxes + scores，但不调用 plt.show()
    plot_results(image, outputs)
    fig = plt.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """返回前端页面。"""
    index_path = Path(__file__).with_name("index.html")
    return index_path.read_text(encoding="utf-8")


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    prompt: str = Form("galaxy"),
):
    """
    接收一张图片和一个文本提示，返回叠加了
    masks + boxes + scores 的 PNG 图片。
    """
    # 读取上传的图片
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    processor = get_processor()

    # 推理
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    # output 里已经包含 masks / boxes / scores
    buf = render_results_to_png(image, output)

    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=False)


