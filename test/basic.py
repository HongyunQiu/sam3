import torch
#################################### For Image ####################################
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def visualize_results(image, masks, boxes, scores, max_items: int = 5):
    """
    简单可视化：
    - 原图
    - 带 mask + bbox + score 的叠加图
    只显示前 max_items 个结果，避免太乱。
    """
    # 转成 numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.array(image)

    # 有些输出可能是 torch.Tensor，这里统一转成 numpy
    if isinstance(masks, torch.Tensor):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.array(masks)

    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.array(boxes)

    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.array(scores)

    num_items = min(max_items, len(masks_np))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 子图1：原图
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 子图2：叠加 mask 和 bbox
    axes[1].imshow(image_np)

    for i in range(num_items):
        mask = masks_np[i]  # 形状通常是 (H, W) 或 (1, H, W)
        if mask.ndim == 3:
            mask = mask[0]

        # 用一个半透明的颜色显示 mask
        colored_mask = np.zeros((*mask.shape, 4))
        colored_mask[..., 0] = 1.0  # 红色通道
        colored_mask[..., 3] = 0.4 * (mask > 0.5)  # 透明度，根据阈值二值化
        axes[1].imshow(colored_mask)

        # 画 bbox（xyxy 或 xywh，这里假定为 xyxy）
        box = boxes_np[i]
        if box.shape[0] == 4:
            x1, y1, x2, y2 = box
        else:
            # 如果是 (x, y, w, h)
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h

        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="lime",
            linewidth=2,
        )
        axes[1].add_patch(rect)

        # 在左上角写 score
        score_text = f"{scores_np[i]:.2f}" if np.ndim(scores_np[i]) == 0 else f"{float(scores_np[i]):.2f}"
        axes[1].text(
            x1,
            y1 - 2,
            score_text,
            color="yellow",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    axes[1].set_title(f"Top {num_items} Masks & Boxes & Scores")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 加载模型
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # 加载图片（假设图片在当前 test 目录）
    image = Image.open("20240613234224.jpg")

    # 模型推理
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt="galaxy")

    # 获取 masks, boxes, scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

    # 可视化
    visualize_results(image, masks, boxes, scores, max_items=5)