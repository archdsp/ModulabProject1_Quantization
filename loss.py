import torch
import torch.nn.functional as F


def detection_loss(pred, target, num_classes=20):
    """
    pred, target: [B, S, S, 4+C]
    """
    B, S, _, _ = pred.shape

    bbox_pred = pred[..., :4].contiguous().reshape(-1, 4)
    bbox_true = target[..., :4].contiguous().reshape(-1, 4)

    cls_pred = pred[..., 4:].contiguous().reshape(-1, num_classes)
    cls_true = target[..., 4:].contiguous().reshape(-1, num_classes)
    
    # bbox regression loss
    loss_bbox = F.smooth_l1_loss(bbox_pred, bbox_true)

    # classification loss
    loss_cls = F.binary_cross_entropy_with_logits(cls_pred, cls_true)

    return loss_bbox + loss_cls


if __name__ == "__main__":
    pred = torch.randn(8, 7, 7, 24)
    target = torch.randn(8, 7, 7, 24)
    loss = detection_loss(pred, target)
    print(loss)
