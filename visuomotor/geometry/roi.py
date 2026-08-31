"""Image ROI / box / crop geometry.

Canonical box convention: continuous, exclusive-edge ``[x1, y1, x2, y2]``
pixel coordinates, with ``full image = [0, 0, W, H]``. Two guards are kept
explicit rather than bundled into one ambiguous default:

- a pixel margin (:func:`expand_box_px`) -- anti-shrink safety padding around
  a tight box before it is used as a crop or consistency-loss target;
- a grid-cell guard (:func:`box_px_to_grid_mask`'s ``guard_cells``) -- extra
  grid cells included when mapping a pixel box onto a coarser attention grid.

Both are intentional stabilization against Seeker's iterative refinement
progressively shrinking its predicted box; neither replaces the other.
"""

from __future__ import annotations

from typing import Optional

import torch
from torchvision.ops import roi_align


def expand_box_px(
    box_px: torch.Tensor, margin_px: float, *, image_size: Optional[float] = None
) -> torch.Tensor:
    """Symmetrically expand ``[x1,y1,x2,y2]`` pixel boxes by ``margin_px``.

    Clamps to ``[0, image_size]`` when ``image_size`` is given.
    """
    x1, y1, x2, y2 = box_px.unbind(dim=-1)
    x1, y1 = x1 - margin_px, y1 - margin_px
    x2, y2 = x2 + margin_px, y2 + margin_px
    if image_size is not None:
        x1, x2 = x1.clamp(0, image_size), x2.clamp(0, image_size)
        y1, y2 = y1.clamp(0, image_size), y2.clamp(0, image_size)
    return torch.stack((x1, y1, x2, y2), dim=-1)


def normalize_box(box, size):
    x1, y1, x2, y2 = box.unbind(dim=-1)  # each is [B, T]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    scale = x2 - x1
    normed_box_px = torch.stack([cx, cy, scale], dim=-1) / size  # [B, T, 3]
    normed_box_px = 2 * normed_box_px - 1.0  # to [-1, 1]
    return normed_box_px


def grid_box_to_mask(box_g, S):
    """
    Convert integer grid box (gx1,gy1,gx2,gy2) on an S×S map to binary mask.
    x2,y2 are EXCLUSIVE and clamped to [0,S].
    Args:
        box_g: [B,Nq,4] or [B,4] in grid coords
        S: int
    Returns:
        mask: [B,Nq,S*S] boolean
    """
    assert isinstance(S, int) and S > 0

    if box_g.dim() == 2:
        box_g = box_g.unsqueeze(1)

    B, Nq, _ = box_g.shape
    device = box_g.device

    # clamp for safety
    box_g = box_g.clone()
    box_g[..., 0::2] = box_g[..., 0::2].clamp(0, S)
    box_g[..., 1::2] = box_g[..., 1::2].clamp(0, S)

    xs = torch.arange(S, device=device).view(1, 1, 1, S)
    ys = torch.arange(S, device=device).view(1, 1, S, 1)

    x1, y1, x2, y2 = [box_g[..., i].view(B, Nq, 1, 1) for i in range(4)]

    mask = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)  # [B,Nq,S,S]
    return mask.view(B, Nq, S * S).bool()  # [B,Nq,S*S]


def box_px_to_grid_mask(box_px, image_size, grid_size, guard_cells: int = 1):
    """
    Map pixel box (x1,y1,x2,y2) -> integer grid box (gx1,gy1,gx2,gy2) on a
    ``grid_size`` x ``grid_size`` map.
    - Square grid enforced via grid_size.
    - ``guard_cells`` extra grid cells of pixel margin are added before
      conversion, so the mask fully contains the pixel box plus a safety
      margin (0 disables the guard).
    - Uses floor for starts and ceil for ends so the grid box fully contains the pixel box.
    - x2,y2 are EXCLUSIVE and clamped to [0,grid_size].
    Args:
        box_px: [B, Nq, 4] or [B, 4] in pixels, cloned inside to avoid in-place modification
        image_size: int
        grid_size: int (square side)
        guard_cells: int, extra grid-cell margin applied in pixel space before conversion
    Returns:
        boxes_g: [B, Nq, 4]
    """
    assert isinstance(grid_size, int) and grid_size > 0, "grid_size must be positive int (square side)"
    box_px = box_px.clone()  # avoid in-place modification
    if box_px.dim() == 2:  # [B,4] -> [B,1,4]
        box_px = box_px.unsqueeze(1)

    if guard_cells:
        pad_px = float(guard_cells) * float(image_size) / float(grid_size)
        box_px = expand_box_px(box_px, pad_px, image_size=image_size)
    # clamp to image bounds to be safe
    x1, y1, x2, y2 = [box_px[..., i].to(torch.float32) for i in range(4)]
    x1 = x1.clamp(0, float(image_size))
    x2 = x2.clamp(0, float(image_size))
    y1 = y1.clamp(0, float(image_size))
    y2 = y2.clamp(0, float(image_size))

    # normalize -> scale -> containment rounding
    nx1, ny1 = x1 / float(image_size), y1 / float(image_size)
    nx2, ny2 = x2 / float(image_size), y2 / float(image_size)

    gx1 = torch.floor(nx1 * grid_size)
    gy1 = torch.floor(ny1 * grid_size)
    gx2 = torch.ceil(nx2 * grid_size)
    gy2 = torch.ceil(ny2 * grid_size)

    # --- enforce square box ---
    box_W = gx2 - gx1
    box_H = gy2 - gy1
    max_side = torch.maximum(box_W, box_H)
    gx2 = gx1 + max_side
    gy2 = gy1 + max_side
    gx2 = gx2.clamp(0, grid_size)
    gy2 = gy2.clamp(0, grid_size)
    gx2 = torch.maximum(gx2, gx1 + 1)
    gy2 = torch.maximum(gy2, gy1 + 1)
    box_low_res_coord = torch.stack([gx1, gy1, gx2, gy2], dim=-1).to(torch.long)
    # convert
    B, Nq, _ = box_low_res_coord.shape
    device = box_low_res_coord.device
    dtype = box_low_res_coord.dtype
    xs = torch.arange(grid_size, device=device, dtype=dtype).view(1, 1, 1, grid_size)
    ys = torch.arange(grid_size, device=device, dtype=dtype).view(1, 1, grid_size, 1)
    x1, y1, x2, y2 = [box_low_res_coord[..., i].view(B, Nq, 1, 1) for i in range(4)]
    inside_x = (xs >= x1) & (xs < x2)
    inside_y = (ys >= y1) & (ys < y2)
    mask = (inside_x & inside_y).float()  # [B,Nq,S,S]
    return mask.view(B, Nq, -1)  # [B,Nq,S*S]


@torch.no_grad()
def grid_mask_to_pixel_box(mask, prev_box_px):
    box = get_square_box(mask)
    return project_inner_box_to_original_px(
        box, prev_box_px, num_patches=mask.shape[-1]
    )


def crop_with_box(
    box,
    image,
    mask=None,
    output_size=(76, 76),
    margin=0,
    box_jitter=0.0,
    center_jitter=0.0,
    scale_jitter=0.0,
    clamp_box=False,
):
    assert margin >= 0, "margin must be nonnegative"
    assert box_jitter >= 0, "box_jitter must be nonnegative"
    assert center_jitter >= 0, "center_jitter must be nonnegative"
    assert scale_jitter >= 0, "scale_jitter must be nonnegative"

    box_px = expand_box_px(box, margin) if margin else box
    x1, y1, x2, y2 = box_px.unbind(dim=-1)

    if box_jitter > 0:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = x2 - x1
        h = y2 - y1

        B = box.shape[0]
        scale = 1 + (torch.rand(B, device=box.device) * 2 - 1) * box_jitter
        w = w * scale
        h = h * scale

        jitter_x = (torch.rand(B, device=box.device) * 2 - 1) * (box_jitter * w)
        jitter_y = (torch.rand(B, device=box.device) * 2 - 1) * (box_jitter * h)
        cx = cx + jitter_x
        cy = cy + jitter_y

        x1 = cx - 0.5 * w
        x2 = cx + 0.5 * w
        y1 = cy - 0.5 * h
        y2 = cy + 0.5 * h

    if scale_jitter > 0:
        w = x2 - x1
        h = y2 - y1
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        scale = 1.0 + (torch.rand_like(w) * 2.0 - 1.0) * scale_jitter
        scale = scale.clamp_min(1e-3)
        w = w * scale
        h = h * scale
        x1 = cx - 0.5 * w
        x2 = cx + 0.5 * w
        y1 = cy - 0.5 * h
        y2 = cy + 0.5 * h

    if center_jitter > 0:
        w = x2 - x1
        h = y2 - y1
        dx = (torch.rand_like(w) * 2.0 - 1.0) * (center_jitter * w)
        dy = (torch.rand_like(h) * 2.0 - 1.0) * (center_jitter * h)
        x1 = x1 + dx
        x2 = x2 + dx
        y1 = y1 + dy
        y2 = y2 + dy

    box_px = torch.stack([x1, y1, x2, y2], dim=-1)

    # Crop image
    crop_img = crop_and_resize(image, box_px, output_size=output_size, clamp=clamp_box)

    # Crop mask with the SAME jittered box
    crop_mask = None
    if mask is not None:
        assert mask.shape[-2:] == image.shape[-2:], "mask and image must have same H,W"
        crop_mask = crop_and_resize(
            mask.float(),
            box_px,
            output_size=output_size,
            clamp=clamp_box,
        )
    return crop_img, crop_mask, box_px


def get_square_box(
    mask: torch.Tensor,
    clamp: bool = True,
    mode: str = "xyxy_exclusive",
    thresh: float = 0.0,  # binarize mask > thresh for "bbox" mode
    max_side: int = None,  # cap on side length; default S if None
):
    """
    Square crop from attention mask with less conservative sizing.

    mask: (B,S,S), float/bool. Weights >=0.
    Returns: (B,4) box (x1,y1,x2,y2) where x2/y2 exclusive if mode='xyxy_exclusive'
    """
    assert mask.dim() == 3, "mask should be (B,S,S)"
    B, S, _ = mask.shape
    dev = mask.device
    w = mask.float().clamp_min(0)
    if max_side is None:
        max_side = S

    yy, xx = torch.meshgrid(
        torch.arange(S, device=dev) + 0.5,
        torch.arange(S, device=dev) + 0.5,
        indexing="ij",
    )
    # fallbacks for empty masks
    sumw = w.view(B, -1).sum(dim=1, keepdim=True)  # (B,1)
    empty = sumw.squeeze(1) <= 1e-8
    if empty.any():
        w = w.clone()
        w[empty, S // 2, S // 2] = 1.0

    # binarize and take tight bounding box, then convert to square
    y_mean = w.mean(dim=2)  # (B,S)
    x_mean = w.mean(dim=1)  # (B,S)
    y_pos = y_mean > thresh
    x_pos = x_mean > thresh
    # find the first and last positive token per batch
    ys = torch.argmax(y_pos.float(), dim=1)  # (B,)
    ye = S - 1 - torch.argmax(torch.flip(y_pos, dims=[1]).float(), dim=1)  # (B,)
    xs = torch.argmax(x_pos.float(), dim=1)  # (B,)
    xe = S - 1 - torch.argmax(torch.flip(x_pos, dims=[1]).float(), dim=1)  # (B,)
    side = torch.clamp(
        (torch.maximum(ye - ys + 1, xe - xs + 1).float()).ceil().long(),
        min=1,
        max=max_side,
    )
    # center from bbox center
    cx = (xs.float() + xe.float() + 1) / 2.0
    cy = (ys.float() + ye.float() + 1) / 2.0

    # compute top-left from center and side
    x1 = (cx - side.float() / 2).floor()
    y1 = (cy - side.float() / 2).floor()
    x2 = x1 + side - 1
    y2 = y1 + side - 1

    if clamp:
        max_x1 = (S - side).to(dtype=x1.dtype, device=x1.device)
        max_y1 = (S - side).to(dtype=y1.dtype, device=y1.device)

        x1 = x1.clamp_min(0)
        y1 = y1.clamp_min(0)
        x1 = torch.minimum(x1, max_x1)
        y1 = torch.minimum(y1, max_y1)

        x2 = x1 + side - 1
        y2 = y1 + side - 1
    else:
        # leave negatives / >S-1 as-is (caller can pad the crop)
        pass
    x1 = x1.round().long()
    y1 = y1.round().long()
    x2 = x2.round().long()
    y2 = y2.round().long()
    if mode == "xyxy_exclusive":
        return torch.stack([x1, y1, x2 + 1, y2 + 1], dim=1)
    elif mode == "xyxy_inclusive":
        return torch.stack([x1, y1, x2, y2], dim=1)
    else:
        raise ValueError("mode must be 'xyxy_exclusive' or 'xyxy_inclusive'")


def project_inner_box_to_original_px(
    inner_boxes: torch.Tensor, global_boxes: torch.Tensor, num_patches: int
) -> torch.Tensor:
    """
    Map each inner-box from the 14×14 grid of the *resized* crop back to an
    inclusive pixel box on the original image.  Works whatever the crop size
    (e.g. 518×518, 512×512, …).

    Returns (B,4)  int64  [px1, py1, px2, py2]  inclusive.

    Note: this funtion does not clamp the boxes to image bounds, left for the cropping function.
    """
    inner_boxes = inner_boxes.to(torch.float32)
    global_boxes = global_boxes.to(torch.float32)

    gx1, gy1, gx2, gy2 = global_boxes.unbind(1)
    ix1, iy1, ix2, iy2 = inner_boxes.unbind(1)

    crop_w = gx2 - gx1 + 1  # pixel width  of current crop
    crop_h = gy2 - gy1 + 1  # pixel height of current crop

    sx = crop_w / num_patches  # px per grid cell (≠ int if 512/14 etc.)
    sy = crop_h / num_patches
    px1_f = gx1 + ix1 * sx
    py1_f = gy1 + iy1 * sy
    px2_f = gx1 + ix2 * sx
    py2_f = gy1 + iy2 * sy

    px1 = torch.floor(px1_f)
    py1 = torch.floor(py1_f)
    px2 = torch.ceil(px2_f)
    py2 = torch.ceil(py2_f)
    return torch.stack([px1, py1, px2, py2], 1).to(torch.float32)


def crop_and_resize(
    imgs: torch.Tensor, boxes: torch.Tensor, output_size, clamp: bool = False
) -> torch.Tensor:
    """
    Vectorised crop-and-resize using visual focusAlign (bilinear, GPU-friendly).

    If `clamp=False` the boxes are clamped to image bounds first.
    If `False` we feed the raw boxes to visual focusAlign – samples falling outside
    are automatically zero-filled.
    """
    B, _, H, W = imgs.shape
    boxes_clamped = boxes.clone()
    if clamp:
        boxes_clamped[:, 0].clamp_(0, W - 1)
        boxes_clamped[:, 1].clamp_(0, H - 1)
        boxes_clamped[:, 2].clamp_(0, W - 1)
        boxes_clamped[:, 3].clamp_(0, H - 1)

    batch_ix = torch.arange(B, device=imgs.device).float().unsqueeze(1)
    regions = torch.cat([batch_ix, boxes_clamped.float()], dim=1)  # (B,5)
    with torch.no_grad():
        crops = roi_align(
            imgs, regions, output_size=output_size, spatial_scale=1.0, aligned=True
        )
    return crops
