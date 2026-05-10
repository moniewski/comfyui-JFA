import torch
import torch.nn.functional as F
import math

class JFANode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_channel": (["Red", "Green", "Blue", "Alpha"], {"default": "Red"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "threshold_preview": ("BOOLEAN", {"default": False}),
                "distance_mode": (["percentage", "pixels"], {"default": "percentage"}),
                "distance": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 2048.0, "step": 0.01}),
                "distance_falloff": (["linear", "smoothstep", "smootherstep"], {"default": "linear"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("dist_mask", "dist_image", "threshold_mask", "jfa_coords")
    FUNCTION = "execute"
    CATEGORY = "custom/jfa"

    def execute(self, source_channel, threshold, threshold_preview, distance_mode, distance, distance_falloff, invert_mask, mask=None, image=None):
        # 1. Normalization
        if mask is not None:
            device = mask.device
            work_mask = mask.unsqueeze(1)
            B, _, H_orig, W_orig = work_mask.shape
        elif image is not None:
            B, H_orig, W_orig, C = image.shape
            device = image.device
            ch_idx = {"Red": 0, "Green": 1, "Blue": 2, "Alpha": 3}[source_channel]
            if ch_idx >= C: ch_idx = 0 
            work_mask = image[:, :, :, ch_idx].unsqueeze(1)
        else:
            raise ValueError("JFA Node Error: No input connected.")

        # 2. Setup PoT Space
        max_side = max(H_orig, W_orig)
        work_res = 2 ** math.ceil(math.log2(max_side))
        scale_factor = work_res / max_side
        new_h, new_w = round(H_orig * scale_factor), round(W_orig * scale_factor)
        offset_y, offset_x = (work_res - new_h) // 2, (work_res - new_w) // 2
        
        resized = F.interpolate(work_mask, size=(new_h, new_w), mode='bilinear', align_corners=False)
        binary_mask = (resized > threshold).float()

        # 3. Fast Preview
        threshold_out = F.interpolate(binary_mask, size=(H_orig, W_orig), mode='nearest').squeeze(1)
        threshold_img = threshold_out.unsqueeze(-1).repeat(1, 1, 1, 3)
        if threshold_preview:
            return (threshold_out, threshold_img, threshold_img, torch.zeros((B, H_orig, W_orig, 3), device=device))

        # 4. JFA Logic
        curr_map = torch.full((B, 2, work_res, work_res), 1e6, device=device)
        y, x = torch.meshgrid(torch.arange(work_res, device=device), torch.arange(work_res, device=device), indexing='ij')
        coords = torch.stack((x, y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1).float()
        
        curr_map[:, :, offset_y:offset_y+new_h, offset_x:offset_x+new_w] = torch.where(
            binary_mask > 0, coords[:, :, offset_y:offset_y+new_h, offset_x:offset_x+new_w], torch.tensor([1e6], device=device)
        )

        num_steps = int(math.log2(work_res))
        for i in range(num_steps - 1, -1, -1):
            step = 2**i
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                s_x, s_y = dx * step, dy * step
                shifted = F.pad(curr_map, (max(0, -s_x), max(0, s_x), max(0, -s_y), max(0, s_y)), value=1e6)
                sampled = shifted[:, :, max(0, s_y):max(0, s_y)+work_res, max(0, s_x):max(0, s_x)+work_res]
                d_c = torch.sum((curr_map - coords)**2, dim=1)
                d_n = torch.sum((sampled - coords)**2, dim=1)
                curr_map = torch.where((d_n < d_c).unsqueeze(1), sampled, curr_map)

        # 5. Normalization & Falloff
        divisor = (distance * work_res) if distance_mode == "percentage" else (distance * scale_factor)
        divisor = max(divisor, 1e-4)
        dist_field = torch.sqrt(torch.sum((curr_map - coords)**2, dim=1))
        norm = (dist_field / divisor).clamp(0.0, 1.0)
        
        if distance_falloff == "smoothstep":
            norm = norm * norm * (3.0 - 2.0 * norm)
        elif distance_falloff == "smootherstep":
            norm = norm * norm * norm * (norm * (6.0 * norm - 15.0) + 10.0)
        
        if not invert_mask: norm = 1.0 - norm

        # 6. Uncrop & Format
        dist_roi = norm[:, offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        jfa_roi = curr_map[:, :, offset_y:offset_y+new_h, offset_x:offset_x+new_w]
        
        final_mask = F.interpolate(dist_roi.unsqueeze(1), size=(H_orig, W_orig), mode='bilinear', align_corners=False).squeeze(1)
        final_coords = F.interpolate(jfa_roi, size=(H_orig, W_orig), mode='nearest')
        
        dist_image = final_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
        coords_vis = torch.cat(((final_coords / work_res).permute(0, 2, 3, 1), torch.zeros((B, H_orig, W_orig, 1), device=device)), dim=-1)
        
        return (final_mask, dist_image, threshold_img, coords_vis)
        