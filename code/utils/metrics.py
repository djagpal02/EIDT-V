import numpy as np
import torch
import cv2
import torchvision
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from pytorch_msssim import ms_ssim as pytorch_ms_ssim
import torch.nn.functional as F
# Local imports
from utils.load import load_clip, load_lpips

class Metrics:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.clip_model, self.clip_processor = load_clip()
        self.clip_model = self.clip_model.to(self.device)
        self.lpips_model = load_lpips().to(self.device)
        self.to_tensor = transforms.ToTensor()
        self.optical_flow = OpticalFlow()

    def compute_ssim(self, image1, image2):
        if image1.size != image2.size:
            raise ValueError("Images must be of the same size")
        image1_np = np.array(image1).astype(np.uint8)
        image2_np = np.array(image2).astype(np.uint8)
        return ssim(image1_np, image2_np, channel_axis=-1, data_range=255, full=False)

    def ms_ssim(self, image1, image2):
        if image1.size != image2.size:
            raise ValueError("Images must be of the same size")
        img1_tensor = self.to_tensor(image1).unsqueeze(0).to(self.device)
        img2_tensor = self.to_tensor(image2).unsqueeze(0).to(self.device)
        return pytorch_ms_ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True).item()

    def lpips(self, image1, image2):
        if image1.size != image2.size:
            raise ValueError("Images must be of the same size")
        img1_tensor = self.to_tensor(image1).unsqueeze(0).to(self.device)
        img2_tensor = self.to_tensor(image2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.lpips_model(img1_tensor, img2_tensor).item()

    def clip_score(self, image, prompt):
        if not image or not prompt:
            raise ValueError("Image and prompt cannot be empty")
        inputs = self.clip_processor(text=prompt, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()

    def text_cosine_similarity(self, texts):
        if not texts:
            raise ValueError("Texts cannot be empty")
        texts = [texts] if isinstance(texts, str) else texts
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_embeds = self.clip_model.get_text_features(**inputs)
        text_embeds_normalized = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        return torch.mm(text_embeds_normalized, text_embeds_normalized.t()).cpu().numpy()

    def clip_score_frames(self, frames, prompt):
        if not frames:
            raise ValueError("Frames cannot be empty")
        inputs = self.clip_processor(text=[prompt] * len(frames), images=frames, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        cosine_similarities = torch.nn.functional.cosine_similarity(outputs.image_embeds, outputs.text_embeds)
        return cosine_similarities.mean().item()

    def ms_ssim_frames(self, frames):
        if not frames:
            raise ValueError("Frames cannot be empty")
        frame_tensors = torch.stack([self.to_tensor(frame) for frame in frames]).to(self.device)
        ms_ssim_scores = [
            pytorch_ms_ssim(frame_tensors[i:i+1], frame_tensors[i+1:i+2], data_range=1.0)
            for i in range(len(frames) - 1)
        ]
        return torch.mean(torch.tensor(ms_ssim_scores)).item()

    def lpips_frames(self, frames):
        if not frames:
            raise ValueError("Frames cannot be empty")
        frame_tensors = torch.stack([self.to_tensor(frame) for frame in frames]).to(self.device)
        with torch.no_grad():
            lpips_scores = [
                self.lpips_model(frame_tensors[i:i+1], frame_tensors[i+1:i+2]).item()
                for i in range(len(frames) - 1)
            ]
        return np.mean(lpips_scores)

    def temporal_consistency_loss(self, frames):
        if not frames:
            raise ValueError("Frames cannot be empty")
        # Total, warp, smooth
        return self.optical_flow(frames)


class OpticalFlow:
    def __init__(self, device='cuda'):
        # Set device to GPU if available, else CPU
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def compute_optical_flow_cv2(self, prev_frame, next_frame):
        if prev_frame.shape != next_frame.shape:
            raise ValueError("Frames must be of the same size")
        """
        Compute optical flow between two frames using OpenCV's Farneback method.

        :param prev_frame: Grayscale image of the previous frame (H, W)
        :param next_frame: Grayscale image of the next frame (H, W)
        :return: Optical flow array of shape (H, W, 2)
        """
        # Compute dense optical flow using Farneback's algorithm
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, next_frame, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        return flow  # Shape: (H, W, 2)

    def warp_frame(self, frame, flow):
        if frame.size()[-2:] != flow.size()[1:3]:
            raise ValueError("Frame and flow must be of the same size")
        B, C, H, W = frame.size()

        # Create a mesh grid for pixel coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'  # Use 'ij' indexing to match (H, W)
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1).float()
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1).float()

        # Adjust grid with flow values
        flow_x = flow[..., 0]  # Shape: (B, H, W)
        flow_y = flow[..., 1]  # Shape: (B, H, W)
        vgrid_x = grid_x + flow_x
        vgrid_y = grid_y + flow_y

        # Normalize grid to [-1, 1]
        vgrid_x = 2.0 * vgrid_x / (W - 1) - 1.0
        vgrid_y = 2.0 * vgrid_y / (H - 1) - 1.0

        vgrid = torch.stack((vgrid_x, vgrid_y), dim=3)  # Shape: (B, H, W, 2)
        warped_frame = F.grid_sample(frame, vgrid, align_corners=True)
        return warped_frame

    def flow_smoothness_loss(self, flow):
        if flow.numel() == 0:
            raise ValueError("Flow cannot be empty")
        """
        Compute the flow smoothness loss to encourage smooth flow fields.

        :param flow: Tensor of shape (1, H, W, 2)
        :return: Smoothness loss (scalar tensor)
        """
        # Compute gradients of the flow field
        flow_dx = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        flow_dy = flow[:, 1:, :, :] - flow[:, :-1, :, :]

        # Compute the smoothness loss as the L1 norm of flow gradients
        loss = torch.mean(torch.abs(flow_dx)) + torch.mean(torch.abs(flow_dy))
        return loss

    def temporal_consistency_loss_cv2(self, frames):
        if not frames:
            raise ValueError("Frames cannot be empty")
        """
        Compute the optical flow-based temporal consistency loss for a sequence of frames.

        :param frames: List of PIL Images
        :return: Total loss (scalar), loss_warp (scalar), loss_smooth (scalar)
        """
        loss_warp = 0.0
        loss_smooth = 0.0

        # Convert frames to grayscale numpy arrays
        frames_gray = [cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY) for frame in frames]

        # Convert frames to tensors and move to device
        frame_tensors = [self.to_tensor(frame).unsqueeze(0).to(self.device) for frame in frames]

        for i in range(len(frames) - 1):
            prev_gray = frames_gray[i]
            next_gray = frames_gray[i + 1]
            frame1 = frame_tensors[i]
            frame2 = frame_tensors[i + 1]

            # Compute optical flow using OpenCV (CPU-based)
            flow_np = self.compute_optical_flow_cv2(prev_gray, next_gray)  # Shape: (H, W, 2)

            # Convert flow to tensor and move to device
            flow = torch.from_numpy(flow_np).unsqueeze(0).to(self.device)  # Shape: (1, H, W, 2)

            # Warp frame1 to frame2 using flow
            warped_frame1 = self.warp_frame(frame1, flow)

            # Compute Warping Loss (L1 Loss between warped frame and actual frame2)
            # This loss ensures that the warped frame aligns with the next frame,
            # promoting temporal consistency between frames.
            loss_warp += F.l1_loss(warped_frame1, frame2)

            # Compute Flow Smoothness Loss
            # This loss encourages the optical flow to be smooth across the spatial domain,
            # preventing abrupt changes in flow and promoting natural motion.
            loss_smooth += self.flow_smoothness_loss(flow)

        # Average the losses over the number of frame pairs
        num_pairs = len(frames) - 1
        loss_warp /= num_pairs
        loss_smooth /= num_pairs

        # Combine the losses with appropriate weighting
        # The weighting factor balances the influence of each loss component.
        total_loss = loss_warp + 0.1 * loss_smooth

        return total_loss.item(), loss_warp.item(), loss_smooth.item()

    def __call__(self, frames):
        if not frames:
            raise ValueError("Frames cannot be empty")
        return self.temporal_consistency_loss_cv2(frames)
