"""
Visual Encoders.

@yjy0625
"""
import torch
import timm
from PIL import Image
import torchvision.transforms as T


class DinoEncoder:
    """
    A wrapper around a DINO ViT model (e.g., vit_small_patch16_224_dino) that:
      1. Loads a pretrained DINO model from timm.
      2. Returns a single embedding vector (e.g., 384 or 768 dims) for a given image.
    """

    def __init__(self, model_name='vit_small_patch16_224_dino', device='cpu'):
        """
        :param model_name: Name of the DINO model in timm (e.g. 'vit_small_patch16_224_dino',
                           'vit_base_patch16_224_dino')
        :param device:     'cpu' or 'cuda' for GPU inference.
        """
        self.device = device

        # Load a pretrained DINO model from timm
        # Examples: 
        #   'vit_small_patch16_224_dino' -> embedding dim ~ 384
        #   'vit_base_patch16_224_dino'  -> embedding dim ~ 768
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Remove or bypass the classification head to get raw embeddings
        # Note: DINO models in timm often store their head in self.model.head.
        # We can set it to an identity layer so forward() returns just the backbone output.
        if hasattr(self.model, 'head'):
            self.model.head = torch.nn.Identity()

        # Create standard transforms for a 224x224 DINO model
        # You can adjust the image size or normalization if using a different variant
        # self.preprocess = T.Compose([
        #     T.Resize((224, 224)),
        #     T.Normalize(
        #         mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225]
        #     )
        # ])
        data_config = timm.data.resolve_model_data_config(self.model)
        self.preprocess = timm.data.create_transform(**data_config, is_training=False)
    
    def _load_and_preprocess(self, images):
        """
        Internal helper to load and preprocess the image for the DINO model.
        :param images: input images as a tensor with shape (N, 3, H, W)
        :return: A preprocessed tensor of shape (N, 3, H, W)
        """
        input_tensor = self.preprocess(images).to(self.device)
        return input_tensor

    def __call__(self, image_path):
        """
        Forward pass to get the DINO embedding.
        :param image_path: Path to the input image
        :return: A 1D numpy array of shape (embedding_dim,)
        """
        input_tensor = self._load_and_preprocess(image_path)
        with torch.no_grad():
            embedding = self.model(input_tensor)  # shape: (1, embed_dim)
        return embedding


class DinoDenseDescriptorEncoder:
    """
    A wrapper around a DINO ViT model that returns dense descriptors for a given image.
    The dense descriptors have shape (H', W', C) for each image.
    """
    def __init__(self, model_name='vit_base_patch16_224_dino', device='cpu'):
        self.device = device

        # Load a pretrained DINO model from timm
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # Remove classification head to get feature map output
        if hasattr(self.model, 'head'):
            self.model.head = torch.nn.Identity()

        # Create transforms for the model
        data_config = timm.data.resolve_model_data_config(self.model)
        self.preprocess = timm.data.create_transform(**data_config, is_training=False)

    def _load_and_preprocess(self, images):
        """
        Internal helper to preprocess the image for the DINO model.
        :param images: input images as a tensor with shape (N, 3, H, W)
        :return: A preprocessed tensor of shape (N, 3, H, W)
        """
        return self.preprocess(images).to(self.device)

    def __call__(self, images):
        """
        Forward pass to get dense descriptors.
        :param images: Input images tensor with shape (N, 3, H, W)
        :return: Dense feature maps of shape (N, 1 + H' * W', C)
        """
        images = self._load_and_preprocess(images)
        with torch.no_grad():
            feature_maps = self.model.forward_features(images)  # Shape: (N, 1 + H' * W', C)
            feature_maps = feature_maps[:, 1:, :] # (N, H'W', C)
        return feature_maps


class DepthAnythingEncoder:
    def __init__(self, model_name='depth-anything/Depth-Anything-V2-Small-hf', device='cpu'):
        self.device = device

        self.preprocess = T.Resize((224, 224))

        self.pipeline = pipeline(task="depth-estimation", model=model_name, device=device)

    def __call__(self, image_path):
        results = []
        for image in self.preprocess(image_path):
            image_np = image.detach().cpu().numpy().transpose(1, 2, 0)
            image_pil = Image.fromarray(np.array(image_np * 255, dtype=np.uint8))
            depth = np.array(self.pipeline(image_pil)["depth"])
            depth = cv2.resize(depth.copy(), (25, 25))
            depth = depth.flatten().astype(np.float32)
            depth /= depth.max()
            results.append(depth)
        return torch.tensor(results)


class PolicyEncoder:
    def __init__(self, config, ckpt_path, device='cpu'):
        print(f"Initializing policy encoder")
        self.device = device
        self.preprocess = T.Resize((116, 116))
        self.encoder_dict = {}
        self.obs_key = None
        self.lang_emb = None

        from mobipi.utils.policy_utils import load_policy
        self.model, rollout_model, self.lang_encoder = load_policy(config, ckpt_path)

    def set_lang(self, lang):
        self.lang_emb = self.lang_encoder.get_lang_emb(lang)

    def set_obs_key(self, obs_key):
        self.obs_key = obs_key
        if not obs_key in self.encoder_dict:
            encoder = self.model.nets.policy.nets["encoder"].nets["obs"].obs_nets[obs_key]
            encoder.to(self.device)
            self.encoder_dict[obs_key] = encoder

    @property
    def encoder(self):
        assert self.obs_key is not None
        return self.encoder_dict[self.obs_key]

    def __call__(self, image_path):
        with torch.no_grad():
            lang_emb_batch = self.lang_emb[None].repeat(len(image_path), 1).to(self.device)
            emb = self.encoder(self.preprocess(image_path).to(self.device), lang_emb_batch)
        return emb
