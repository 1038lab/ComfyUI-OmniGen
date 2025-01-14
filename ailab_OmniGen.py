import sys
import os.path as osp
import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
import requests
import folder_paths
import tempfile
import shutil
import json
import uuid

# Define all path constants
class Paths:
    ROOT_DIR = osp.dirname(__file__)
    MODELS_DIR = folder_paths.models_dir
    LLM_DIR = osp.join(MODELS_DIR, "LLM")
    OMNIGEN_DIR = osp.join(LLM_DIR, "OmniGen-v1")
    OMNIGEN_CODE_DIR = osp.join(ROOT_DIR, "OmniGen")
    TMP_DIR = osp.join(ROOT_DIR, "tmp")
    MODEL_FILE_FP16 = osp.join(OMNIGEN_DIR, "model.safetensors")
    MODEL_FILE_FP8 = osp.join(OMNIGEN_DIR, "model-fp8_e4m3fn.safetensors")

# Ensure necessary directories exist
os.makedirs(Paths.LLM_DIR, exist_ok=True)
sys.path.append(Paths.ROOT_DIR)

class ailab_OmniGen:
    VERSION = "1.2.0"
    _model_instance = None
    _current_precision = None
    
    # Load preset prompts
    try:
        json_path = osp.join(osp.dirname(__file__), "data.json")
        if osp.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                PRESET_PROMPTS = data.get("PRESET_PROMPTS", {"None": ""})
        else:
            PRESET_PROMPTS = {"None": ""}
    except Exception as e:
        print(f"Error loading preset prompts: {e}")
        PRESET_PROMPTS = {"None": ""}

    def __init__(self):
        self._ensure_code_exists()
        self._ensure_model_exists()
        
        try:
            from OmniGen import OmniGenPipeline
            self.OmniGenPipeline = OmniGenPipeline
        except ImportError as e:
            print(f"Error importing OmniGen: {e}")
            raise RuntimeError("Failed to import OmniGen. Please check if the code was downloaded correctly.")

    def _ensure_code_exists(self):
        """Ensure OmniGen code exists, download from GitHub if not"""
        try:
            if not osp.exists(Paths.OMNIGEN_CODE_DIR):
                print("Downloading OmniGen code from GitHub...")

                files = [
                    "model.py",
                    "pipeline.py", 
                    "processor.py",
                    "scheduler.py",
                    "transformer.py",
                    "utils.py",
                    "__init__.py"
                ]
                
                os.makedirs(Paths.OMNIGEN_CODE_DIR, exist_ok=True)
                base_url = "https://raw.githubusercontent.com/VectorSpaceLab/OmniGen/main/OmniGen/"
                
                for file in files:
                    url = base_url + file
                    response = requests.get(url)
                    if response.status_code == 200:
                        with open(osp.join(Paths.OMNIGEN_CODE_DIR, file), 'wb') as f:
                            f.write(response.content)
                        print(f"Downloaded {file}")
                    else:
                        raise RuntimeError(f"Failed to download {file}: {response.status_code}")
                
                print("OmniGen code setup completed")
                
                if Paths.OMNIGEN_CODE_DIR not in sys.path:
                    sys.path.append(Paths.OMNIGEN_CODE_DIR)
                        
            else:
                print("OmniGen code already exists")
                    
        except Exception as e:
            print(f"Error downloading OmniGen code: {e}")
            raise RuntimeError(f"Failed to download OmniGen code: {str(e)}")

    def _ensure_model_exists(self, model_precision=None):
        """Ensure model file exists, download if not"""
        try:
            os.makedirs(Paths.OMNIGEN_DIR, exist_ok=True)
            
            # Download FP8 model if specified and not exists
            if model_precision == "FP8" and not osp.exists(Paths.MODEL_FILE_FP8):
                print("FP8 model not found, downloading from Hugging Face...")
                url = "https://huggingface.co/silveroxides/OmniGen-V1/resolve/main/model-fp8_e4m3fn.safetensors"
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(Paths.MODEL_FILE_FP8, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print("FP8 model downloaded successfully")
                else:
                    raise RuntimeError(f"Failed to download FP8 model: {response.status_code}")
            
            # Check if FP16 model exists
            if not osp.exists(Paths.MODEL_FILE_FP16):
                print("FP16 model not found, starting download from Hugging Face...")
                snapshot_download(
                    repo_id="silveroxides/OmniGen-V1",
                    local_dir=Paths.OMNIGEN_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    token=None,
                    tqdm_class=None,
                )
                print("FP16 model downloaded successfully")
            
            # Verify model files exist after download
            if model_precision == "FP8" and not osp.exists(Paths.MODEL_FILE_FP8):
                raise RuntimeError("FP8 model download failed")
            if not osp.exists(Paths.MODEL_FILE_FP16):
                raise RuntimeError("FP16 model download failed")
                
            print("OmniGen models verified successfully")
            
        except Exception as e:
            print(f"Error during model initialization: {e}")
            raise RuntimeError(f"Failed to initialize OmniGen model: {str(e)}")

    def _setup_temp_dir(self):
        """Set up temporary directory with unique name"""
        self._temp_dir = osp.join(Paths.TMP_DIR, str(uuid.uuid4()))
        os.makedirs(self._temp_dir, exist_ok=True)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if hasattr(self, '_temp_dir') and osp.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    def _auto_select_precision(self):
        """Automatically select precision based on available VRAM"""
        if torch.cuda.is_available():
            vram_size = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_size < 8:
                print(f"Auto selecting FP8 (Available VRAM: {vram_size:.1f}GB)")
                return "FP8"
            print(f"Auto selecting FP16 (Available VRAM: {vram_size:.1f}GB)")
        return "FP16"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset_prompt": (list(s.PRESET_PROMPTS.keys()), {"default": "None"}),
                "prompt": ("STRING", {"multiline": True, "forceInput": False, "default": ""}),
                "model_precision": (["Auto", "FP16", "FP8"], {"default": "Auto"}),
                "memory_management": (["Balanced", "Speed Priority", "Memory Priority"], {"default": "Balanced"}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 5.0, "step": 0.1, "round": 0.01}),
                "img_guidance_scale": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 2.0, "step": 0.1, "round": 0.01}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "separate_cfg_infer": ("BOOLEAN", {"default": True}),
                "use_input_image_size_as_output": ("BOOLEAN", {"default": False}),
                "width": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_input_image_size": ("INT", {"default": 1024, "min": 128, "max": 2048, "step": 16}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generation"
    CATEGORY = "🧪AILab/OmniGen"

    def save_input_img(self, image):
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=Paths.TMP_DIR) as f:
                img_np = image.numpy()[0] * 255
                img_pil = Image.fromarray(img_np.astype(np.uint8))
                img_pil.save(f.name)
                return f.name
        except Exception as e:
            print(f"Error saving input image: {e}")
            return None

    def _process_prompt_and_images(self, prompt, images):
        """Process prompt and images, return updated prompt and image paths"""
        input_images = []
        
        # Auto-generate prompt if empty but images provided
        if not prompt and any(images):
            prompt = " ".join(f"<img><|image_{i+1}|></img>" for i, img in enumerate(images) if img is not None)
        
        # Process each image
        for i, img in enumerate(images, 1):
            if img is not None:
                img_path = self.save_input_img(img)
                if img_path:
                    input_images.append(img_path)
                    img_tag = f"<img><|image_{i}|></img>"
                    if f"image_{i}" in prompt:
                        prompt = prompt.replace(f"image_{i}", img_tag)
                    elif f"image{i}" in prompt:
                        prompt = prompt.replace(f"image{i}", img_tag)
                    elif img_tag not in prompt:
                        prompt += f" {img_tag}"
        
        return prompt.strip(), input_images

    def _check_sdpa_support(self):
        """Check if system supports Scaled Dot Product Attention"""
        try:
            import torch
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                return True
            return False
        except Exception as e:
            print(f"Error checking SDPA support: {e}")
            return False

    def _get_pipeline(self, model_precision, keep_in_vram):
        try:
            # Reuse existing instance if available
            if keep_in_vram and self._model_instance and self._current_precision == model_precision:
                print("Reusing existing pipeline instance")
                return self._model_instance

            # Check model file
            model_file = Paths.MODEL_FILE_FP8 if model_precision == "FP8" else Paths.MODEL_FILE_FP16
            if not os.path.exists(model_file):
                raise RuntimeError(f"Model file not found: {model_file}")
                
            # Create pipeline
            try:
                # Initialize pipeline
                pipe = self.OmniGenPipeline.from_pretrained(Paths.OMNIGEN_DIR)
                    
                if pipe is None:
                    raise RuntimeError("Initial pipeline creation failed")
                    
                # Save original pipeline reference before moving to device
                original_pipe = pipe
                    
                # Move to device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    # Move model components first
                    if hasattr(pipe, 'text_encoder'):
                        pipe.text_encoder = pipe.text_encoder.to(device)
                    if hasattr(pipe, 'unet'):
                        pipe.unet = pipe.unet.to(device)
                    if hasattr(pipe, 'vae'):
                        pipe.vae = pipe.vae.to(device)
                        
                    # Then move entire pipeline
                    pipe = pipe.to(device)
                    
                    # Use original pipeline if None after moving
                    if pipe is None:
                        print("Warning: Pipeline.to(device) returned None, using original pipeline")
                        pipe = original_pipe
                        
                except Exception as e:
                    print(f"Warning: Error moving pipeline to device: {e}, using original pipeline")
                    pipe = original_pipe
                    
                # Validate pipeline
                if not callable(pipe):
                    raise RuntimeError("Pipeline is not callable after initialization")
                    
                # Save instance if needed
                if keep_in_vram:
                    self._model_instance = pipe
                    self._current_precision = model_precision
                    
                return pipe
                    
            except Exception as pipe_error:
                print(f"Pipeline creation error: {pipe_error}")
                raise
                
        except Exception as e:
            print(f"Fatal error in pipeline creation: {str(e)}")
            raise RuntimeError(f"Failed to create pipeline: {str(e)}")

    def generation(self, preset_prompt, model_precision, prompt, memory_management, num_inference_steps, guidance_scale,
            img_guidance_scale, max_input_image_size, separate_cfg_infer,
            use_input_image_size_as_output, width, height, seed,
            image_1=None, image_2=None, image_3=None):

        keep_in_vram = (memory_management == "Speed Priority")
        offload_model = (memory_management == "Memory Priority")

        try:
            # Auto select precision if Auto is chosen
            if model_precision == "Auto":
                model_precision = self._auto_select_precision()
            
            self._setup_temp_dir()
            
            # Clear existing instance if precision doesn't match
            if self._model_instance and self._current_precision != model_precision:
                print(f"Precision changed from {self._current_precision} to {model_precision}, clearing instance")
                self._model_instance = None
                self._current_precision = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("VRAM cleared")
                    
            # Memory management strategy
            if memory_management == "Memory Priority":
                print("Memory Priority mode: Forcing pipeline recreation")
                self._model_instance = None
                self._current_precision = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Check model instance status
            print(f"Current model instance: {'Present' if self._model_instance else 'None'}")
            print(f"Current model precision: {self._current_precision}")
            
            final_prompt = prompt.strip() if prompt.strip() else self.PRESET_PROMPTS[preset_prompt]
            pipe = self._get_pipeline(model_precision, keep_in_vram)
            
            # Monitor VRAM usage
            if torch.cuda.is_available():
                print(f"VRAM usage after pipeline creation: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            # Process prompt and images
            final_prompt, input_images = self._process_prompt_and_images(final_prompt, [image_1, image_2, image_3])
            input_images = input_images if input_images else None
            
            print(f"Processing with prompt: {final_prompt}")
            print(f"Model will be {'offloaded' if offload_model else 'kept'} during generation")
            
            output = pipe(
                prompt=final_prompt,
                input_images=input_images,
                guidance_scale=guidance_scale,
                img_guidance_scale=img_guidance_scale,
                num_inference_steps=num_inference_steps,
                separate_cfg_infer=separate_cfg_infer, 
                use_kv_cache=True,
                offload_kv_cache=True,
                offload_model=offload_model,
                use_input_image_size_as_output=use_input_image_size_as_output,
                width=width,
                height=height,
                seed=seed,
                max_input_image_size=max_input_image_size,
            )
            
            # Print VRAM usage after generation
            if torch.cuda.is_available():
                print(f"VRAM usage after generation: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
            img = np.array(output[0]) / 255.0
            img = torch.from_numpy(img).unsqueeze(0)
            
            # Clean up if not keeping in VRAM
            if not keep_in_vram:
                del pipe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            return (img,)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            raise e
        finally:
            self._cleanup_temp_dir()
            if not keep_in_vram and torch.cuda.is_available():
                torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {
    "ailab_OmniGen": ailab_OmniGen
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ailab_OmniGen": "OmniGen 🖼️"
}