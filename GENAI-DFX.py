import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, AutoPipelineForText2Image, ControlNetModel, StableDiffusionControlNetPipeline, EulerAncestralDiscreteScheduler
import ollama
import io
from PIL import Image, ImageFilter
import base64
import random
import time
import gc
import numpy as np
from contextlib import contextmanager
import os
import logging


# Add this after the imports in Ideate_Indus_design.py
MODEL_NAME = "mistral"

def generate_text_with_mistral(prompt: str, max_tokens: int = 256) -> str:
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.7, "max_tokens": max_tokens, "stop": ["</s>"]}
        )
        return response["response"].strip()
    except Exception as e:
        logger.error(f"Text generation failed: {str(e)}")
        st.error(f"Text generation failed: {e}")
        return ""

def generate_image_description(image: Image.Image, aspect: str, user_text: str, category: str, form: str) -> str:
    rules = get_dfx_rules(aspect)
    description = (
        f"A {category} product, specifically {user_text}, designed with a {form} form. "
        f"It features a minimalist aesthetic, solid black strap (if applicable), bold colors, and a matte finish. "
        f"The design is optimized for {aspect} with guidelines: {', '.join(rules['positive'])}. "
        f"It avoids: {', '.join(rules['negative'])}."
    )
    try:
        gray = image.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        complexity = np.mean(np.array(edges)) / 255.0
        if complexity < 0.3:
            description += f" The design exhibits low visual complexity, with smooth, simple shapes ideal for {category} products and {form} aesthetics."
        elif complexity > 0.7:
            description += f" The design has high visual complexity, with intricate details that may challenge {category} and {form} requirements."
        else:
            description += f" The design has moderate visual complexity, balancing simplicity and detail suitable for {category} and {form}."
        img_array = np.array(image)
        dominant_color = np.mean(img_array, axis=(0, 1))[:3]
        if dominant_color.max() < 100:
            description += " The color palette is dark, emphasizing the matte finish and minimalist design."
        elif dominant_color.max() > 200:
            description += " The color palette is vibrant, aligning with bold color preferences."
        else:
            description += " The color palette is balanced, supporting the minimalist and bold aesthetic."
    except Exception as e:
        logger.warning(f"Failed to analyze image for description: {str(e)}")
        st.warning(f"Failed to analyze image for description: {e}")
        description += " Visual analysis unavailable due to processing error."
    return description
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sdcache_directory = r"E:\Projet-metier-GenAi\Models used\huggingface\hub"

@st.cache_resource
def load_model(model_id, device, use_low_memory=False, model_type="sd"):
    """Load and cache the model to improve performance"""
    logger.debug(f"Loading model: {model_id}, type: {model_type}, device: {device}, low_memory: {use_low_memory}")
    try:
        if not os.path.exists(sdcache_directory):
            os.makedirs(sdcache_directory)
            logger.info(f"Created cache directory: {sdcache_directory}")

        if model_type == "sd":
            if device == "cuda" and use_low_memory:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    revision="fp16",
                    safety_checker=None,
                    cache_dir=sdcache_directory,
                    load_in_8bit=True
                )
                pipe = pipe.to(device)
                pipe.enable_xformers_memory_efficient_attention()
                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    safety_checker=None,
                    use_safetensors=True,
                    cache_dir=sdcache_directory
                )
                pipe = pipe.to(device)
            
            if model_id in ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1-base", "prompthero/openjourney-v4"]:
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-canny",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    cache_dir=sdcache_directory
                )
                pipe_cn = StableDiffusionControlNetPipeline(
                    vae=pipe.vae,
                    text_encoder=pipe.text_encoder,
                    tokenizer=pipe.tokenizer,
                    unet=pipe.unet,
                    controlnet=controlnet,
                    scheduler=EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
                    safety_checker=pipe.safety_checker,
                    feature_extractor=pipe.feature_extractor
                ).to(device)
                return {"base": pipe, "controlnet": pipe_cn}
        
        elif model_type == "sdxl":
            if device == "cuda":
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16" if use_low_memory else None,
                    use_safetensors=True,
                    cache_dir=sdcache_directory
                )
                pipe = pipe.to(device)
                if use_low_memory:
                    pipe.enable_model_cpu_offload()
                    pipe.enable_vae_slicing()
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    cache_dir=sdcache_directory
                )
                pipe = pipe.to(device)
            return {"base": pipe, "controlnet": None}
        
        elif model_type == "if":
            if device == "cuda":
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    cache_dir=sdcache_directory
                )
                pipe = pipe.to(device)
                if use_low_memory:
                    pipe.enable_model_cpu_offload()
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    cache_dir=sdcache_directory
                )
                pipe = pipe.to(device)
            return {"base": pipe, "controlnet": None}
        
        return {"base": pipe, "controlnet": None}
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        st.error(f"Error loading model: {str(e)}. Check if model files exist in {sdcache_directory} or if internet is available to download from Hugging Face.")
        return None

def clear_previous_model():
    """Clear the cached model from memory"""
    logger.debug("Clearing previous model from cache")
    load_model.clear()
    torch.cuda.empty_cache()
    gc.collect()

@contextmanager
def track_memory_usage(show_stats=False):
    if torch.cuda.is_available() and show_stats:
        torch.cuda.empty_cache()
        gc.collect()
        start_mem = torch.cuda.memory_allocated() / 1024**2
        yield
        end_mem = torch.cuda.memory_allocated() / 1024**2
        st.info(f"GPU Memory: {start_mem:.2f}MB → {end_mem:.2f}MB (Diff: {end_mem - start_mem:.2f}MB)")
    else:
        yield

def process_sketch(sketch_file, image_size, threshold=100, blur_radius=0.5):
    """Process uploaded sketch into a ControlNet-compatible mask"""
    logger.debug(f"Processing sketch: size={image_size}, threshold={threshold}, blur_radius={blur_radius}")
    try:
        sketch = Image.open(sketch_file).convert("RGB")
        if sketch.size[0] < 50 or sketch.size[1] < 50:
            st.error("Sketch image is too small. Minimum size is 50x50 pixels.")
            return None
        sketch = sketch.resize(image_size, Image.Resampling.LANCZOS)
        gray = sketch.convert('L').filter(ImageFilter.GaussianBlur(radius=blur_radius))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        mask_array = np.array(edges)
        mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_array).convert('RGB')
        return mask
    except Exception as e:
        logger.error(f"Failed to process sketch: {str(e)}")
        st.error(f"Failed to process sketch: {str(e)}. Ensure the file is a valid PNG or JPEG image.")
        return None

# Integrated from projet-dfx.py: DfX rules for manual scoring
def get_dfx_rules(aspect):
    rules = {
        'DFA': {
            'positive': [
                'minimize part count',
                'use self-locating features',
                'no more than 2 fastener types'
            ],
            'negative': [
                'complex multi-part assemblies',
                'excessive fasteners',
                'fragile thin walls'
            ]
        },
        'DFM': {
            'positive': [
                'uniform wall thickness >= 1.5mm',
                'filleted edges',
                'draft angles for injection molding'
            ],
            'negative': [
                'sharp corners',
                'thin fragile walls < 1.5mm',
                'undercuts'
            ]
        },
        'DFS': {
            'positive': [
                'modular components for easy repair',
                'accessible fasteners',
                'standardized parts'
            ],
            'negative': [
                'permanently bonded components',
                'non-standard fasteners',
                'complex disassembly'
            ]
        },
        'DFSust': {
            'positive': [
                'recyclable materials like ABS or aluminum',
                'minimal material usage',
                'biodegradable packaging'
            ],
            'negative': [
                'non-recyclable composites',
                'excessive material waste',
                'single-use plastics'
            ]
        }
    }
    return rules.get(aspect, {'positive': [], 'negative': []})

class DesignPromptGenerator:
    def __init__(self):
        self.llm_model = "mistral"
        self.category_items = {
            "Consumer Electronics": ["smartphone", "laptop", "speaker", "tablet", "headphones", "earbuds", "smartwatch", "monitor", "keyboard", "controller", "computer mouse"],  # Added "computer mouse"
            "Furniture": ["chair", "desk", "shelf", "sofa", "table", "bookcase", "bed", "stool", "storage unit", "workstation"],
            "Transportation": ["scooter", "bicycle", "dashboard", "steering wheel", "helmet", "drone", "luggage", "backpack", "charging station"],
            "Kitchen Appliances": ["coffee maker", "blender", "toaster", "kettle", "food processor", "refrigerator", "microwave", "mixer", "juicer"],
            "Industrial Equipment": ["power tool", "control panel", "robotic arm", "3D printer", "monitoring device", "sensor", "factory equipment"],
            "Lighting": ["desk lamp", "floor lamp", "pendant light", "wall sconce", "track lighting", "outdoor light", "ceiling light"],
            "Wearables": ["fitness tracker", "smart glasses", "health monitor", "VR headset", "smart jewelry", "medical wearable", "safety equipment", "Clothes", "Headbands", "T-Shirts", "Jeans", "Hijab", "Pants", "Hats", "Earrings", "Jackets", "Ties", "Hoodies", "Shorts"]
        }
        self.focus_data = {
            "Form Factor": {"keyword": "form factor", "details": ["sleek contours", "geometric proportions", "compact design", "innovative shape", "distinctive silhouette"]},
            "Material Study": {"keyword": "material innovation", "details": ["sustainable materials", "novel material combinations", "textured surfaces", "high-performance composites", "recycled materials"]},
            "Ergonomics": {"keyword": "ergonomics", "details": ["user comfort", "intuitive controls", "accessibility features", "adaptive interfaces", "anthropometric design"]},
            "Sustainable Design": {"keyword": "sustainability", "details": ["energy efficiency", "recyclable components", "reduced environmental impact", "biodegradable elements", "cradle-to-cradle design"]},
            "Mechanism": {"keyword": "mechanical design", "details": ["folding mechanism", "adjustable components", "precision engineering", "modular assembly", "innovative joints"]},
            "Manufacturing": {"keyword": "manufacturing innovation", "details": ["3D-printed components", "injection molding", "CNC machining", "advanced fabrication", "efficient assembly"]}
        }
        self.design_styles = {
            "Minimalist": "clean lines and simplified form",
            "Brutalist": "bold, angular forms with exposed materials",
            "Organic": "flowing, nature-inspired shapes",
            "Futuristic": "forward-thinking with advanced technological elements",
            "Retro": "vintage-inspired with contemporary functionality",
            "Biomorphic": "biology-inspired fluid forms",
            "Scandinavian": "clean, functional design with natural elements",
            "Industrial": "utilitarian design with exposed mechanical elements",
            "Modular": "reconfigurable components with a systematic approach"
        }

    def generate_prompt(self, category, focus, style, user_input="", model_type="sd", fixed_item=None):
        logger.debug(f"Generating prompt: category={category}, focus={focus}, style={style}, user_input={user_input}, model_type={model_type}")
        try:
            # If fixed_item is provided, use it
            if fixed_item:
                item = fixed_item
            # If user_input is provided, try to find a matching category item
            elif user_input:
                # Check if user_input contains any predefined items
                matching_items = [item for item in self.category_items[category] if item in user_input.lower()]
                if matching_items:
                    item = matching_items[0]  # Use the first matching item
                else:
                    item = user_input  # Use the user_input directly if no match
            else:
                item = self.category_items[category][0]  # Default to first category item

            focus_details = self.focus_data[focus]
            style_desc = self.design_styles[style]
            base_prompt = f"A {style.lower()} {item} with {focus_details['keyword']}"
            detail = random.choice(focus_details['details'])
            base_prompt += f", {detail}, {style_desc}"

            system_prompt = f"""You are an expert industrial designer. Create a concise prompt for:
            - Item: {item}
            - Focus: {focus} ({focus_details['keyword']}: {detail})
            - Style: {style} ({style_desc})
            {'- User Input: ' + user_input if user_input else ''}
            Output a single, professional 2-line description for product visualization. Avoid unrealistic features or long lists."""
            
            response = ollama.generate(
                model=self.llm_model,
                prompt=system_prompt,
                options={"temperature": 0.7, "max_tokens": 55, "stop": ["</s>"]}
            )
            raw_prompt = response["response"]
            prompt = self._clean_prompt(raw_prompt, model_type)
            logger.debug(f"Generated prompt: {prompt}")
            return prompt
        except Exception as e:
            logger.error(f"Prompt generation failed: {str(e)}")
            st.error(f"Prompt generation failed: {str(e)}")
            return self._fallback_prompt(category, focus, style, model_type, fixed_item or user_input or self.category_items[category][0])

    def _clean_prompt(self, prompt, model_type):
        prompt = prompt.strip().replace('"', '').replace('\n', ', ')
        suffixes = {
            "sd": "professional product visualization, studio lighting, high detail",
            "sdxl": "8k resolution, product photography, CAD rendering, precise engineering details",
            "if": "technical blueprint, realistic materials, orthographic projection"
        }
        return f"{prompt}, {suffixes.get(model_type, '')}"

    def _fallback_prompt(self, category, focus, style, model_type, item):
        focus_details = self.focus_data[focus]
        style_desc = self.design_styles[style]
        return self._clean_prompt(f"A {style.lower()} {item} with {focus_details['keyword']}, {random.choice(focus_details['details'])}, {style_desc}", model_type)

def render_header():
    st.set_page_config(page_title="Ideate Industrial Product Design", page_icon="🔷", layout="wide")
    st.markdown("""
    <style>
    /* Import Inter font for Claude-like typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Reset default Streamlit styles and enforce dark theme */
    * {
        box-sizing: border-box;
        font-family: 'Inter', 'Roboto', sans-serif !important;
    }

    /* Main container and body */
    body, .main, .main .block-container, .stApp {
        background-color: #1A1A1A !important; /* Dark gray background */
        color: #E0E0E0 !important; /* Light gray text */
        max-width: 1000px;
        padding-top: 2rem;
        margin: 0 auto;
    }

    /* Headers */
    h1 {
        color: #FFFFFF !important; /* White for main title */
        text-align: center;
        font-weight: 600;
    }
    h2, h3, h4, h5, h6 {
        color: #E0E0E0 !important; /* Light gray for subheadings */
    }

    /* Buttons */
    .stButton>button {
        background-color: #4A90E2 !important; /* Claude-inspired blue */
        color: #FFFFFF !important;
        border-radius: 8px;
        border: none !important;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #357ABD !important; /* Darker blue on hover */
    }

    /* Cards */
    .card {
        border: 1px solid #333333 !important;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #2D2D2D !important; /* Slightly lighter dark */
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        margin-bottom: 1rem;
        color: #E0E0E0 !important;
    }

    /* Inspiration card */
    .inspiration-card {
        padding: 15px;
        border-radius: 8px;
        background-color: #2D2D2D !important;
        border-left: 4px solid #4A90E2 !important;
        margin-bottom: 15px;
        color: #E0E0E0 !important;
    }

    /* Download buttons */
    .download-btn {
        display: inline-block;
        padding: 8px 16px;
        background-color: #4A90E2 !important;
        color: #FFFFFF !important;
        text-decoration: none;
        border-radius: 6px;
        text-align: center;
        margin-top: 10px;
        font-weight: 500;
    }
    .download-btn:hover {
        background-color: #357ABD !important;
    }

    /* Model info */
    .model-info {
        font-size: 0.85rem;
        margin-top: 5px;
        color: #A0A0A0 !important; /* Muted gray for secondary text */
    }

    /* Input fields and selectboxes */
    .stSelectbox, .stTextInput, .stTextArea, .stSlider {
        background-color: #2D2D2D !important;
        color: #E0E0E0 !important;
        border-radius: 6px;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2D2D2D !important;
        border: 1px solid #444444 !important;
        color: #E0E0E0 !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #2D2D2D !important;
        color: #E0E0E0 !important;
        border: 1px solid #444444 !important;
    }
    .stSelectbox div[data-baseweb="select"] ul {
        background-color: #2D2D2D !important;
        color: #E0E0E0 !important;
    }
    .stSelectbox div[data-baseweb="select"] li:hover {
        background-color: #4A90E2 !important;
        color: #FFFFFF !important;
    }

    /* Slider */
    .stSlider [data-baseweb="slider"] > div > div {
        background-color: #4A90E2 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #4A90E2 !important;
    }

    /* Markdown and text */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: #E0E0E0 !important;
        background-color: transparent !important;
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        background-color: #1A1A1A !important;
    }
    .stTabs [role="tab"] {
        background-color: #2D2D2D !important;
        color: #E0E0E0 !important;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #4A90E2 !important;
        color: #FFFFFF !important;
    }

    /* Expander */
    .stExpander {
        background-color: #2D2D2D !important;
        border: 1px solid #444444 !important;
        border-radius: 6px;
    }
    .stExpander summary, .stExpander div {
        color: #E0E0E0 !important;
        background-color: #2D2D2D !important;
    }

    /* Alerts */
    .stAlert {
        background-color: #333333 !important;
        color: #E0E0E0 !important;
        border: 1px solid #444444 !important;
        border-radius: 6px;
    }
    .stSuccess {
        background-color: #2E7D32 !important; /* Green for success */
        color: #FFFFFF !important;
    }
    .stError {
        background-color: #D32F2F !important; /* Red for errors */
        color: #FFFFFF !important;
    }
    .stWarning {
        background-color: #FBC02D !important; /* Yellow for warnings */
        color: #000000 !important;
    }
    .stInfo {
        background-color: #0288D1 !important; /* Blue for info */
        color: #FFFFFF !important;
    }

    /* Override Streamlit's theme variables */
    :root {
        --background-color: #1A1A1A !important;
        --text-color: #E0E0E0 !important;
        --primary-color: #4A90E2 !important;
        --secondary-background-color: #2D2D2D !important;
    }

    /* Disable Streamlit's theme toggle */
    .stThemeToggle, [data-testid="stThemeToggle"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    st.title("Ideate Industrial Product Design")
    st.markdown("Generate concept visualizations to inspire your industrial design process")

def render_design_inspiration_section(prompt_generator, model_type, tab_prefix="tab1"):
    st.markdown("## Design Concept Generator")
    col1, col2 = st.columns([3, 2])
    with col1:
        category = st.selectbox(
            "Product Category", 
            list(prompt_generator.category_items.keys()),
            key=f"{tab_prefix}_category_selectbox"
        )
        focus = st.selectbox(
            "Design Focus", 
            list(prompt_generator.focus_data.keys()),
            key=f"{tab_prefix}_focus_selectbox"
        )
        style = st.selectbox(
            "Design Style", 
            list(prompt_generator.design_styles.keys()),
            key=f"{tab_prefix}_style_selectbox"
        )
        user_input = st.text_input(
            "Your Custom Description (optional)",
            placeholder="e.g., 'with a sleek curved edge and LED display'",
            key=f"{tab_prefix}_user_input"
        )
    with col2:
        st.markdown("#### Design Focus Description")
        if focus:
            focus_details = ", ".join(prompt_generator.focus_data[focus]["details"][:3])
            st.markdown(f"*{focus_details}*")
        st.markdown("#### Style Characteristics")
        if style:
            st.markdown(f"*{prompt_generator.design_styles[style]}*")
    
    if st.button("🔍 Generate Design Brief", use_container_width=True, key=f"{tab_prefix}_generate_brief"):
        with st.spinner("Creating design brief..."):
            time.sleep(0.5)
            design_prompt = prompt_generator.generate_prompt(category, focus, style, user_input, model_type)
            st.session_state.inspiration = design_prompt
            logger.debug(f"Generated design brief: {design_prompt}")
    
    if "inspiration" in st.session_state:
        st.markdown(f"<div class='inspiration-card'>{st.session_state.inspiration}</div>", unsafe_allow_html=True)
        if st.button("Use this design brief", use_container_width=True, key=f"{tab_prefix}_use_brief"):
            st.session_state.user_prompt = st.session_state.inspiration
    
    st.session_state[f"{tab_prefix}_category"] = category
    st.session_state[f"{tab_prefix}_focus"] = focus
    st.session_state[f"{tab_prefix}_style"] = style

def render_design_input_section():
    st.markdown("## Design Brief")
    mode = st.selectbox("Generation Mode", ["Text → Image", "Sketch → Image"], key="mode_selectbox")
    st.session_state.mode = mode
    default_prompt = st.session_state.get("user_prompt", "")
    prompt = st.text_area(
        "Enter your design concept description:", 
        value=default_prompt,
        height=100,
        placeholder="Example: A minimalist desk lamp with adjustable arm, made of brushed aluminum and wood, product visualization with white background"
    )
    sketch_file = None
    if mode == "Sketch → Image":
        sketch_file = st.file_uploader("Upload Sketch", type=["png", "jpg", "jpeg"], key="sketch_uploader")
        if sketch_file:
            st.session_state.sketch_file = sketch_file
    return prompt, mode, sketch_file

def render_design_specs_section(model_type, tab_prefix="tab1"):
    st.markdown("## Visualization Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        viewpoint = st.selectbox(
            "Viewpoint", 
            ["three-quarter view", "front view", "side view", "top-down view", 
             "isometric view", "exploded view", "cross-section view"],
            key=f"{tab_prefix}_viewpoint_selectbox"
        )
    with col2:
        render_style = st.selectbox(
            "Rendering Style", 
            ["product photography", "technical illustration", "sketch", 
             "CAD rendering", "wireframe", "concept art", "blueprint"],
            key=f"{tab_prefix}_render_style_selectbox"
        )
    with col3:
        add_suffix = st.checkbox(
            "Add technical render suffixes", 
            value=True,
            key=f"{tab_prefix}_add_suffix_checkbox"
        )
    
    # Integrated from projet-dfx.py: DfX aspect selection
    aspect = st.selectbox(
        "DfX Aspect",
        ["DFA", "DFM", "DFS", "DFSust"],
        key=f"{tab_prefix}_aspect_selectbox"
    )
    
    if model_type == "sdxl":
        height_options = [512, 640, 768, 1024]
        width_options = [512, 640, 768, 1024]
        default_height = 768
        default_width = 768
    elif model_type == "if":
        height_options = [256, 384, 512, 640]
        width_options = [256, 384, 512, 640]
        default_height = 512
        default_width = 512
    else:
        height_options = [256, 384, 512, 576, 640, 768]
        width_options = [256, 384, 512, 576, 640, 768]
        default_height = 512
        default_width = 512
    
    col1, col2 = st.columns(2)
    with col1:
        height = st.select_slider(
            "Image Height", 
            options=height_options, 
            value=default_height,
            key=f"{tab_prefix}_height_slider"
        )
    with col2:
        width = st.select_slider(
            "Image Width", 
            options=width_options, 
            value=default_width,
            key=f"{tab_prefix}_width_slider"
        )
    
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            negative_prompt = st.text_area(
                "Negative Prompt (elements to avoid):", 
                value="low quality, distorted proportions, unrealistic materials, blurry, watermark, asymmetrical",
                height=100,
                key=f"{tab_prefix}_negative_prompt"
            )
        with col2:
            if model_type == "sdxl":
                num_inference_steps_min = 25
                num_inference_steps_max = 100
                num_inference_steps_default = 50
            elif model_type == "if":
                num_inference_steps_min = 20
                num_inference_steps_max = 100
                num_inference_steps_default = 50
            else:
                num_inference_steps_min = 20
                num_inference_steps_max = 100
                num_inference_steps_default = 50
            
            num_inference_steps = st.slider(
                "Quality", 
                min_value=num_inference_steps_min, 
                max_value=num_inference_steps_max, 
                value=num_inference_steps_default,
                help="Higher values produce better quality but take longer",
                key=f"{tab_prefix}_quality_slider"
            )
            use_low_memory = st.checkbox(
                "Low Memory Mode", 
                value=False,
                help="Slower but uses less VRAM",
                key=f"{tab_prefix}_low_memory_checkbox"
            )
            if model_type != "if":
                guidance_scale = st.slider(
                    "Guidance Scale",
                    min_value=5.0,
                    max_value=15.0,
                    value=7.5,
                    step=0.5,
                    help="How closely to follow the prompt (higher = more literal)",
                    key=f"{tab_prefix}_guidance_scale_slider"
                )
            else:
                guidance_scale = 7.5
            use_random_seed = st.checkbox(
                "Use random seed", 
                value=True,
                key=f"{tab_prefix}_random_seed_checkbox"
            )
            if not use_random_seed:
                seed = st.number_input(
                    "Seed", 
                    value=42, 
                    min_value=0, 
                    max_value=2147483647,
                    key=f"{tab_prefix}_seed_input"
                )
            else:
                seed = random.randint(0, 2147483647)
            if st.session_state.get("mode") == "Sketch → Image":
                threshold = st.slider(
                    "Edge Threshold",
                    min_value=0,
                    max_value=255,
                    value=100,
                    help="Controls edge detection sensitivity for sketch",
                    key=f"{tab_prefix}_threshold_slider"
                )
                blur_radius = st.slider(
                    "Blur Radius",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    help="Controls blur applied to sketch before edge detection",
                    key=f"{tab_prefix}_blur_radius_slider"
                )
                controlnet_scale = st.slider(
                    "ControlNet Scale",
                    min_value=0.5,
                    max_value=1.0,
                    value=0.8,
                    help="Strength of ControlNet influence",
                    key=f"{tab_prefix}_controlnet_scale_slider"
                )
            else:
                threshold = 100
                blur_radius = 0.5
                controlnet_scale = 0.8
    
    return {
        "viewpoint": viewpoint,
        "render_style": render_style,
        "add_suffix": add_suffix,
        "height": height,
        "width": width,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "use_low_memory": use_low_memory,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "use_random_seed": use_random_seed,
        "threshold": threshold,
        "blur_radius": blur_radius,
        "controlnet_scale": controlnet_scale,
        "aspect": aspect
    }

def generate_image_optimized(prompt, model_id, specifications, model_type, mode, sketch_file=None):
    logger.debug(f"Starting image generation: prompt={prompt[:50]}..., model_id={model_id}, mode={mode}, sketch_file={sketch_file is not None}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start_time = time.time()
    
    with track_memory_usage(show_stats=True):
        try:
            pipelines = load_model(model_id, device, specifications["use_low_memory"], model_type)
            if pipelines is None:
                logger.error("Model loading failed, pipelines is None")
                return None, None
            
            pipe = pipelines["base"]
            pipe_cn = pipelines.get("controlnet")
            
            generator = None
            if not specifications["use_random_seed"]:
                generator = torch.Generator(device=device).manual_seed(specifications["seed"])
            
            progress_bar = st.progress(0)
            common_params = {
                "prompt": prompt,
                "negative_prompt": specifications["negative_prompt"],
                "num_inference_steps": specifications["num_inference_steps"],
                "guidance_scale": specifications["guidance_scale"],
                "generator": generator
            }
            
            def progress_callback(step, timestep, latents):
                progress = (step + 1) / specifications["num_inference_steps"]
                progress_bar.progress(min(int(progress * 100), 100))
            
            common_params["callback"] = progress_callback
            common_params["callback_steps"] = 1
            
            if mode == "Sketch → Image":
                if not sketch_file:
                    logger.error("No sketch file provided in Sketch → Image mode")
                    st.error("Please upload a sketch for Sketch → Image mode.")
                    return None, None
                if model_type == "sdxl" or model_type == "if":
                    logger.error(f"Sketch → Image mode not supported for {model_type}")
                    st.error("Sketch → Image mode is only supported for Stable Diffusion 1.5, 2.1, or OpenJourney.")
                    return None, None
                if not pipe_cn:
                    logger.error("ControlNet pipeline not available")
                    st.error("ControlNet pipeline not available for this model.")
                    return None, None
                
                mask = process_sketch(sketch_file, (specifications["height"], specifications["width"]), 
                                    specifications["threshold"], specifications["blur_radius"])
                if mask is None:
                    logger.error("Sketch processing failed, mask is None")
                    return None, None
                
                image = pipe_cn(
                    **common_params,
                    image=[mask],
                    height=specifications["height"],
                    width=specifications["width"],
                    controlnet_conditioning_scale=specifications["controlnet_scale"]
                ).images[0]
            
            else:
                if model_type == "sdxl":
                    image = pipe(
                        **common_params,
                        height=specifications["height"],
                        width=specifications["width"],
                    ).images[0]
                elif model_type == "if":
                    common_params.pop("guidance_scale", None)
                    image = pipe(
                        **common_params,
                        height=specifications["height"],
                        width=specifications["width"],
                    ).images[0]
                else:
                    image = pipe(
                        **common_params,
                        height=specifications["height"],
                        width=specifications["width"],
                    ).images[0]
            
            progress_bar.progress(100)
            elapsed = time.time() - start_time
            st.success(f"Image generated in {elapsed:.2f} seconds")
            st.info(f"Seed used: {specifications['seed']}")
            logger.debug(f"Image generated successfully, seed: {specifications['seed']}")
            return image, specifications["seed"]
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory")
                st.error("💡 GPU out of memory. Try enabling Low Memory Mode or reducing image size.")
                st.info("Tip: Close other applications using GPU to free up memory.")
            else:
                logger.error(f"Runtime error during image generation: {str(e)}")
                st.error(f"Error generating image: {str(e)}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error during image generation: {str(e)}")
            st.error(f"Unexpected error: {str(e)}")
            return None, None
        finally:
            if 'progress_bar' in locals():
                progress_bar.empty()

def display_image_with_download(image, prompt, seed, aspect):
    st.markdown("## Design Visualization")
    st.image(image, use_column_width=False)
    rules = get_dfx_rules(aspect)
    st.markdown(f"**{aspect} Guidelines**:\n- **Positive**: {', '.join(rules['positive'])}\n- **Negative (avoid)**: {', '.join(rules['negative'])}")
    with st.expander("Image Details"):
        st.markdown(f"**Prompt:** {prompt}")
        st.markdown(f"**Seed:** {seed}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{img_str}" download="design_concept.png" class="download-btn">Download PNG</a>'
        st.markdown(href, unsafe_allow_html=True)
    with col2:
        buffered_jpg = io.BytesIO()
        rgb_image = image.convert('RGB')
        rgb_image.save(buffered_jpg, format="JPEG", quality=95)
        jpg_str = base64.b64encode(buffered_jpg.getvalue()).decode()
        href_jpg = f'<a href="data:file/jpeg;base64,{jpg_str}" download="design_concept.jpg" class="download-btn">Download JPEG</a>'
        st.markdown(href_jpg, unsafe_allow_html=True)
    with col3:
        buffered_webp = io.BytesIO()
        rgb_image.save(buffered_webp, format="WebP", quality=95)
        webp_str = base64.b64encode(buffered_webp.getvalue()).decode()
        href_webp = f'<a href="data:file/webp;base64,{webp_str}" download="design_concept.webp" class="download-btn">Download WebP</a>'
        st.markdown(href_webp, unsafe_allow_html=True)
    
    # Add Generate DfX Report button
    st.subheader("Generate DfX Report")
    if st.button("Generate Report"):
        category = st.session_state.get("tab1_category", list(DesignPromptGenerator().category_items.keys())[0])
        form = st.session_state.get("tab1_style", list(DesignPromptGenerator().design_styles.keys())[0]).lower()
        user_text = prompt  # Use the prompt as the user_text
        description = generate_image_description(image, aspect, user_text, category, form)
        mistral_prompt = (
            f"Analyze the following design for a {category} product, specifically {user_text}, "
            f"with a {form} form, optimized for {aspect} (Design for {aspect}). "
            f"Description: {description}. "
            f"Positive guidelines: {', '.join(rules['positive'])}. "
            f"Negative guidelines to avoid: {', '.join(rules['negative'])}. "
            f"Provide a detailed comparison between the original concept and the final design. "
            f"Include specific changes related to {aspect} guidelines, suggested materials (e.g., ABS plastic, aluminum, silicone for straps), "
            f"and how the design aligns with the {category} category and {form} form. "
            f"Describe visual elements like texture, color, and form."
        )
        st.session_state.report = generate_text_with_mistral(mistral_prompt, max_tokens=512)
        st.session_state.report_generated = True
    
    # Display report if generated
    if st.session_state.report_generated and st.session_state.report:
        st.subheader("DfX Report")
        st.markdown(st.session_state.report)
def iterative_prompt_refinement(prompt_generator, category, focus, style, user_input, model_id, model_type, specifications, max_iterations=3):
    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    results = []
    fixed_item = next((item for item in prompt_generator.category_items[category] if item in user_input.lower()), None) or prompt_generator.category_items[category][0]
    
    current_focus = focus
    current_style = style
    current_user_input = user_input
    last_score = 0.5  # Initial score for prompt refinement logic

    for iteration in range(max_iterations):
        st.write(f"Iteration {iteration + 1}/{max_iterations}")
        current_prompt = prompt_generator.generate_prompt(
            category, 
            current_focus, 
            current_style, 
            current_user_input, 
            model_type, 
            fixed_item
        )
        image, seed = generate_image_optimized(
            current_prompt, 
            model_id, 
            specifications,
            model_type,
            mode="Text → Image"
        )
        if image is None:
            st.error(f"Iteration {iteration + 1} failed to generate an image.")
            continue
        st.image(image, caption=f"Image {iteration + 1}", use_column_width=True)
        rules = get_dfx_rules(specifications["aspect"])
        st.markdown(f"**{specifications['aspect']} Guidelines**:\n- **Positive**: {', '.join(rules['positive'])}\n- **Negative (avoid)**: {', '.join(rules['negative'])}")
        score = st.number_input(
            f"Score for Image {iteration + 1} (0 to 1, based on {specifications['aspect']} adherence)",
            min_value=0.0,
            max_value=1.0,
            value=last_score,
            step=0.01,
            key=f"score_{iteration}"
        )
        result = {
            "prompt": current_prompt,
            "image": image,
            "seed": seed,
            "score": score,
            "iteration": iteration + 1,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_id
        }
        results.append(result)
        st.session_state.generated_images.append(result)
        st.markdown(f"### Iteration {iteration + 1}")
        st.write(f"Prompt: {current_prompt}")
        st.write(f"Score: {score:.2f}")
        if iteration < max_iterations - 1:
            if score < 0.5:
                available_foci = [f for f in prompt_generator.focus_data.keys() if f != current_focus]
                current_focus = random.choice(available_foci) if available_foci else current_focus
                st.write(f"Score low: Switching focus to {current_focus}")
            elif score < 0.7:
                available_styles = [s for s in prompt_generator.design_styles.keys() if s != current_style]
                if available_styles and random.random() > 0.5:
                    current_style = random.choice(available_styles)
                    st.write(f"Score moderate: Switching style to {current_style}")
                else:
                    current_user_input += ", enhanced detail, sharper edges"
                    st.write("Score moderate: Enhancing detail in user input")
        last_score = score
    return results

def models():
    return {
        "Stable Diffusion 1.5": {
            "model_id": "runwayml/stable-diffusion-v1-5",
            "type": "sd",
            "description": "Classic model with good balance of quality and speed, supports sketch-based generation"
        },
        "Stable Diffusion 2.1": {
            "model_id": "stabilityai/stable-diffusion-2-1-base", 
            "type": "sd",
            "description": "Improved version with better detail consistency, supports sketch-based generation"
        },
        "Stable Diffusion XL": {
            "model_id": "stabilityai/stable-diffusion-xl-base-1.0", 
            "type": "sdxl",
            "description": "High-quality model with exceptional detail and composition"
        },
        "SDXL Turbo": {
            "model_id": "stabilityai/sdxl-turbo",
            "type": "sdxl",
            "description": "Ultra-fast generative model with near real-time image generation"
        },
        "OpenJourney v4": {
            "model_id": "prompthero/openjourney-v4", 
            "type": "sd",
            "description": "Midjourney-like aesthetic specialized for product design, supports sketch-based generation"
        }
    }

def main():

    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Text → Image"
    if "sketch_file" not in st.session_state:
        st.session_state.sketch_file = None
    if "generation_state" not in st.session_state:
        st.session_state.generation_state = "idle"
    if "generate_triggered" not in st.session_state:
        st.session_state.generate_triggered = False
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = ""
    if "last_model" not in st.session_state:
        st.session_state.last_model = ""
    # Add report-related session state
    if "report_generated" not in st.session_state:
        st.session_state.report_generated = False
    if "report" not in st.session_state:
        st.session_state.report = ""
    
    render_header()
    prompt_generator = DesignPromptGenerator()
    
    col_models, col_status = st.columns([4, 1])
    with col_models:
        model_options = models()
        if "previous_model" not in st.session_state:
            st.session_state.previous_model = None
        selected_model = st.selectbox("Select Model Engine", list(model_options.keys()))
        if st.session_state.previous_model != selected_model:
            clear_previous_model()
            st.session_state.previous_model = selected_model
        st.markdown(f"<div class='model-info'>{model_options[selected_model]['description']}</div>", unsafe_allow_html=True)
    
    with col_status:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"GPU: {gpu_name}")
            try:
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.info(f"VRAM: {vram:.1f} GB")
            except:
                pass
        else:
            st.warning("Using CPU (slower)")
    
    model_type = model_options[selected_model]["type"]
    
    # Reset generation state if prompt or model changes
    current_prompt = st.session_state.get("user_prompt", "")
    if current_prompt != st.session_state.last_prompt or selected_model != st.session_state.last_model:
        st.session_state.generation_state = "idle"
        st.session_state.generate_triggered = False
        st.session_state.last_prompt = current_prompt
        st.session_state.last_model = selected_model
        logger.debug("Reset generation state due to prompt or model change")
    
    tab1, tab4 = st.tabs(["Create New Design", "Iterative Design"])
    
    with tab1:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_design_inspiration_section(prompt_generator, model_type, tab_prefix="tab1")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            prompt, mode, sketch_file = render_design_input_section()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            specifications = render_design_specs_section(model_type, tab_prefix="tab1")
            st.markdown('</div>', unsafe_allow_html=True)
        
        
        if st.button("🔷 Generate Design Visualization", use_container_width=True, type="primary"):
            logger.debug("Generate button clicked")
            st.session_state.generate_triggered = True
            st.session_state.generation_state = "generating"
        
        if st.session_state.generate_triggered and st.session_state.generation_state == "generating":
            logger.debug("Starting generation process")
            if not prompt:
                st.error("Please enter a design brief.")
                st.session_state.generate_triggered = False
                st.session_state.generation_state = "idle"
            elif mode == "Sketch → Image" and not sketch_file:
                st.error("Please upload a sketch for Sketch → Image mode.")
                st.session_state.generate_triggered = False
                st.session_state.generation_state = "idle"
            else:
                with st.spinner("Generating design visualization..."):
                    final_prompt = prompt
                    if specifications["add_suffix"]:
                        if specifications["viewpoint"] and specifications["render_style"]:
                            final_prompt += f", {specifications['viewpoint']}, {specifications['render_style']}"
                        if model_type == "sdxl":
                            final_prompt += ", professional industrial design, high detail, product visualization, 8k, studio lighting, product photography"
                        elif model_type == "if":
                            final_prompt += ", highly detailed, realistic, professionally made"
                        else:
                            final_prompt += ", professional industrial design, high detail, product visualization"
                    
                    logger.debug(f"Final prompt: {final_prompt[:50]}...")
                    image, seed = generate_image_optimized(
                        final_prompt,
                        model_options[selected_model]["model_id"],
                        specifications,
                        model_type,
                        mode,
                        sketch_file
                    )
                    
                    if image:
                        st.session_state.current_image = {
                            "image": image,
                            "prompt": final_prompt,
                            "seed": seed,
                            "model": selected_model,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "accepted": False,
                            "score": 0.5
                        }
                        st.image(image, caption="Generated Design", use_column_width=False)
                        st.info(f"Seed used: {seed}")
                        rules = get_dfx_rules(specifications["aspect"])
                        st.markdown(f"**{specifications['aspect']} Guidelines**:\n- **Positive**: {', '.join(rules['positive'])}\n- **Negative (avoid)**: {', '.join(rules['negative'])}")
                        score = st.number_input(
                            f"Score for Generated Image (0 to 1, based on {specifications['aspect']} adherence)",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.5,
                            step=0.01,
                            key="score_current"
                        )
                        st.session_state.current_image["score"] = score
                        
                        col_accept, col_not_accept = st.columns(2)
                        with col_accept:
                            if st.button("Accept", key="accept_image"):
                                current = st.session_state.current_image
                                current["accepted"] = True
                                st.session_state.generated_images.append(current)
                                st.success(f"Design added! Generated images size: {len(st.session_state.generated_images)}")
                                st.session_state.accepted_image = current
                                del st.session_state.current_image
                                st.session_state.generate_triggered = False
                                st.session_state.generation_state = "idle"
                        
                        with col_not_accept:
                            if st.button("Not Accept", key="not_accept_image"):
                                with st.spinner("Re-generating design..."):
                                    if specifications["use_random_seed"]:
                                        specifications["seed"] = random.randint(0, 2147483647)
                                    new_image, new_seed = generate_image_optimized(
                                        final_prompt,
                                        model_options[selected_model]["model_id"],
                                        specifications,
                                        model_type,
                                        mode,
                                        sketch_file
                                    )
                                    if new_image:
                                        st.session_state.current_image = {
                                            "image": new_image,
                                            "prompt": final_prompt,
                                            "seed": new_seed,
                                            "model": selected_model,
                                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                            "accepted": False,
                                            "score": score
                                        }
                                        st.success("New design generated!")
                                    st.session_state.generate_triggered = False
                                    st.session_state.generation_state = "idle"
                    else:
                        st.session_state.generate_triggered = False
                        st.session_state.generation_state = "idle"
        
        if "accepted_image" in st.session_state:
            display_image_with_download(
                st.session_state.accepted_image["image"],
                st.session_state.accepted_image["prompt"],
                st.session_state.accepted_image["seed"],
                specifications["aspect"]
            )
            if st.button("Generate New Image", key="new_after_accept"):
                del st.session_state.accepted_image
                st.session_state.generate_triggered = False
                st.session_state.generation_state = "idle"
    
    with tab4:
        st.markdown("## Iterative Design Process")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            render_design_inspiration_section(prompt_generator, model_type, tab_prefix="tab4")
            st.markdown('</div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            specifications = render_design_specs_section(model_type, tab_prefix="tab4")
            st.markdown('</div>', unsafe_allow_html=True)
        if st.button("🔍 Start Iterative Design", use_container_width=True):
            with st.spinner("Running iterative refinement..."):
                category = st.session_state.get("tab4_category", list(prompt_generator.category_items.keys())[0])
                focus = st.session_state.get("tab4_focus", list(prompt_generator.focus_data.keys())[0])
                style = st.session_state.get("tab4_style", list(prompt_generator.design_styles.keys())[0])
                user_input = st.session_state.get("tab4_user_input", "")
                results = iterative_prompt_refinement(
                    prompt_generator=prompt_generator,
                    category=category,
                    focus=focus,
                    style=style,
                    user_input=user_input,
                    model_id=model_options[selected_model]["model_id"],
                    model_type=model_type,
                    specifications=specifications,
                    max_iterations=4
                )
                if results:
                    best_result = max(results, key=lambda x: x["score"])
                    st.markdown("## Best Result")
                    st.image(best_result["image"])
                    st.write(f"Final Prompt: {best_result['prompt']}")
                    st.write(f"Score: {best_result['score']:.2f}")
                    display_image_with_download(
                        best_result["image"],
                        best_result["prompt"],
                        best_result["seed"],
                        specifications["aspect"]
                    )

if __name__ == "__main__":
    main()