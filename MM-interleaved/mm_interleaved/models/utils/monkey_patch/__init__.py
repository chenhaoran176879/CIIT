from .llama_flash_attn_train_monkey_patch import replace_llama_attn_with_flash_attn
from .blip2_qknorm_monkey_patch import replace_blip2_attn_with_qknorm_attn
from .beam_search_monkey_patch import replace_beam_search
from .sd_pipeline_monkey_patch import replace_stable_diffusion_pipeline_call
from .sd_unet_forward_monkey_patch import replace_stable_diffusion_unet_forward