import torch
#from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

default_chat_config = ModelChatConfig(
    max_session_length=8192,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.6, top_p=0.9),
)

MODEL_FAMILIES = {
    "Llama 2": [
        ModelConfig(
            ModelBackendConfig(repository="meta-llama/Llama-2-70b-chat-hf"),
            ModelFrontendConfig(
                name="Llama 2 (70B-Chat)",
                model_card="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
                license="https://bit.ly/llama2-license",
            ),
            default_chat_config,
        ),
    ],
}

#INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
INITIAL_PEERS = [
    "/ip4/86.48.21.229/tcp/31337/p2p/12D3KooWPyRUNtfz4f9RiPwhT99GVLvfVAALMXZ27AV1DubSc6oR",
    # '/ip4/86.48.21.229/udp/31337/quic/p2p/12D3KooWJm4rZZDqwKU66wN7fzHk1aDbvkRjiDdxusQEPcHoH3X4',
    "/ip4/81.0.246.28/tcp/31337/p2p/12D3KooWA6QWj4eP5D9xfsuAfarhDJX1nGhHasmFYJEVkjhSydyV",
    # '/ip4/81.0.246.28/udp/31337/quic/p2p/12D3KooWFRqPYQ4o46F5dRQMshT24ypEzs8Qz5RMGHbHqHLT3J67'
]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
