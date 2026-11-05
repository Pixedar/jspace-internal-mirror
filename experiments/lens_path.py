"""Resolve a lens spec: 'hf:qwen3-8b' -> local path of Neuronpedia's Jacobian lens (downloads it once)."""
import sys

NAMES = {'qwen3-1.7b': 'Qwen3-1.7B', 'qwen3-4b': 'Qwen3-4B', 'qwen3-8b': 'Qwen3-8B', 'qwen3-14b': 'Qwen3-14B',
         'qwen3-32b': 'Qwen3-32B'}


def lens_path(spec):
    if not spec or not str(spec).startswith('hf:'):
        return spec
    from huggingface_hub import hf_hub_download
    tag = spec[3:]
    return hf_hub_download('neuronpedia/jacobian-lens', f'{tag}/jlens/Salesforce-wikitext/{NAMES[tag]}_jacobian_lens.pt')


if __name__ == '__main__':
    print(lens_path(sys.argv[1]))
