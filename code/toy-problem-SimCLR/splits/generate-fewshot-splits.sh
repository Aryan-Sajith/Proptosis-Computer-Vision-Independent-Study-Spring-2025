python - << 'EOF'
from src.datasets import make_fewshot_splits
make_fewshot_splits(
    root_dir='data/UTKFace',
    out_file='splits/fewshot_splits.json',
    seed=42,
    k_per_class=100,
    age_bins=(29, 30),  
    val_per_class=10,
    test_per_class=10
)
EOF