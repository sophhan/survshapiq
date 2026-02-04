"""
Utility functions and classes for TCGA BRCA Survival Analysis with MIL.

This module provides:
- MILSurvivalModel: Attention-based MIL model for survival prediction
- TCGAFeatureDataset: Dataset for loading precomputed UNI2 features
- create_feature_dataloaders: DataLoader creation with collation for variable-length batches
- compute_concordance_index: C-index metric for survival analysis
"""

import torch
import torch.nn as nn
import torchtuples as tt
from pycox.evaluation import EvalSurv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image


# ============================================================================
# Model Architecture
# ============================================================================

class MILSurvivalModel(nn.Module):
    """
    Multiple Instance Learning Survival Model with Gated Attention.

    Uses gated attention mechanism from Ilse et al. (2018):
    "Attention-based Deep Multiple Instance Learning"

    Attention formula: a = softmax(w^T * (tanh(V*h) ⊙ sigm(U*h)))
    where ⊙ is element-wise multiplication (gating mechanism).

    Takes variable number of patch features per patient and aggregates them
    using gated attention before survival prediction.

    Args:
        feature_dim (int): Dimension of patch features (1536 for UNI2-h)
        num_durations (int): Number of discrete time bins
        tabular_dim (int): Dimension of tabular features (0 to disable)
        hidden_dim (int): Hidden dimension for survival head
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        feature_dim=1536,
        num_durations=20,
        tabular_dim=0,
        hidden_dim=512,
        attention_dim=256,
        dropout=0.3
    ):
        super().__init__()

        self.use_tabular = tabular_dim > 0
        self.attention_dim = attention_dim

        # Dropout before attention for regularization
        self.feature_dropout = nn.Dropout(dropout * 0.5)

        # Gated Attention mechanism (Ilse et al., 2018, "Attention-based Deep Multiple Instance Learning")
        # a = softmax(w^T * (tanh(V*h) ⊙ sigm(U*h)))
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(attention_dim, 1)

        self.post_attention = nn.Sequential(
            #nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Tabular network (optional) - with feature-wise gating
        if self.use_tabular:

            # Project weighted features
            self.tab_net = nn.Sequential(
                nn.LayerNorm(tabular_dim),
                nn.Linear(tabular_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 64),
                nn.GELU()
            )

            head_input = 64 + 64 * 2  # aggregated + tabular + interactions
        else:
            head_input = 64

        # Survival head (simple, no BatchNorm for small datasets)
        self.head = nn.Sequential(
            nn.Linear(head_input, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_durations)
        )

    def forward(self, patch_features, tabular=None, return_attention=False):
        """
        Args:
            patch_features: [batch_size, num_patches, feature_dim]
            tabular: [batch_size, tabular_dim] (optional)
            return_attention: Return attention weights for interpretability

        Returns:
            hazards: [batch_size, num_durations]
            attention_weights: [batch_size, num_patches] (if return_attention=True)
        """
        # Apply dropout to patch features for regularization
        patch_features = self.feature_dropout(patch_features)

        # Compute gated attention weights
        # a = softmax(w^T * (tanh(V*h) ⊙ sigm(U*h)))
        A_V = self.attention_V(patch_features)  # [B, N, attention_dim]
        A_U = self.attention_U(patch_features)  # [B, N, attention_dim]
        A = self.attention_w(A_V * A_U)  # [B, N, 1] - element-wise gating

        attn_weights = torch.softmax(A, dim=1)  # [B, N, 1]

        # Aggregate features
        aggregated = torch.sum(attn_weights * patch_features, dim=1)  # [B, feature_dim]

        # Post-attention processing
        aggregated = self.post_attention(aggregated)  # [B, 64]

        # Add tabular features with feature-wise gating
        if self.use_tabular and tabular is not None:

            # Project weighted features
            tab_feat = self.tab_net(tabular)  # [B, 646 ]

            # Multiply aggregated features with tabular features (feature-wise gating)
            interactions = aggregated * tab_feat  # [B, 64]
            aggregated = torch.cat([aggregated, tab_feat, interactions], dim=1)  # [B, 64 + 64 + 64]

        # Predict hazards
        hazards = self.head(aggregated)

        if return_attention:
            return hazards, attn_weights.squeeze(-1)
        return hazards


# ============================================================================
# Dataset and DataLoader
# ============================================================================

class TCGAFeatureDataset(Dataset):
    """
    Dataset for TCGA BRCA using precomputed UNI2 features.

    Args:
        df (pd.DataFrame): Clinical data with 'patient_id', 'time', 'event', 'split'
        feature_dir (str/Path): Directory containing .npy feature files
        img_dir (str/Path, optional): Directory with original images (for interpretability)
        split (str): 'train', 'val', or 'test'
        load_images (bool): If True, also loads original images

    Returns:
        If load_images=False:
            (patch_features, tabular), (time, event), patient_id
        If load_images=True:
            (patch_features, tabular, images), (time, event), patient_id
    """

    def __init__(self, df, feature_dir, img_dir=None, split='train', load_images=False, max_patches=None):
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.feature_dir = Path(feature_dir)
        self.img_dir = Path(img_dir) if img_dir else None
        self.load_images = load_images
        self.max_patches = max_patches

        # Get tabular feature columns (exclude metadata and label columns)
        base_cols = ['patient_id', 'time', 'event', 'split', 'idx_duration', 'event_transformed']
        self.feature_cols = [c for c in df.columns if c not in base_cols]

        # Filter patients with features
        self.samples = []
        for idx, row in self.df.iterrows():
            patient_id = row['patient_id']
            feature_path = self.feature_dir / f"{patient_id}.npy"

            if feature_path.exists():
                self.samples.append({'patient_id': patient_id, 'df_idx': idx})

        print(f"{split.upper()}: {len(self.samples)} patients with features")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        patient_id = sample['patient_id']

        # Load features
        features = np.load(self.feature_dir / f"{patient_id}.npy")

        # Random sampling if max_patches is set and we have more patches
        if self.max_patches is not None and features.shape[0] > self.max_patches:
            # Random sample max_patches indices
            indices = np.random.choice(features.shape[0], self.max_patches, replace=False)
            features = features[indices]
        if self.max_patches is not None and features.shape[0] < self.max_patches:
            # Sample with replacement to reach max_patches
            indices = np.random.choice(features.shape[0], self.max_patches - features.shape[0], replace=True)
            features = np.concatenate([features, features[indices]], axis=0)

        features = torch.tensor(features, dtype=torch.float32)

        # Get clinical data
        row = self.df.iloc[sample['df_idx']]
        tabular = torch.tensor([row[c] for c in self.feature_cols], dtype=torch.float32)

        # Get transformed labels (discrete time bins for pycox)
        idx_duration = torch.tensor(row['idx_duration'], dtype=torch.long)
        event_transformed = torch.tensor(row['event_transformed'], dtype=torch.float32)

        # Optionally load images
        if self.load_images and self.img_dir:
            patient_folder = self.img_dir / patient_id
            image_files = sorted(list(patient_folder.glob("*.jpg")))
            images = [Image.open(img).convert('RGB') for img in image_files]
            return (features, tabular, images), (idx_duration, event_transformed), patient_id

        return (features, tabular), (idx_duration, event_transformed), patient_id


def collate_mil_batch(batch):
    """
    Collate function for batching patients with variable number of patches.
    Pads features to the same length so pycox can handle them.

    Args:
        batch: List of samples from TCGAFeatureDataset

    Returns:
        tuple: ((padded_features, tabular), (times, events))
            - padded_features: Tensor [batch_size, max_patches, feature_dim]
            - tabular: Tensor [batch_size, tabular_dim]
            - times: Tensor [batch_size]
            - events: Tensor [batch_size]
    """
    features_list = []
    tabular_list = []
    times_list = []
    events_list = []

    for item in batch:
        (features, tabular), (time, event), patient_id = item
        features_list.append(features)
        tabular_list.append(tabular)
        times_list.append(time)
        events_list.append(event)

    return (torch.stack(features_list), torch.stack(tabular_list)), (torch.stack(times_list), torch.stack(events_list))



def create_feature_dataloaders(
    clinical_csv,
    feature_dir,
    img_dir=None,
    batch_size=32,
    num_workers=4,
    load_images=False,
    max_patches=None
):
    """
    Create dataloaders for precomputed features with transformed labels.

    Args:
        clinical_csv (str/Path): Path to clinical data CSV (must contain 'idx_duration'
            and 'event_transformed' columns from label transformation)
        feature_dir (str/Path): Directory with .npy feature files
        img_dir (str/Path, optional): Directory with images (for interpretability)
        batch_size (int): Batch size
        num_workers (int): Number of workers
        load_images (bool): Load original images
        max_patches (int, optional): Maximum number of patches per patient. If set,
            randomly samples this many patches (good for regularization and speed)

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(clinical_csv)

    train_ds = TCGAFeatureDataset(df, feature_dir, img_dir, split='train', load_images=load_images, max_patches=max_patches)
    val_ds = TCGAFeatureDataset(df, feature_dir, img_dir, split='val', load_images=load_images, max_patches=max_patches)
    test_ds = TCGAFeatureDataset(df, feature_dir, img_dir, split='test', load_images=load_images, max_patches=max_patches)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_mil_batch,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=len(val_ds),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mil_batch,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=len(test_ds),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mil_batch,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

# ============================================================================
# SHAPiQ Utility functions
# ============================================================================

import random
from shapiq import Game
from PIL import ImageDraw, ImageFont


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from io import BytesIO
import urllib.request

# Cache for downloaded emoji images
_EMOJI_IMAGE_CACHE = {}


def _get_emoji_image(emoji: str, size: int = 72) -> Image.Image | None:
    """
    Download emoji image from Twemoji CDN and cache it.

    Args:
        emoji: The emoji character
        size: Desired size (72 is max available)

    Returns:
        PIL Image of the emoji, or None if download fails
    """
    if emoji in _EMOJI_IMAGE_CACHE:
        return _EMOJI_IMAGE_CACHE[emoji]

    try:
        # Convert emoji to Twemoji filename (hex codepoints joined by '-')
        codepoints = '-'.join(f'{ord(c):x}' for c in emoji if ord(c) not in (0xfe0f, 0x200d) or len(emoji) > 1)
        # Handle variation selectors
        codepoints = codepoints.replace('-fe0f', '').replace('fe0f-', '')
        if not codepoints:
            codepoints = f'{ord(emoji[0]):x}'

        # Twemoji CDN URL
        url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/{codepoints}.png"

        # Download the image
        with urllib.request.urlopen(url, timeout=5) as response:
            img_data = response.read()

        img = Image.open(BytesIO(img_data)).convert('RGBA')
        _EMOJI_IMAGE_CACHE[emoji] = img
        return img
    except Exception:
        # Try alternative codepoint format (with fe0f)
        try:
            codepoints = '-'.join(f'{ord(c):x}' for c in emoji)
            url = f"https://cdn.jsdelivr.net/gh/twitter/twemoji@latest/assets/72x72/{codepoints}.png"
            with urllib.request.urlopen(url, timeout=5) as response:
                img_data = response.read()
            img = Image.open(BytesIO(img_data)).convert('RGBA')
            _EMOJI_IMAGE_CACHE[emoji] = img
            return img
        except Exception:
            _EMOJI_IMAGE_CACHE[emoji] = None
            return None


# Mapping for categorical variables (binary features)
CATEGORICAL_ENCODINGS = {
    'prospective_collection': {0: 'Retrospective', 1: 'Prospective'},
    'lumpectomy': {0: 'No', 1: 'Yes'},
    'event': {0: 'Censored', 1: 'Event'},
    'er_status': {0: 'ER-', 1: 'ER+'},
    'pr_status': {0: 'PR-', 1: 'PR+'},
    'her2_status': {0: 'HER2-', 1: 'HER2+'},
    'chemotherapy': {0: 'No Chemo', 1: 'Chemo'},
    'radiation': {0: 'No Radiation', 1: 'Radiation'},
    'hormone_therapy': {0: 'No Hormone Tx', 1: 'Hormone Tx'},
}

# Emoji mapping for feature types
FEATURE_EMOJIS = {
    'age': '🎂',
    'ln': '🔬', 'lymph': '🔬', 'node': '🔬',
    'collection': '📋', 'prospective': '📋',
    'lumpectomy': '🏥', 'surgery': '🏥', 'mastectomy': '🏥',
    'grade': '📊', 'stage': '📊',
    'tumor': '📏', 'size': '📏',
    'receptor': '🧬', 'er': '🧬', 'pr': '🧬', 'her2': '🧬',
    'margin': '✂️',
    'treatment': '💊', 'therapy': '💊', 'chemo': '💊',
    'radiation': '☢️',
    'menopause': '👩',
    'stage': '📊',
    't-stage': '📏', 'n-stage': '📋', 'm-stage': '📊',
    'default': '📋',
}


# ============================================================================
# Feature Groups Configuration
# ============================================================================

# Define feature groups for one-hot encoded categorical variables
# Each group maps a semantic name to the list of individual one-hot columns
FEATURE_GROUPS = {
    'Surgery': ['surgery_lumpectomy', 'surgery_other', 'surgery_simple_mastectomy'],
    'Menopause': ['menopause_indeterminate', 'menopause_pre'],  # post_menopausal is reference
    'T-Stage': ['T_T1', 'T_T3', 'T_T4', 'T_TX'],  # T2 is reference
    'N-Stage': ['N_N1', 'N_N2', 'N_N3', 'N_NX'],  # N0 is reference
    'M-Stage': ['M_M1', 'M_MX'],  # M0 is reference
    'Stage': ['stage_I', 'stage_III', 'stage_IV', 'stage_X'],  # Stage_II is reference
}

# Continuous features (not grouped)
CONTINUOUS_FEATURES = ['age', 'LN_examined']


def get_group_value_for_display(tabular_tensor, feature_groups: dict, tabular_cols: list) -> dict:
    """
    Extract display values for each feature group from a tabular tensor.

    Args:
        tabular_tensor: Tensor of shape [1, n_features] with tabular feature values
        feature_groups: Dict from get_feature_group_mapping()
        tabular_cols: List of tabular column names

    Returns:
        Dict mapping group names to display strings
    """
    import torch
    group_values = {}
    for group_name in feature_groups['group_names']:
        col_indices = feature_groups['group_to_cols'][group_name]
        col_names = [tabular_cols[i] for i in col_indices]

        if group_name in ['age', 'LN_examined']:
            # Continuous features - get the value directly
            val = tabular_tensor[0, col_indices[0]].item()
            if group_name == 'age':
                group_values[group_name] = f"{int(val * 100)} years"
            else:
                group_values[group_name] = f"{int(val * 44)}"
        else:
            # One-hot encoded - find which category is active
            vals = [tabular_tensor[0, i].item() for i in col_indices]
            if max(vals) > 0.5:
                active_idx = vals.index(max(vals))
                active_col = col_names[active_idx]
                group_values[group_name] = active_col.replace('_', ' ')
            else:
                # Reference category (all zeros)
                group_values[group_name] = "(ref)"
    return group_values


def get_feature_group_mapping(feature_cols: list) -> dict:
    """
    Create mapping from individual features to their groups.

    Args:
        feature_cols: List of feature column names

    Returns:
        dict with keys:
            - 'col_to_group': maps column name to group name (or itself if ungrouped)
            - 'group_to_cols': maps group name to list of column indices
            - 'group_names': ordered list of group names (for players)
            - 'n_groups': number of groups (= number of players for tabular)
    """
    col_to_group = {}
    group_to_cols = {}

    # First, map continuous features (each is its own group)
    for col in feature_cols:
        if col in CONTINUOUS_FEATURES:
            col_to_group[col] = col
            group_to_cols[col] = [feature_cols.index(col)]

    # Then, map grouped features
    for group_name, group_cols in FEATURE_GROUPS.items():
        indices = []
        for col in group_cols:
            if col in feature_cols:
                col_to_group[col] = group_name
                indices.append(feature_cols.index(col))
        if indices:  # Only add group if at least one column exists
            group_to_cols[group_name] = indices

    # Handle any remaining ungrouped features
    for i, col in enumerate(feature_cols):
        if col not in col_to_group:
            col_to_group[col] = col
            group_to_cols[col] = [i]

    # Create ordered list of group names
    group_names = list(group_to_cols.keys())

    return {
        'col_to_group': col_to_group,
        'group_to_cols': group_to_cols,
        'group_names': group_names,
        'n_groups': len(group_names)
    }


def get_emoji_for_feature(feature_name: str) -> str:
    """Get appropriate emoji for a feature based on its name."""
    name_lower = feature_name.lower()
    for key, emoji in FEATURE_EMOJIS.items():
        if key == name_lower:
            return emoji
    return FEATURE_EMOJIS['default']


def encode_categorical_value(feature_name: str, value: float) -> str:
    """Encode categorical values to human-readable strings."""
    name_lower = feature_name.lower()

    # Check if this is a known categorical variable
    for key, encoding in CATEGORICAL_ENCODINGS.items():
        if key in name_lower:
            int_val = int(round(value))
            if int_val in encoding:
                return encoding[int_val]

    # Check if it looks like a binary variable (0 or 1)
    if value in [0, 1, 0.0, 1.0]:
        return 'Yes' if value == 1 else 'No'

    return None  # Not categorical


def create_tabular_feature_image(
    feature_name: str,
    feature_value: float,
    size: tuple = (224, 224),
    bg_color: str = "#E8F4FD",
    border_color: str = "#727272",
    text_color: str = "#000000",
    emoji: str = None,
    corner_radius: float = 0.15,
    dpi: int = 100
) -> Image.Image:
    """
    Create a placeholder image for a tabular feature showing name and value.
    Uses matplotlib for proper emoji rendering and rounded corners.

    Args:
        feature_name: Name of the tabular feature
        feature_value: Value of the feature
        size: Image size (width, height)
        bg_color: Background color
        border_color: Border color
        text_color: Text color
        emoji: Optional emoji to display. If None, auto-selects based on feature name.
        corner_radius: Radius of rounded corners (as fraction of smaller dimension)
        dpi: DPI for rendering

    Returns:
        PIL Image with feature name, value and emoji in a rounded box
    """
    # Auto-select emoji based on feature name if not provided
    if emoji is None:
        emoji = get_emoji_for_feature(feature_name)

    # Try to encode categorical value
    categorical_str = encode_categorical_value(feature_name, feature_value)

    # Format value string
    if categorical_str is not None:
        value_str = categorical_str
    elif isinstance(feature_value, (int, np.integer)):
        value_str = str(feature_value)
    else:
        value_str = str(feature_value)

    # Create figure with matplotlib
    fig_width = size[0] / dpi
    fig_height = size[1] / dpi
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw rounded rectangle border
    padding = 0.05
    rect = mpatches.FancyBboxPatch(
        (padding, padding),
        1 - 2*padding,
        1 - 2*padding,
        boxstyle=f"round,pad=0.02,rounding_size={corner_radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=3
    )
    ax.add_patch(rect)

    # Draw emoji (top) - use Twemoji image if available
    emoji_img = _get_emoji_image(emoji)
    if emoji_img is not None:
        # Convert to numpy array for matplotlib
        emoji_arr = np.array(emoji_img)
        # Create OffsetImage and place it
        imagebox = OffsetImage(emoji_arr, zoom=0.55)
        ab = AnnotationBbox(imagebox, (0.5, 0.72), frameon=False,
                           xycoords='axes fraction', box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    else:
        # Fallback to text (may not render correctly)
        ax.text(0.5, 0.72, emoji, fontsize=36, ha='center', va='center',
                transform=ax.transAxes)

    # Draw feature name (middle)
    # Clean up feature name for display
    display_name = feature_name.replace('_', ' ').title()
    ax.text(0.5, 0.4, display_name, fontsize=15, ha='center', va='center',
            transform=ax.transAxes, color=text_color, fontweight='bold')

    # Draw value (bottom)
    ax.text(0.5, 0.18, value_str, fontsize=17, ha='center', va='center',
            transform=ax.transAxes, color=text_color)

    # Convert matplotlib figure to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                edgecolor='none')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    plt.close(fig)

    # Resize to exact size if needed
    if img.size != size:
        img = img.resize(size, Image.Resampling.LANCZOS)

    return img


def load_patch_images_for_shapiq(
    patient_id: str,
    top_k_indices: np.ndarray,
    tabular_features: dict,
    img_dir: Path = Path('./data/processed_images'),
    image_size: tuple = (224, 224),
    show_rank: bool = True,
    attention_weights: np.ndarray = None,
    title_height: int = 56,
    title_font_size: int = 24
) -> list:
    """
    Load patch images and create tabular feature images for shapiq plot_network.

    Args:
        patient_id: Patient ID for loading patch images
        top_k_indices: Indices of top K patches to load (assumed sorted by attention, highest first)
        tabular_features: Dictionary mapping feature names to values, e.g.
                         {'age': 65, 'lumpectomy': 1, 'LN_examined': 12}
        img_dir: Directory containing patient image folders
        image_size: Size to resize images to
        show_rank: If True, display attention rank as centered title above patch
        attention_weights: Optional array of attention weights for all patches.
                          If provided, also shows attention percentage in title.
        title_height: Height of the title area above the patch (default: 56)
        title_font_size: Font size for the title text (default: 24)

    Returns:
        List of PIL Images: [patch_1, patch_2, ..., patch_k, tabular_1, tabular_2, ...]
    """
    images = []
    patient_img_dir = img_dir / patient_id

    # Try to load fonts for the title
    title_font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in font_paths:
        try:
            title_font = ImageFont.truetype(fp, title_font_size)
            break
        except (OSError, IOError):
            continue

    # Load patch images
    for rank, patch_idx in enumerate(top_k_indices, start=1):
        patch_path = patient_img_dir / f"{patch_idx}.jpg"
        if patch_path.exists():
            patch_img = Image.open(patch_path).convert('RGB')
            if patch_img.size != image_size:
                patch_img = patch_img.resize(image_size, Image.Resampling.LANCZOS)
        else:
            # Create placeholder for missing patch
            patch_img = Image.new('RGB', image_size, '#CCCCCC')
            draw = ImageDraw.Draw(patch_img)
            draw.text((image_size[0]//2, image_size[1]//2), f"Patch {patch_idx}\n(missing)",
                     fill='#666666', anchor="mm")

        # Add centered title above patch if requested
        if show_rank:
            # Create new image with space for title
            total_height = image_size[1] + title_height
            img_with_title = Image.new('RGB', (image_size[0], total_height), '#FFFFFF')

            # Paste the patch below the title area
            img_with_title.paste(patch_img, (0, title_height))

            # Draw title text
            draw = ImageDraw.Draw(img_with_title)

            # Build title text
            if attention_weights is not None:
                attn_value = attention_weights[patch_idx]
                title_text = f"#{rank}  {attn_value:.1%}"
            else:
                title_text = f"#{rank}"

            # Draw centered title
            draw.text(
                (image_size[0] // 2, title_height // 2),
                title_text,
                fill="#333333",
                font=title_font,
                anchor="mm"
            )

            images.append(img_with_title)
        else:
            images.append(patch_img)

    # Create tabular feature images
    for feature_name, feature_value in tabular_features.items():
        tab_img = create_tabular_feature_image(feature_name, feature_value, size=image_size)
        images.append(tab_img)

    return images


class SurvivalGame(Game):
    """ShapIQ game for MIL survival with multiple baseline strategies.

    Players: Each patch + each tabular feature (or feature group)
    Value: Predicted survival probability or cumulative mortality at time_idx

    Baseline strategies:
    - 'zero': Features not in coalition = zero vectors
    - 'mean': Features not in coalition = global mean from all training patients
    - 'marginal': Marginalize over background feature pools (Feature-Level Marginal SHAP)

    Prediction methods:
    - 'predict_surv': Survival probability S(t) - higher means better survival
    - 'predict_pmf': Probability mass function F(t) - higher means worse outcome

    Feature Groups:
    - If feature_groups is provided, tabular features are grouped and treated as single players
    - feature_groups should be a dict from get_feature_group_mapping()
    """

    def __init__(self, pycox_model, x,
                 x_other_patches = None,
                 time_idx=5,
                 baseline_patch='zero',
                 baseline_tabular='zero',
                 data_patch=None,
                 data_tabular=None,
                 n_background_samples=50, # Number of Monte Carlo samples for marginal
                 prediction_method='predict_surv',  # 'predict_surv' or 'predict_pmf'
                 device='cuda',
                 verbose=True,
                 seed = 42,
                 feature_groups=None):  # Optional: dict from get_feature_group_mapping()
        self.pycox_model = pycox_model  # DeepHitSingle model
        self.device = device
        self.time_idx = time_idx
        self.n_background_samples = n_background_samples
        self.x_patches = x[0].shape[1]
        self.seed = seed
        self.feature_groups = feature_groups

        if verbose:
            print(f"Initializing MILSurvivalGame with parameters:")
            print(f" - Time index for prediction: {time_idx}")
            print(f" - Prediction method: {prediction_method}")
            print(f" - Number of background samples for marginal: {n_background_samples}")
            print(f" - Number of patch features in instance: {self.x_patches}")

        # Store prediction method
        if prediction_method == 'predict_surv':
            self.predict_fn = self.pycox_model.predict_surv
            self.prediction_name = 'survival probability'
        elif prediction_method == 'predict_pmf':
            self.predict_fn = self.pycox_model.predict_pmf
            self.prediction_name = 'probability mass function'
        else:
            raise ValueError(f"Unknown prediction_method: {prediction_method}. Choose 'predict_surv' or 'predict_pmf'")

        # Instance to be explained
        self.x = x  # Tuple: (patch_features, tabular_features)
        self.x_other_patches = x_other_patches

        # Calculate number of players (accounting for feature groups)
        n_tabular_raw = x[1].shape[1]  # Number of individual tabular features
        if feature_groups is not None:
            n_tabular_players = feature_groups['n_groups']
            self.group_to_cols = feature_groups['group_to_cols']
            self.group_names = feature_groups['group_names']
            if verbose:
                print(f" - Feature groups enabled: {n_tabular_raw} features -> {n_tabular_players} groups")
                for name in self.group_names:
                    print(f"     {name}: columns {self.group_to_cols[name]}")
        else:
            n_tabular_players = n_tabular_raw
            self.group_to_cols = None
            self.group_names = None

        self.n_tabular_raw = n_tabular_raw
        self.n_tabular_players = n_tabular_players
        n_players = x[0].shape[1] + n_tabular_players

        if verbose:
            print(f" - Number of players (patches + tabular): {n_players} ({x[0].shape[1]} patches + {n_tabular_players} tabular)")

        # Baseline setup (patch features)
        if baseline_patch == 'zero':
            self.baseline_patch = torch.zeros((1,1,x[0].shape[2])).to(device)
            self.baseline_patch_mode = 'zero'
            if verbose:
                print(f" - Patch Baseline: zero (features not in coalition = zero vectors)")
        elif baseline_patch == 'mean_all':
            self.baseline_patch = data_patch.mean(dim = (0,1), keepdim = True).to(device)  # Global mean over all patches
            self.baseline_patch_mode = 'mean_all'
            if verbose:
                print(f" - Patch Baseline: mean_all (features not in coalition = population average over all patches)")
        elif baseline_patch == 'mean_patient':
            self.baseline_patch = x_other_patches.mean(dim = 1, keepdim = True).to(device)  # Mean over other patches of same patient
            self.baseline_patch_mode = 'mean_patient'
            if verbose:
                print(f" - Patch Baseline: mean_patient (features not in coalition = patient average over other patches)")
        elif baseline_patch == 'marginal_patient':
            self.baseline_patch = x_other_patches.to(device)  # Pool of patch features for marginal sampling
            self.baseline_patch_mode = 'marginal_patient'
            if verbose:
                print(f" - Patch Baseline: marginal_patient (feature-level sampling from {x_other_patches.shape[1]:,} patches of same patient)")
        elif baseline_patch == 'marginal_all':
            self.baseline_patch = data_patch.to(device)  # Pool of patch features for marginal sampling
            self.baseline_patch_mode = 'marginal_all'
            if verbose:
                print(f" - Patch Baseline: marginal_all (feature-level sampling from {len(data_patch):,} patches)")
        else:
            raise ValueError(f"Unknown baseline_patch: {baseline_patch}. Choose from: 'zero', 'mean_all', 'mean_patient', 'marginal_patient', 'marginal_all'")
        

        # Baseline setup (tabular features)
        if baseline_tabular == 'zero':
            self.baseline_tabular = torch.zeros_like(x[1]).to(device)
            self.baseline_tabular_mode = 'zero'
            if verbose:
                print(f" - Tabular Baseline: zero (features not in coalition = zero vectors)")
        elif baseline_tabular == 'mean':
            self.baseline_tabular = data_tabular.mean(dim = 0, keepdim=True).to(device)
            self.baseline_tabular_mode = 'mean'
            if verbose:
                print(f" - Tabular Baseline: mean (features not in coalition = population average)")
        elif baseline_tabular == 'marginal':
            self.baseline_tabular = data_tabular.to(device)  # Pool of tabular features for marginal sampling
            self.baseline_tabular_mode = 'marginal'
            if verbose:
                print(f" - Tabular Baseline: marginal (feature-level sampling from {len(data_tabular):,} instances)")
        else:
            raise ValueError(f"Unknown baseline_tabular: {baseline_tabular}. Choose from: 'zero', 'mean', 'marginal'")
       
        # Compute empty value
        torch.manual_seed(seed)
        if self.baseline_tabular_mode in ['zero', 'mean'] and self.baseline_patch_mode in ['zero', 'mean_all', 'mean_patient']:
            with torch.no_grad():
                empty_input = (self.baseline_patch, self.baseline_tabular)
                pred = self.predict_fn(input=empty_input)
            empty_value = float(pred[0, time_idx].cpu().item())
        elif self.baseline_tabular_mode == 'marginal' and self.baseline_patch_mode in ['zero', 'mean_all', 'mean_patient']:
            batch_patch = self.baseline_patch.repeat(self.baseline_tabular.shape[0], 1, 1)
            batch_tabular = self.baseline_tabular
            with torch.no_grad():
                empty_input = (batch_patch, batch_tabular)
                pred = self.predict_fn(input=empty_input)
            empty_value = float(pred[0, time_idx].cpu().item())
        elif self.baseline_tabular_mode in ['zero', 'mean'] and self.baseline_patch_mode in ['marginal_patient', 'marginal_all']:
            num_patches = self.x[0].shape[1]
            max_samples = min(self.n_background_samples, self.baseline_patch.shape[0])

            batch_patch = self.baseline_patch[
                torch.randint(0, self.baseline_patch.shape[0], size=(max_samples, 1)),
                torch.randint(0, self.baseline_patch.shape[1], size=(max_samples, num_patches)),
                :
            ]
            batch_tabular = self.baseline_tabular.repeat(batch_patch.shape[0], 1)
            with torch.no_grad():
                empty_input = (batch_patch, batch_tabular)
                pred = self.predict_fn(input=empty_input)
            empty_value = float(pred.mean(dim = 0)[time_idx].cpu().item())
        else: # both marginal
            max_samples = min(self.n_background_samples, self.baseline_tabular.shape[0])
            sample_idx = torch.randperm(len(self.baseline_tabular))[:max_samples]
            batch_tabular = self.baseline_tabular[sample_idx]
            num_patches = self.x[0].shape[1]
            if self.baseline_patch_mode == 'marginal_patient':
                batch_patch = self.baseline_patch[
                    0,  # Since baseline_patch is from same patient
                    torch.randint(0, self.baseline_patch.shape[1], size=(max_samples, num_patches)),
                    :
                ]
            else:
                batch_patch = self.baseline_patch[
                    sample_idx[:, None],
                    torch.randint(0, self.baseline_patch.shape[1], size=(max_samples, num_patches)),
                    :
                ]
            with torch.no_grad():
                empty_input = (batch_patch, batch_tabular)
                pred = self.predict_fn(input=empty_input)
            empty_value = float(pred.mean(dim = 0)[time_idx].cpu().item())
        if verbose:
            print(f" - Empty coalition value (baseline prediction at time index {time_idx}): {empty_value:.6f} ({self.prediction_name})")
        
        super().__init__(n_players=n_players, normalize=True, normalization_value=empty_value)
    
    def _expand_group_coalition_to_features(self, group_coalition: np.ndarray) -> np.ndarray:
        """Expand group-level coalition to individual feature mask.

        Args:
            group_coalition: Binary array of length n_tabular_players (groups)

        Returns:
            Binary array of length n_tabular_raw (individual features)
        """
        if self.feature_groups is None:
            return group_coalition

        feature_mask = np.zeros(self.n_tabular_raw, dtype=bool)
        for i, group_name in enumerate(self.group_names):
            if group_coalition[i]:
                # Include all features in this group
                for col_idx in self.group_to_cols[group_name]:
                    feature_mask[col_idx] = True
        return feature_mask

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        n_patches = self.x[0].shape[1]
        values = []

        for coalition in coalitions:

            torch.manual_seed(self.seed)  # For reproducibility

            # Get mask for tabular features (expand groups if needed)
            tabular_coalition = coalition[n_patches:]
            if self.feature_groups is not None:
                # Expand group-level coalition to individual features
                tabular_mask = self._expand_group_coalition_to_features(tabular_coalition)
            else:
                tabular_mask = tabular_coalition

            sample_idx = torch.randperm(self.baseline_tabular.shape[0])[:min(self.n_background_samples, self.baseline_tabular.shape[0])]
            baseline_tabular_sample = self.baseline_tabular[sample_idx]

            masked_tabular = torch.where(
                            torch.tensor(tabular_mask, dtype=torch.bool, device=self.device),
                            self.x[1],
                            baseline_tabular_sample
                        )
            
            # Get mask for patch features
            patch_mask = coalition[:n_patches]
            if self.baseline_patch_mode != "marginal_all":
                sample_idx = torch.tensor([0])
            elif self.baseline_tabular_mode != "marginal":
                sample_idx = torch.randperm(self.baseline_patch.shape[0])[:min(self.n_background_samples, self.baseline_patch.shape[0])]
            
            batch_patch = self.baseline_patch[
                sample_idx[:, None],
                torch.randint(0, self.baseline_patch.shape[1], size=(len(sample_idx), n_patches)),
                :
            ]
            masked_patches = torch.where(
                            torch.tensor(patch_mask, dtype=torch.bool, device=self.device).unsqueeze(1),
                            self.x[0],
                            batch_patch
                        )
            
            # Check shapes
            if masked_patches.shape[0] < masked_tabular.shape[0]:
                masked_patches = masked_patches.repeat(masked_tabular.shape[0], 1, 1)
            elif masked_tabular.shape[0] < masked_patches.shape[0]:
                masked_tabular = masked_tabular.repeat(masked_patches.shape[0], 1)

            with torch.no_grad():
                input_tuple = (masked_patches, masked_tabular)
                pred = self.predict_fn(input=input_tuple)
                value = pred.mean(dim = 0)[self.time_idx].cpu().item()
            values.append(value)

        return np.array(values)