#!/usr/bin/env python3
"""
VideoMAE UCF-101 Classification Pipeline Architecture Diagram
This script generates a comprehensive architecture diagram showing the complete
VideoMAE pipeline for UCF-101 video classification.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_videomae_architecture_diagram():
    """Create a comprehensive VideoMAE UCF-101 classification architecture diagram"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#FFF2CC', 
        'backbone': '#E1D5E7',
        'attention': '#D5E8D4',
        'classification': '#FFE6CC',
        'output': '#F8CECC',
        'arrow': '#666666'
    }
    
    # Title
    ax.text(10, 15.5, 'VideoMAE UCF-101 Classification Pipeline Architecture', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # ================== INPUT STAGE ==================
    # Video Input
    video_box = FancyBboxPatch((0.5, 13.5), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['input'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(video_box)
    ax.text(2, 14.1, 'UCF-101 Video\n(16 frames, 224×224×3)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ================== PREPROCESSING STAGE ==================
    # Video Loading & Sampling
    load_box = FancyBboxPatch((5, 13.5), 3, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['preprocessing'],
                              edgecolor='black', linewidth=1)
    ax.add_patch(load_box)
    ax.text(6.5, 14.1, 'Video Loading\n• Decord VideoReader\n• Frame Sampling (rate=4)', 
            ha='center', va='center', fontsize=9)
    
    # Data Augmentation
    aug_box = FancyBboxPatch((0.5, 11.8), 3.5, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['preprocessing'],
                             edgecolor='black', linewidth=1)
    ax.add_patch(aug_box)
    ax.text(2.25, 12.4, 'Data Augmentation\n• GroupMultiScaleCrop\n• Stack & ToTensor\n• Normalize', 
            ha='center', va='center', fontsize=9)
    
    # Tube Masking
    mask_box = FancyBboxPatch((4.5, 11.8), 3.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['preprocessing'],
                              edgecolor='black', linewidth=1)
    ax.add_patch(mask_box)
    ax.text(6.25, 12.4, 'Tube Masking\n• TubeMaskingGenerator\n• Mask Ratio: 75%\n• Window Size: (8,14,14)', 
            ha='center', va='center', fontsize=9)
    
    # ================== PATCH EMBEDDING ==================
    patch_box = FancyBboxPatch((9, 12.5), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['backbone'],
                               edgecolor='black', linewidth=2)
    ax.add_patch(patch_box)
    ax.text(11, 13.25, 'Patch Embedding\n• 3D Conv (tubelet_size=2)\n• Kernel: (2,16,16)\n• Output: 768-dim tokens', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ================== POSITIONAL EMBEDDING ==================
    pos_box = FancyBboxPatch((14, 12.5), 3, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['backbone'],
                             edgecolor='black', linewidth=1)
    ax.add_patch(pos_box)
    ax.text(15.5, 13.25, 'Positional\nEmbedding\n• Sinusoidal/Learnable\n• Shape: (1,1568,768)', 
            ha='center', va='center', fontsize=9)
    
    # ================== TRANSFORMER ENCODER ==================
    # Vision Transformer Backbone
    vit_box = FancyBboxPatch((1, 9.5), 16, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['backbone'],
                             edgecolor='black', linewidth=3)
    ax.add_patch(vit_box)
    ax.text(9, 10.4, 'Vision Transformer Encoder (ViT-Base)\n12 Layers × 12 Heads × 768 Dimensions', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Individual Transformer Blocks
    for i in range(4):
        block_x = 2 + i * 3.5
        
        # Transformer Block
        block_box = FancyBboxPatch((block_x, 7.5), 3, 1.5,
                                   boxstyle="round,pad=0.05",
                                   facecolor=colors['attention'],
                                   edgecolor='black', linewidth=1)
        ax.add_patch(block_box)
        
        if i == 0:
            ax.text(block_x + 1.5, 8.7, 'Multi-Head\nAttention', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(block_x + 1.5, 8.0, '12 heads\n64 dim/head', 
                    ha='center', va='center', fontsize=7)
        elif i == 1:
            ax.text(block_x + 1.5, 8.7, 'Add & Norm', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(block_x + 1.5, 8.0, 'Residual +\nLayerNorm', 
                    ha='center', va='center', fontsize=7)
        elif i == 2:
            ax.text(block_x + 1.5, 8.7, 'Feed Forward\nNetwork', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(block_x + 1.5, 8.0, '768→3072→768\nGELU activation', 
                    ha='center', va='center', fontsize=7)
        else:
            ax.text(block_x + 1.5, 8.7, 'Add & Norm', 
                    ha='center', va='center', fontsize=8, fontweight='bold')
            ax.text(block_x + 1.5, 8.0, 'Residual +\nLayerNorm', 
                    ha='center', va='center', fontsize=7)
    
    # Attention visualization
    attn_box = FancyBboxPatch((2, 5.8), 13, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['attention'],
                              edgecolor='black', linewidth=1)
    ax.add_patch(attn_box)
    ax.text(8.5, 6.4, 'Self-Attention Mechanism: Q, K, V ∈ ℝ^(1568×768)\nAttention(Q,K,V) = softmax(QK^T/√d)V', 
            ha='center', va='center', fontsize=10)
    
    # ================== CLASSIFICATION HEAD ==================
    # Global Average Pooling / CLS Token
    pool_box = FancyBboxPatch((1, 4.2), 4, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['classification'],
                              edgecolor='black', linewidth=1)
    ax.add_patch(pool_box)
    ax.text(3, 4.8, 'Feature Aggregation\n• Mean Pooling or\n• CLS Token Selection', 
            ha='center', va='center', fontsize=9)
    
    # Layer Normalization
    norm_box = FancyBboxPatch((6, 4.2), 3, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['classification'],
                              edgecolor='black', linewidth=1)
    ax.add_patch(norm_box)
    ax.text(7.5, 4.8, 'Layer Norm\n& Dropout\n(p=0.5)', 
            ha='center', va='center', fontsize=9)
    
    # Classification Head
    head_box = FancyBboxPatch((10, 4.2), 4, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['classification'],
                              edgecolor='black', linewidth=2)
    ax.add_patch(head_box)
    ax.text(12, 4.8, 'Classification Head\nLinear: 768 → 101\n(UCF-101 Classes)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ================== OUTPUT STAGE ==================
    # Softmax & Output
    output_box = FancyBboxPatch((15, 4.2), 4, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['output'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(17, 4.8, 'Output\nSoftmax → Probabilities\n101 Action Classes', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # ================== TRAINING COMPONENTS ==================
    # Loss Function
    loss_box = FancyBboxPatch((1, 2.5), 4, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor='#F0F0F0',
                              edgecolor='black', linewidth=1)
    ax.add_patch(loss_box)
    ax.text(3, 3.1, 'Training Loss\nCross Entropy Loss\n+ Label Smoothing', 
            ha='center', va='center', fontsize=9)
    
    # Optimizer
    opt_box = FancyBboxPatch((6, 2.5), 4, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor='#F0F0F0',
                             edgecolor='black', linewidth=1)
    ax.add_patch(opt_box)
    ax.text(8, 3.1, 'Optimizer\nAdamW (lr=5e-4)\nCosine LR Schedule\nWeight Decay=0.05', 
            ha='center', va='center', fontsize=9)
    
    # Evaluation Metrics
    metrics_box = FancyBboxPatch((11, 2.5), 4, 1.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#F0F0F0',
                                 edgecolor='black', linewidth=1)
    ax.add_patch(metrics_box)
    ax.text(13, 3.1, 'Evaluation\nTop-1 & Top-5 Accuracy\nConfusion Matrix\nPer-class F1 Score', 
            ha='center', va='center', fontsize=9)
    
    # ================== MODEL SPECIFICATIONS ==================
    spec_box = FancyBboxPatch((16, 2.5), 3.5, 1.2,
                              boxstyle="round,pad=0.1",
                              facecolor='#E6F3FF',
                              edgecolor='black', linewidth=1)
    ax.add_patch(spec_box)
    ax.text(17.75, 3.1, 'Model Specs\n• Parameters: ~86M\n• Input: 16×224×224×3\n• Patches: 1568\n• Classes: 101', 
            ha='center', va='center', fontsize=9)
    
    # ================== ARROWS ==================
    arrows = [
        # Main pipeline flow
        ((3.5, 14.1), (5, 14.1)),  # Video to Loading
        ((6.5, 13.5), (6.25, 13)),  # Loading to augmentation/masking
        ((8, 12.4), (9, 13.25)),  # Preprocessing to patch embedding
        ((13, 13.25), (14, 13.25)),  # Patch to positional embedding
        ((11, 12.5), (9, 11.3)),  # Patch embedding to transformer
        ((9, 9.5), (8.5, 7)),  # Transformer to attention detail
        ((8.5, 5.8), (3, 5.4)),  # Attention to pooling
        ((5, 4.8), (6, 4.8)),  # Pooling to norm
        ((9, 4.8), (10, 4.8)),  # Norm to head
        ((14, 4.8), (15, 4.8)),  # Head to output
        
        # Training flow
        ((17, 4.2), (13, 3.7)),  # Output to metrics (feedback)
        ((3, 2.5), (8, 2.5)),  # Loss to optimizer (horizontal)
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc=colors['arrow'], 
                               ec=colors['arrow'], linewidth=2)
        ax.add_patch(arrow)
    
    # ================== LEGEND ==================
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Data'),
        patches.Patch(color=colors['preprocessing'], label='Preprocessing'),
        patches.Patch(color=colors['backbone'], label='Model Backbone'),
        patches.Patch(color=colors['attention'], label='Attention Mechanism'),
        patches.Patch(color=colors['classification'], label='Classification'),
        patches.Patch(color=colors['output'], label='Output'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 1.5), fontsize=10)
    
    # ================== ADDITIONAL ANNOTATIONS ==================
    # Dataset info
    ax.text(1, 1.5, 'UCF-101 Dataset:', fontweight='bold', fontsize=10)
    ax.text(1, 1.2, '• 101 action classes', fontsize=9)
    ax.text(1, 0.9, '• ~13,320 videos', fontsize=9)
    ax.text(1, 0.6, '• 25 fps, variable length', fontsize=9)
    ax.text(1, 0.3, '• 3-fold cross-validation', fontsize=9)
    
    # Training details
    ax.text(6, 1.5, 'Training Configuration:', fontweight='bold', fontsize=10)
    ax.text(6, 1.2, '• Batch size: 16 (8 GPUs)', fontsize=9)
    ax.text(6, 0.9, '• Epochs: 100', fontsize=9)
    ax.text(6, 0.6, '• Warmup: 5 epochs', fontsize=9)
    ax.text(6, 0.3, '• Mixed precision (FP16)', fontsize=9)
    
    # Performance
    ax.text(12, 1.5, 'Expected Performance:', fontweight='bold', fontsize=10)
    ax.text(12, 1.2, '• Top-1 Accuracy: ~91.3%', fontsize=9)
    ax.text(12, 0.9, '• Top-5 Accuracy: ~99.4%', fontsize=9)
    ax.text(12, 0.6, '• Inference time: ~50ms', fontsize=9)
    ax.text(12, 0.3, '• Model size: ~340MB', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_attention_detail_diagram():
    """Create a detailed diagram of the attention mechanism"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.text(7, 9.5, 'VideoMAE Multi-Head Self-Attention Mechanism', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input tokens
    input_box = FancyBboxPatch((1, 7.5), 3, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#E8F4FD',
                               edgecolor='black', linewidth=1)
    ax.add_patch(input_box)
    ax.text(2.5, 8.1, 'Input Tokens\nX ∈ ℝ^(1568×768)', ha='center', va='center', fontsize=10)
    
    # Linear projections
    qkv_boxes = []
    labels = ['Q = XW_Q', 'K = XW_K', 'V = XW_V']
    colors = ['#FFE6CC', '#D5E8D4', '#E1D5E7']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        box = FancyBboxPatch((5.5 + i*2.5, 7.5), 2, 1.2,
                             boxstyle="round,pad=0.1",
                             facecolor=color,
                             edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(6.5 + i*2.5, 8.1, label, ha='center', va='center', fontsize=10)
    
    # Multi-head split
    ax.text(7, 6.8, 'Split into 12 heads: 768 = 12 × 64', 
            ha='center', va='center', fontsize=10, style='italic')
    
    # Attention computation
    attn_box = FancyBboxPatch((3, 5), 8, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor='#F0F8FF',
                              edgecolor='black', linewidth=2)
    ax.add_patch(attn_box)
    ax.text(7, 5.75, 'Attention(Q,K,V) = softmax(QK^T/√64)V\nfor each of 12 heads in parallel', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Concatenate and project
    concat_box = FancyBboxPatch((3, 3), 8, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#F5F5DC',
                                edgecolor='black', linewidth=1)
    ax.add_patch(concat_box)
    ax.text(7, 3.6, 'Concatenate heads and project: Concat(head₁,...,head₁₂)W_O', 
            ha='center', va='center', fontsize=10)
    
    # Output
    output_box = FancyBboxPatch((5, 1.5), 4, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F4FD',
                                edgecolor='black', linewidth=1)
    ax.add_patch(output_box)
    ax.text(7, 2.1, 'Output\nℝ^(1568×768)', ha='center', va='center', fontsize=10)
    
    # Arrows
    arrows = [
        ((4, 8.1), (5.5, 8.1)),
        ((7, 7.5), (7, 6.5)),
        ((7, 6.3), (7, 6.5)),
        ((7, 5), (7, 4.2)),
        ((7, 3), (7, 2.7))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc='black', ec='black')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig

def create_masking_strategy_diagram():
    """Create a diagram showing the tube masking strategy"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.text(6, 7.5, 'VideoMAE Tube Masking Strategy', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Original video
    ax.text(2, 6.5, 'Original Video\n(16 frames)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw video frames
    for t in range(4):
        for i in range(3):
            for j in range(3):
                rect = patches.Rectangle((0.5 + j*0.3, 5.5 - i*0.3 + t*0.1), 0.25, 0.25,
                                       linewidth=1, edgecolor='black', facecolor='lightblue')
                ax.add_patch(rect)
    
    # Masked video
    ax.text(6, 6.5, 'Tube Masked Video\n(75% masked)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw masked frames
    mask_pattern = np.random.random((4, 3, 3)) > 0.25  # 75% masking
    for t in range(4):
        for i in range(3):
            for j in range(3):
                color = 'lightblue' if mask_pattern[t, i, j] else 'gray'
                rect = patches.Rectangle((4.5 + j*0.3, 5.5 - i*0.3 + t*0.1), 0.25, 0.25,
                                       linewidth=1, edgecolor='black', facecolor=color)
                ax.add_patch(rect)
    
    # Reconstruction target
    ax.text(10, 6.5, 'Reconstruction Target\n(Masked patches only)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw target
    for t in range(4):
        for i in range(3):
            for j in range(3):
                color = 'lightcoral' if not mask_pattern[t, i, j] else 'white'
                alpha = 1.0 if not mask_pattern[t, i, j] else 0.3
                rect = patches.Rectangle((8.5 + j*0.3, 5.5 - i*0.3 + t*0.1), 0.25, 0.25,
                                       linewidth=1, edgecolor='black', facecolor=color, alpha=alpha)
                ax.add_patch(rect)
    
    # Explanation
    ax.text(6, 4, 'Tube Masking Properties:', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(6, 3.5, '• Temporal consistency: entire tubes (spatiotemporal patches) are masked', ha='center', va='center', fontsize=11)
    ax.text(6, 3.1, '• High masking ratio (75-90%) forces model to learn strong representations', ha='center', va='center', fontsize=11)
    ax.text(6, 2.7, '• Reconstruction task: predict RGB values of masked patches', ha='center', va='center', fontsize=11)
    ax.text(6, 2.3, '• Window size: (T=8, H=14, W=14) patches', ha='center', va='center', fontsize=11)
    
    # Arrows
    arrow1 = ConnectionPatch((2.5, 5.2), (4, 5.2), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc='red', ec='red', linewidth=2)
    ax.add_patch(arrow1)
    ax.text(3.25, 5.4, 'Mask', ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    arrow2 = ConnectionPatch((6.5, 5.2), (8, 5.2), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc='green', ec='green', linewidth=2)
    ax.add_patch(arrow2)
    ax.text(7.25, 5.4, 'Reconstruct', ha='center', va='center', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create and save the main architecture diagram
    print("Creating VideoMAE UCF-101 Architecture Diagram...")
    fig1 = create_videomae_architecture_diagram()
    fig1.savefig('videomae_ucf101_architecture.png', dpi=300, bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
    print("Main architecture diagram saved as 'videomae_ucf101_architecture.png'")
    
    # Create and save the attention mechanism diagram
    print("Creating Attention Mechanism Diagram...")
    fig2 = create_attention_detail_diagram()
    fig2.savefig('videomae_attention_mechanism.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("Attention mechanism diagram saved as 'videomae_attention_mechanism.png'")
    
    # Create and save the masking strategy diagram
    print("Creating Tube Masking Strategy Diagram...")
    fig3 = create_masking_strategy_diagram()
    fig3.savefig('videomae_tube_masking.png', dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print("Tube masking diagram saved as 'videomae_tube_masking.png'")
    
    plt.show()
    
    print("\nArchitecture Summary:")
    print("=" * 50)
    print("1. Input: UCF-101 videos (16 frames, 224×224×3)")
    print("2. Preprocessing: Video loading, augmentation, tube masking")
    print("3. Patch Embedding: 3D convolution (tubelet_size=2)")
    print("4. Transformer: 12 layers, 12 heads, 768 dimensions")
    print("5. Classification: Linear layer (768 → 101 classes)")
    print("6. Training: AdamW optimizer, cosine LR schedule")
    print("7. Performance: ~91.3% top-1 accuracy on UCF-101")
    print("=" * 50)
